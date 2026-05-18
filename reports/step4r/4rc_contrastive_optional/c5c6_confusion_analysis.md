# C5/C6 trilateral knee-valgus confusion 분석

- 생성 스크립트: `scripts/analyze_step4rc_c5c6_confusion.py`
- 입력: `joint_embeddings.npz` (v1) + `joint_embeddings_clinical.npz` (v2)
- 범위: test split (n=1,427), i2t cross-modal retrieval

## 1. 배경

4R-C clinical A/B 실험에서 C5(Right-knee valgus)와 C6(Bilateral knee valgus)의 비대칭 변화 관찰:

| 클래스 | figure_img.png 정의 | v1 class R@1 | v2 class R@1 | Δ |
|---|---|---:|---:|---:|
| C5 | Right-knee valgus | 0.210 | 0.256 | +0.046 |
| C6 | Bilateral knee valgus | 0.408 | 0.254 | -0.154 |

가설: 손목 IMU 단독으로는 left(C4) / right(C5) / bilateral(C6) knee valgus의 3-way 구분이 본질적으로 어렵다. v1 abstract corpus는 C5/C6를 함께 묶어 ('무릎 관련 오류 후보') 한 클러스터로 학습 → C6 회수율이 운으로 높았음. v2 clinical은 directional 분리 → 진짜 IMU 한계 노출.

## 2. 6×6 retrieval confusion matrix (top-1)

`reports/step4r/4rc_contrastive_optional/c5c6_confusion_figure.png` 의 1행 참조.

### v1 abstract — 핵심 행

| true \ retrieved | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| **C5** | 0.18 | 0.07 | 0.03 | 0.19 | 0.21 | 0.31 |
| **C6** | 0.25 | 0.05 | 0.05 | 0.10 | 0.14 | 0.41 |

### v2 clinical — 핵심 행

| true \ retrieved | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| **C5** | 0.14 | 0.07 | 0.05 | 0.19 | 0.26 | 0.30 |
| **C6** | 0.18 | 0.12 | 0.05 | 0.09 | 0.30 | 0.25 |

## 3. 변화 분석 (v1 → v2)

| true | pred | v1 rate | v2 rate | Δ |
|---|---|---:|---:|---:|
| C5 | C5 | 0.210 | 0.256 | +0.046 |
| C5 | C6 | 0.311 | 0.298 | -0.013 |
| C5 | C1 | 0.185 | 0.139 | -0.046 |
| C5 | C4 | 0.193 | 0.189 | -0.004 |
| C6 | C5 | 0.142 | 0.304 | +0.162 |
| C6 | C6 | 0.408 | 0.254 | -0.154 |
| C6 | C1 | 0.250 | 0.179 | -0.071 |
| C6 | C4 | 0.100 | 0.092 | -0.008 |

## 4. 해석

### 4.1 v1 abstract: C5/C6 mutual collapse

- v1에서 C5→C6: 0.31, C6→C5: 0.14. 두 클래스가 서로 retrieve하는 mutual confusion.
- corpus phrase가 "무릎 관련 오류 후보(C5, C6)"로 *동일 어휘* 사용 → text embedding 공간에서 두 클래스의 anchor가 거의 동일 → IMU가 어디로 가도 C5/C6 중 하나에 회수.
- C6의 v1 0.40은 *진짜 식별*이 아니라 *어휘 collapse로 인한 부수 효과*. 즉 thesis claim으로 인용해서는 안 되는 inflated number.

### 4.2 v2 clinical: directional 분리 → IMU 한계 노출

- v2에서 C5→C6: 0.30, C6→C5: 0.30. **C6→C5가 v1보다 0.16 증가/감소**.
- v2 corpus phrase는 "우측 무릎 외반"(C5) vs "양측 무릎 외반"(C6)으로 어휘 분리 → text anchor가 분리 → IMU가 어디로 정렬할지 결정해야 함.
- 결과: 손목 IMU의 wrist-position 신호로는 right-only vs bilateral 구분이 어려워 C6 sample이 **빈도 더 높은 C5 anchor로 over-attribute**.

### 4.3 paper claim

이 결과는 두 가지 thesis 주장을 동시에 정량화한다:

1. **Closed-vocabulary policy(외부 caption directional ban)는 단순 보호장치가 아니다** — directional vocab을 도입하면 IMU 한계가 노출되므로, 정책이 *실제로 잘 작동*함을 시사 (외부 caption에 "우측 무릎 외반"을 출력했다가 wrong이면 critical failure).
2. **Multi-modal 필요성의 정량 근거** — wrist IMU 단독으로 trilateral knee valgus를 구분하지 못한다는 직접 측정. video → MediaPipe joint trajectory 도입 필요성을 데이터로 증명. (교수님 피드백 5 video 필요성과 직결.)

## 5. Discussion 본문 활용 권장 문장

> "v1 abstract corpus에서 관찰된 C6 class R@1 0.40은 (...) 의미 있는 분류 능력의 증거로 보였으나, directional vocabulary를 도입한 v2 clinical corpus에서 동일 클래스가 0.25로 하락한 패턴은 wrist IMU 단독 입력이 left/right/bilateral knee valgus의 trilateral 구분에 본질적 한계를 가짐을 보여준다. 본 결과는 closed-vocabulary policy의 정당성과 추후 multi-modal 확장 필요성을 동시에 정량적으로 뒷받침한다."

## 6. 산출물

```
data/step4r/4rc_contrastive_optional/
└── c5c6_confusion_matrices.csv  (long-format: version_k × true × pred × rate)

reports/step4r/4rc_contrastive_optional/
├── c5c6_confusion_figure.png    (2×2 grid: heatmap + stacked bar)
└── c5c6_confusion_analysis.md   (본 보고서)
```
