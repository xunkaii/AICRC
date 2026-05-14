# AICRC_v2

## 프로젝트
손목 IMU 스쿼트 평가 - uncertainty-aware sensor-to-schema-to-caption pipeline (석사 논문)

## 현재 상태 (2026-05-12 갱신)
- 완료:
  - Step 1~3, Step 5_v2~7_v2 (legacy single-seed schema 기반)
  - **Step 4R-A**: HGB ceiling + temperature scaling 보강 (calibrated/)
  - **Step 4R-B ablation**: b00(baseline 3-seed) → b01(aug 약함) → **b01.5(aug 강함, 채택)** → b02/b02b/b03(미채택). **b01.5가 confirmed best: test F1 0.513±0.008, std 3× 감소.** median seed=42.
  - **Step 4R-C 전체 사이클** (Day 1~6): text corpus + frozen ko-sroberta-multitask text emb + frozen 4R-B IMU emb + contrastive projection 학습 + retrieval 평가 + t-SNE. test template R@1 0.69 (24× chance), ambig R@1 0.91.
  - Step 7_v2 b01.5/seed42 rerun: mock provider, schema_faithfulness_rate 0.9998 (mock-golden mismatch 2건만).
- 다음: Step 8_v2 (automatic validation) — 옵션 3 (real LLM caption rerun) 도 후보.
- 브랜치: step4-sensor-schema-uncertainty

## 환경
- **3개 conda env 용도 분리**:
  - `aicrc_step4`: sklearn 전용 (Step 4R-A HGB / temperature scaling)
  - `aicrc_env`: torch 2.3.1+cu118 (Step 4R-B 학습/calibrate/schema)
  - `dl_env`: torch 2.5.1+cu118 + transformers 4.40.2 + sentence-transformers (Step 4R-C 전용)
  - 자세한 경로/설치 패키지: `~/.claude/projects/.../memory/reference_python_env.md`
- CSV: encoding="utf-8-sig" 필수 (Windows Excel)
- API key: .env (python-dotenv)
- 데이터: C:\Users\user\data\AICRC_DSQ_Data(Combined_segmented)_v2\

## 규칙
- 기존 파일 수정/삭제 금지
- split은 manifest_split.csv 그대로 사용
- 새 산출물은 지정 경로에만 저장
- Step 1~3 산출물 절대 수정 금지

## 핵심 문서
- reports/step4r/step4_research_reframing.md (연구 방향)
- reports/step6_v2/step6_v2_prompt_finalization.md (caption pipeline)
- reports/step4r/4rb_attention/experiments/b01_5_aug_jitter_scale_strong/aggregate_results.md (b01.5 final)
- reports/step4r/4rc_contrastive_optional/results_final.md (4R-C 최종)
- reports/step4r/4ra_feature_ceiling/calibrated/step4r_hgb_schema_results.md (4R-A calibrated)

## 아카이브
- `_archive/legacy_step5_to_step8/` — v1 step5~8 구버전 (사용 안 함)
- 현재 사용 버전: step5_v2 ~ step8_v2