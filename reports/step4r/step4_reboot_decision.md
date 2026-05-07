# Step 4 Reboot Decision

- 작성일: 2026-05-08
- 위치: reports/step4r/step4_reboot_decision.md

## 1. 유지하는 단계

Step 1~3은 유지한다.

- Step 1: dataset audit, manifest, participant-disjoint split
- Step 2.5: v2-only evidence audit
- Step 3: uncertainty-aware output schema and policy

## 2. reference로 보존하는 기존 산출물

기존 Step 4~8 산출물은 삭제하지 않는다. 단, main pipeline의 핵심 근거로 사용하지 않는다.

- 기존 Step 4: Logistic Regression baseline/reference
- 기존 Step 5~7: rule/template caption prototype
- 기존 Step 8: human review package, final main evaluation에서 제외

## 3. 새 Step 4 구조

새 Step 4는 Step 4R로 통합한다.

- 4R-A: HistGradientBoosting feature-based ceiling
- 4R-B: BiGRU + Attention raw IMU sensor-to-schema model
- 4R-C: contrastive IMU-text alignment optional extension

## 4. 새 Step 5~8 구조

- Step 5_v2: schema-grounded caption policy
- Step 6_v2: LLM prompt and vocabulary design
- Step 7_v2: guarded LLM caption generation
- Step 8_v2: automatic schema-caption validation

## 5. 최종 결정

본 프로젝트의 main pipeline은 sensor-to-text generation이 아니라 uncertainty-aware sensor-to-schema-to-caption pipeline이다.

LLM은 판단자가 아니라 schema를 한국어로 표현하는 presentation layer로 사용한다.

Human review는 main evaluation에서 제외하고, automatic schema-caption validation으로 대체한다.
