# AICRC_v2

## 프로젝트
손목 IMU 스쿼트 평가 - uncertainty-aware sensor-to-schema-to-caption pipeline (석사 논문)

## 현재 상태
- 완료: Step 1~3, Step 4R-A/B, Step 5~7_v2
- 다음: Step 8_v2 (automatic validation)
- 브랜치: step4-sensor-schema-uncertainty

## 환경
- conda: aicrc_step4
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

## 아카이브
- `_archive/legacy_step5_to_step8/` — v1 step5~8 구버전 (사용 안 함)
- 현재 사용 버전: step5_v2 ~ step8_v2