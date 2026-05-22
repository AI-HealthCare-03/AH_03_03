# Challenge Recommendation Design

## Overview

The challenge master data is managed from `docs/data/challenges/team_challenge_master.csv`.
`scripts/seed_mvp_challenges.py` reads this CSV and upserts `challenges` rows by
`title + category`. Existing user participation and logs are not deleted.

## Challenge Types

| Type | Meaning | Use |
| --- | --- | --- |
| `SPECIAL` | 질환군에게 우선 추천하는 특수 챌린지 | 당뇨, 고혈압, 이상지질혈증, 비만 등 위험도 결과 기반 추천 |
| `COMMON` | 모든 질환군에게 추천 가능한 공통 챌린지 | 수분, 걷기, 수면, 건강기록 같은 기본 건강관리 |
| `GENERAL` | 질환 여부와 무관한 일반 챌린지 | 일반 생활습관 기록과 실천 |

## Target Disease

Supported target disease values:

- `HYPERTENSION`
- `DIABETES`
- `DYSLIPIDEMIA`
- `OBESITY`
- `COMMON`
- `GENERAL`

`target_disease` is recommendation metadata. It must not be shown as a diagnosis.
User-facing labels should use phrases such as "고혈압 관리" and "건강관리 참고용".

## Caution And Contraindication Policy

Some general or exercise challenges may not fit every user. The challenge master
therefore includes:

- `caution_message`: 진행 전 또는 진행 중 주의할 안내
- `contraindication_message`: 특정 증상/상황에서 중단 또는 의료진 상담을 권장하는 안내

Allowed wording:

- 건강관리 참고용
- 필요한 경우 의료진 상담 권장
- 무리가 느껴지면 강도를 낮추거나 중단

Avoid:

- 확진
- 치료
- 처방
- 완치
- 진단 단정

## Recommendation Direction

Current implementation focuses on the data structure and display. Advanced
recommendation logic is deferred.

Initial filtering can use:

- `challenge_type`
- `target_disease`
- `category`
- `difficulty`

Future algorithm:

1. Map latest analysis result disease and risk level.
2. Prioritize `SPECIAL` challenges for that disease.
3. Add `COMMON` challenges as safe baseline habits.
4. Include `GENERAL` challenges only when caution text is suitable.
5. Use rules first; LLM-based recommendation can be added later with prompt
   templates and safety constraints.

## Seed Policy

The local seed script:

- Creates new CSV rows if missing.
- Updates changed challenge metadata.
- Marks active challenges missing from the CSV as `INACTIVE`.
- Does not delete `user_challenges` or `challenge_logs`.

This keeps existing participation history intact while replacing temporary
master data with the team challenge master.
