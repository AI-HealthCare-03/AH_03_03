# 만성질환 유병 예측 모델 개발 — 전체 종합 보고서

> **프로젝트:** 만성질환 생활습관 챌린지 웹 서비스 — 예측 모델 개발
> **작성일:** 2026-05-20
> **대상 질환:** 고혈압 · 당뇨 · 이상지질혈증
> **모델:** XGBoost · CatBoost
> **데이터:** 국민건강영양조사 2015~2024년
> **평가 기준:** Recall ≥ 0.80 우선 · F1 ≥ 0.60 목표
> **검증:** Stratified 5-Fold CV

---

## 1. 프로젝트 개요

### 1-1. 목적

국민건강영양조사 데이터를 활용해 고혈압·당뇨·이상지질혈증 유병 여부를 예측하는 ML 모델을 개발하고, 만성질환 생활습관 챌린지 서비스에서 사용자의 질환 위험도를 사전 스크리닝하는 데 활용한다.

### 1-2. 평가 기준 선택 근거

- **Recall 우선:** 실제 환자를 놓치는 것(FN)이 정상인을 과탐지(FP)하는 것보다 스크리닝 목적상 치명적
- **F1 보조:** FP 과다 시 서비스 신뢰도 하락. Recall·Precision 균형 관리
- **AUC-ROC:** 모델 전반적 판별력 확인

### 1-3. 실험 파이프라인

```
데이터 준비 → Baseline → Threshold 탐색 → Optuna 튜닝 → SHAP 분석
→ Feature Engineering → 데이터 확장 → Optuna 재튜닝 → Threshold 재탐색
```

---

## 2. 데이터 구성 및 전처리

### 2-1. 데이터 버전 히스토리

| 버전 | 연도 범위 | 수집 전략 | 전체 샘플 |
| --- | --- | --- | --- |
| v0.1 | 2024년 | 전체 성인 | 6,033명 |
| v2.1 | 2018~2024년 | 22~24년 전체 + 18~21년 당뇨 유병자 선별 | 19,765명 |
| v4 | 2015~2024년 | v2.1 + 15~17년 당뇨 유병자 추가 | 22,102명 |

### 2-2. 질환별 클래스 분포 변화

| 질환 | v0.1 비율 | v2.1 비율 | v4 비율 |
| --- | --- | --- | --- |
| 고혈압 | 2.59:1 | 2.10:1 | 1.84:1 |
| **당뇨** | **6.44:1** | **3.23:1** | **2.49:1** |
| 이상지질혈증 | 2.87:1 | 2.52:1 | 2.15:1 |

### 2-3. 주요 전처리 항목

| 항목 | 처리 방식 |
| --- | --- |
| 타겟 이진화 | 1=유병 / 8(해당없음)=정상 / 0·9=제외 |
| 가족력 결측 | 8·9 → 0 (정보 없음 = 가족력 없음) |
| 직업 | OHE (8개 범주) |
| 음주빈도 | 6→과거음주_현재금주 파생 후 0, OrdinalEncoding(0~5) |
| 음주량 | 8→0(비음주), 9→NaN, OrdinalEncoding(unknown=NaN) |
| BMI 결측 | 키·체중으로 재계산 (불가 시 NaN 유지) |

### 2-4. 전처리 버그 수정 이력

| 버그 | 원인 | 수정 |
| --- | --- | --- |
| 음주량_enc 음수값 (-1) | OrdinalEncoder `unknown_value=-1` | `unknown_value=np.nan` 변경 |
| 가족력 8·9 이상값 | v2 데이터 합산 시 미처리 | 전처리 파일 섹션에 `replace({8:0, 9:0})` 추가 |
| DROP_COLS 중복 | 파일 생성 시 치환 오류 | 항상 3개 질환 모두 고정 작성 |

---

## 3. Feature Engineering 모듈

### 3-1. 구현된 FE 파일 (features/ 폴더)

| 파일 | 함수 | 주요 파라미터 | 추가 컬럼 |
| --- | --- | --- | --- |
| `fe_age_bin.py` | `add_age_bin` | `drop_original` | 나이_구간 (0~3) |
| `fe_age_bin5.py` | `add_age_bin5` | `drop_original` | 나이_구간5 (0~4) |
| `fe_family_sum.py` | `add_family_sum` | `hypertension/diabetes/dyslipidemia`, `drop_original` | 가족력_합계 (0~3) |
| `fe_bmi_bin.py` | `add_bmi_bin` | `korean_standard`, `drop_original` | BMI_구간 |
| `fe_alcohol.py` | `add_alcohol_load` | `drop_original` | 음주_총부하 |
| `fe_exercise.py` | `add_exercise_total` | `drop_original` | 총운동일수 |
| `fe_body.py` | `add_body_features` | `weight_height_ratio`, `bmi_age_interaction`, `drop_original` | 체중_키_비율, BMI_나이_상호작용 |
| `fe_age_family.py` | `add_age_family_interaction` | `hypertension`, `diabetes` | 나이_고혈압/당뇨가족력 |

### 3-2. FE 적용 순서 (의존성)

```
add_age_bin / add_age_bin5  →  add_family_sum  →  나머지
(fe_body, fe_age_family는 나이_구간 + 가족력_합계 선행 필요)
```

---

## 4. 고혈압 실험 결과

### 4-1. 전체 성능 추이

| 단계 | 모델 | 데이터 | AUC | Recall | F1 | THR |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | XGBoost | v0.1 | 0.8531 | 0.8155 | 0.6458 | 0.50 |
| Optuna | XGBoost | v0.1 | 0.8591 | 0.8262 | 0.6549 | 0.50 |
| Baseline | CatBoost | v0.1 | 0.8582 | 0.8310 | 0.6475 | 0.50 |
| Optuna | CatBoost | v0.1 | 0.8553 | 0.8357 | 0.6520 | 0.50 |
| v2 baseline | CatBoost | v2.1 | 0.8562 | 0.8321 | 0.6874 | 0.50 |
| **v2 Optuna** | **CatBoost** | **v2.1** | **0.8566** | **0.8399** | **0.6866** | **0.50** |

### 4-2. 최종 파라미터 (CatBoost v2.1)

```python
iterations=701, learning_rate=0.0281, depth=3,
l2_leaf_reg=7.2375, bagging_temperature=0.6630,
random_strength=0.5959, border_count=159,
class_weights={0:1.0, 1:2.1050}
```

### 4-3. SHAP Top 5

| 순위 | Feature | SHAP |
| --- | --- | --- |
| 1 | 나이 | 1.5802 |
| 2 | BMI | 0.5006 |
| 3 | 고혈압가족력_형제 | 0.2511 |
| 4 | 고혈압가족력_모 | 0.2207 |
| 5 | 고혈압가족력_부 | 0.1881 |

### 4-4. 현재 확정 사항

| 모델 | THR | Recall | F1 | 데이터 |
| --- | --- | --- | --- | --- |
| **CatBoost** | **0.50** | **0.8399** | **0.6866** | v2.1 |

> FE 실험 미진행. v4 데이터 적용 및 Optuna 재튜닝 예정

---

## 5. 당뇨 실험 결과

### 5-1. 전체 성능 추이

| 단계 | 모델 | 데이터 | AUC | Recall | F1 | THR |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | XGBoost | v0.1 | 0.7831 | 0.7398 | 0.3972 | 0.50 |
| Threshold 조정 | XGBoost | v0.1 | 0.7831 | 0.8163 | 0.3868 | 0.45 |
| Optuna | XGBoost | v0.1 | 0.8004 | 0.8212 | 0.3994 | 0.45 |
| Baseline | CatBoost | v0.1 | 0.8065 | 0.8027 | 0.4005 | 0.50 |
| Optuna | CatBoost | v0.1 | 0.8091 | 0.8249 | 0.4036 | 0.50 |
| v2 baseline | CatBoost | v2.1 | 0.8177 | 0.8074 | 0.5596 | 0.50 |
| v2 Optuna | CatBoost | v2.1 | 0.8095 | 0.8196 | 0.5548 | 0.50 |
| v4 baseline | CatBoost | v4 | 0.8059 | 0.8148 | 0.6022 | 0.50 |
| v4 Optuna | CatBoost | v4 | 0.8067 | 0.8139 | 0.6037 | 0.50 |
| **v4 Threshold** | **CatBoost** | **v4** | **0.8067** | **0.8139** | **0.6037** | **0.50** |

### 5-2. 최종 파라미터 (CatBoost v4)

```python
iterations=585, learning_rate=0.0342, depth=5,
l2_leaf_reg=4.2105, bagging_temperature=0.3555,
random_strength=0.9046, border_count=115,
class_weights={0:1.0, 1:2.4900}
```

### 5-3. FE 실험 결과 요약

| 실험 | F1 | FN | 결론 |
| --- | --- | --- | --- |
| v2.1 baseline | 0.5548 | 840 | — |
| age_bin5 + BMI 한국 | **0.5616** | **874** | v2.1 최적 |
| v4 baseline | 0.6022 | 1,170 | — |
| v4 age_bin5 + BMI | 0.6012 | 1,199 | ❌ 효과 없음 |

> v4에서 FE 효과 없음 — 충분한 데이터로 CatBoost가 이미 비선형 패턴 학습

### 5-4. SHAP Top 5

| 순위 | Feature | SHAP |
| --- | --- | --- |
| 1 | 나이 | 1.0584 |
| 2 | 당뇨가족력_형제 | 0.3091 |
| 3 | BMI | 0.2970 |
| 4 | 성별 | 0.2284 |
| 5 | 당뇨가족력_모 | 0.2002 |

### 5-5. 현재 확정 사항

| 모델 | THR | Recall | F1 | 데이터 |
| --- | --- | --- | --- | --- |
| **CatBoost** | **0.50** | **0.8139** | **0.6037** ✅ | v4 |

---

## 6. 이상지질혈증 실험 결과

### 6-1. 전체 성능 추이

| 단계 | 모델 | 데이터 | AUC | Recall | F1 | THR |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | XGBoost | v0.1 | 0.7879 | 0.7953 | 0.5554 | 0.50 |
| Threshold 조정 | XGBoost | v0.1 | 0.7879 | 0.8427 | 0.5542 | 0.45 |
| Optuna | XGBoost | v0.1 | 0.7898 | 0.8556 | 0.5726 | 0.45 |
| Baseline | CatBoost | v0.1 | 0.8035 | 0.8241 | 0.5718 | 0.50 |
| Optuna | CatBoost | v0.1 | 0.8046 | 0.8408 | 0.5738 | 0.50 |
| v2 baseline | XGBoost | v2.1 | 0.7855 | 0.8414 | 0.5845 | 0.45 |
| v2 Optuna | XGBoost | v2.1 | 0.7890 | 0.8508 | 0.5868 | 0.45 |
| v4 baseline | XGBoost | v4 | 0.7757 | 0.8529 | 0.6076 | 0.45 |
| v4 Optuna | XGBoost | v4 | 0.7782 | 0.8776 | 0.6089 | 0.45 |
| **v4 Threshold** | **XGBoost** | **v4** | **0.7782** | **0.8776** | **0.6089** | **0.45** |

### 6-2. 최종 파라미터 (XGBoost v4)

```python
n_estimators=556, learning_rate=0.0336, max_depth=3,
min_child_weight=7, subsample=0.7171, colsample_bytree=0.7645,
gamma=0.4590, reg_alpha=0.9609, reg_lambda=2.4030,
scale_pos_weight=2.1527
```

### 6-3. FE 실험 결과 요약 (v2.1 기준)

| 실험 | Recall | F1 | FN | 결론 |
| --- | --- | --- | --- | --- |
| v2.1 baseline | 0.8416 | **0.5846** | 862 | — |
| 체중_키_비율만 | 0.8596 | 0.5837 | **764** | FN 최소 |
| 이상지질3종+합산 | **0.8624** | 0.5793 | 749 | Recall 최고 |
| 나이+4구간 | 0.8554 | 0.5841 | 787 | — |

> XGBoost에서 체형·음주·운동 복합 피처 효과 확인 (CatBoost 당뇨와 반대 패턴)
> v4 기준 FE 실험 미진행

### 6-4. SHAP Top 5

| 순위 | Feature | SHAP |
| --- | --- | --- |
| 1 | 나이 | 1.0422 |
| 2 | BMI | 0.2619 |
| 3 | 키 | 0.1309 |
| 4 | 체중 | 0.0902 |
| 5 | 고지혈증가족력_형제 | 0.0624 |

### 6-5. 현재 확정 사항

| 모델 | THR | Recall | F1 | 데이터 |
| --- | --- | --- | --- | --- |
| **XGBoost** | **0.45** | **0.8776** | **0.6089** ✅ | v4 |

---

## 7. 세 질환 최종 성능 비교

| 질환 | 모델 | AUC | Recall | Precision | F1 | FP | FN | THR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 고혈압 | CatBoost | 0.8566 | 0.8399 | 0.5807 | 0.6866 | 3,807 | 1,005 | 0.50 |
| 당뇨 | CatBoost | 0.8067 | 0.8139 | 0.4798 | **0.6037** ✅ | 5,576 | 1,176 | 0.50 |
| 이상지질혈증 | XGBoost | 0.7782 | 0.8776 | 0.4662 | **0.6089** ✅ | 6,823 | 831 | 0.45 |

---

## 8. 모델 선택 근거

| 질환 | 추천 모델 | 근거 |
| --- | --- | --- |
| 고혈압 | **CatBoost** | FN·Recall 우위. Threshold 조정 불필요 |
| 당뇨 | **CatBoost** | 전 지표 우위. 6.44:1 불균형에서도 THR 0.50 유지 |
| 이상지질혈증 | **XGBoost** | Recall·FN 우위. 체형 복합 피처 효과 반영 가능 |

---

## 9. 데이터 확장 효과 분석

| 질환 | v0.1 F1 | v2.1 F1 | v4 F1 | 총 개선 |
| --- | --- | --- | --- | --- |
| 당뇨 | 0.4036 | 0.5548 | **0.6037** | ▲ 0.200 |
| 이상지질혈증 | 0.5726 | 0.5868 | **0.6089** | ▲ 0.036 |

> **당뇨에서 데이터 확장 효과 극적** — 불균형 6.44→2.49 완화로 F1이 0.40→0.60 달성
> **FE보다 데이터 확장이 더 효과적** — v2.1에서 FE 최대 F1 0.5616이었는데, v4 baseline이 0.6022로 즉시 초과

---

## 10. 주요 인사이트

### 공통 인사이트

1. **나이·BMI 세 질환 모두 SHAP 1·2위** — 인구통계·비만이 만성질환 예측의 공통 핵심
2. **CatBoost가 불균형 처리 우수** — 당뇨(6.44:1)에서 XGBoost는 THR 0.45 필요, CatBoost는 0.50 유지
3. **데이터 확장이 FE보다 효과적** — 충분한 데이터에서 모델이 비선형 패턴 자동 학습

### 질환별 차별화 인사이트

| 질환 | 핵심 발견 |
| --- | --- |
| 고혈압 | 가족력 순서: 형제>모>부 (v2.1부터 변화). 음주량이 생활습관 변수 중 최상위 |
| 당뇨 | 가족력 순서: 형제>모>부 (생활환경 공유 효과). 근력운동 Top 10 (인슐린 저항성) |
| 이상지질혈증 | 키·체중이 BMI와 독립적으로 상위권. XGBoost v3→v4 best_iter 조기 수렴 문제 해소 |

### CatBoost vs XGBoost 패턴 차이

| 항목 | CatBoost | XGBoost |
| --- | --- | --- |
| 불균형 처리 | class_weights 내부 결합으로 우수 | scale_pos_weight 단순 조정 |
| FE 반응 | 충분한 데이터 시 효과 없음 | 복합 피처 명시 시 효과 있음 |
| Threshold | 대부분 0.50 유지 | 불균형 심할 시 하향 필요 |
| 체형 피처 | 내부 학습으로 중복 | 명시적 추가 시 FN 감소 |

---

## 11. 현재 파일 구조

```
Desktop/final_project/ML/
├── data/
│   ├── v0.1_x1_preprocessed.csv
│   ├── hn_all_preprocessed_v2.1.csv
│   └── hn_all_preprocessed_v4.csv      ← 현재 사용
├── features/
│   ├── fe_age_bin.py
│   ├── fe_age_bin5.py
│   ├── fe_family_sum.py
│   ├── fe_bmi_bin.py
│   ├── fe_alcohol.py
│   ├── fe_exercise.py
│   ├── fe_body.py
│   └── fe_age_family.py
├── models/
│   ├── cat_v4_1_DM_alldata_fe.ipynb    ← 당뇨 베이스 파일
│   └── xgb_v4_1_HL_alldata_fe.ipynb   ← 이상지질혈증 베이스 파일
├── outputs/oof/
│   ├── oof_proba_DM_catboost_v4_threshold.npy
│   └── oof_proba_HL_xgboost_v4_threshold.npy
├── model_result.db
└── model_logger.py
```

---

## 12. 다음 단계

- [ ] **고혈압 v4 데이터 적용** — Optuna 재튜닝 및 Threshold 재탐색
- [ ] **이상지질혈증 v4 FE 실험** — v2.1에서 효과 있던 피처 v4 기준 재확인
- [ ] **앙상블** — XGBoost + CatBoost OOF proba 가중 평균
- [ ] **서비스 연동** — 최종 모델 pkl 저장 및 API 연동 준비
