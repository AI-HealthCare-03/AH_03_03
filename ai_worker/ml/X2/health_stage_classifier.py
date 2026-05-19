# ================================================================
# 건강 단계 룰 기반 판정기
# health_stage_classifier.py
#
# 대상 질환 5개:
#   HTN  — 고혈압        (수축기/이완기 혈압)
#   DM   — 당뇨병        (공복혈당 + HbA1c)
#   DL   — 이상지질혈증  (총콜레스테롤 / LDL / TG / HDL)
#   OBE  — 비만          (BMI)
#   ANEM — 빈혈          (헤모글로빈, 성별 기준 다름)
#
# 사용법:
#   from health_stage_classifier import classify_all
#   result = classify_all(sbp=128, dbp=82, glu=105, hba1c=5.9,
#                         chol=210, ldl=135, tg=160, hdl=55,
#                         bmi=25.5, hb=11.5, sex='F')
#   print(result)
#
# 실행환경: Python 3.8+  |  의존성 없음 (표준 라이브러리만)
# ================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ================================================================
# 결과 데이터 클래스
# ================================================================
@dataclass
class StageResult:
    disease: str           # 질환명 (HTN / DM / DL / OBE / ANEM)
    stage: Optional[int]   # 단계 (0부터 시작, None = 판정 불가)
    label: str             # 단계 레이블 (예: "고혈압 전단계")
    detail: str            # 판정 근거 설명
    missing: list[str] = field(default_factory=list)  # 누락 수치 목록

    def is_normal(self) -> bool:
        return self.stage == 0

    def is_classifiable(self) -> bool:
        return self.stage is not None

    def to_dict(self) -> dict:
        return {
            "disease":  self.disease,
            "stage":    self.stage,
            "label":    self.label,
            "detail":   self.detail,
            "missing":  self.missing,
        }


# ================================================================
# 1. HTN — 고혈압
# ================================================================
# 기준: AHA/ACC 2017 가이드라인 (국내 고혈압 학회 동일 적용)
# 단계:
#   0 정상        sbp < 120  AND  dbp < 80
#   1 주의        120 ≤ sbp ≤ 129  AND  dbp < 80
#   2 고혈압전단계  130 ≤ sbp ≤ 139  OR   80 ≤ dbp ≤ 89
#   3 1단계고혈압  140 ≤ sbp ≤ 159  OR   90 ≤ dbp ≤ 99
#   4 2단계고혈압  sbp ≥ 160        OR   dbp ≥ 100
# 판정 원칙: sbp / dbp 각각 단계 계산 후 높은 쪽 채택
# ================================================================
HTN_LABELS = {
    0: "정상",
    1: "주의혈압",
    2: "고혈압 전단계",
    3: "고혈압 1단계",
    4: "고혈압 2단계",
}

def _htn_stage_single(sbp: float, dbp: float) -> tuple[int, str]:
    """sbp/dbp 각각 단계 → 높은 쪽 반환"""
    def sbp_stage(s: float) -> int:
        if s >= 160: return 4
        if s >= 140: return 3
        if s >= 130: return 2
        if s >= 120: return 1
        return 0

    def dbp_stage(d: float) -> int:
        if d >= 100: return 4
        if d >= 90:  return 3
        if d >= 80:  return 2
        return 0

    ss, ds = sbp_stage(sbp), dbp_stage(dbp)
    stage = max(ss, ds)
    if stage == 4:
        detail = f"수축기 {sbp} mmHg / 이완기 {dbp} mmHg → 2단계 고혈압"
    elif stage == 3:
        detail = f"수축기 {sbp} mmHg / 이완기 {dbp} mmHg → 1단계 고혈압"
    elif stage == 2:
        detail = f"수축기 {sbp} mmHg / 이완기 {dbp} mmHg → 고혈압 전단계"
    elif stage == 1:
        detail = f"수축기 {sbp} mmHg (주의혈압 범위, 이완기 정상)"
    else:
        detail = f"수축기 {sbp} mmHg / 이완기 {dbp} mmHg → 정상"
    return stage, detail

def classify_htn(
    sbp: Optional[float] = None,
    dbp: Optional[float] = None,
) -> StageResult:
    missing = []
    if sbp is None: missing.append("수축기혈압(sbp)")
    if dbp is None: missing.append("이완기혈압(dbp)")

    if missing:
        return StageResult("HTN", None, "판정 불가", "필수 수치 누락", missing)

    stage, detail = _htn_stage_single(sbp, dbp)
    return StageResult("HTN", stage, HTN_LABELS[stage], detail)


# ================================================================
# 2. DM — 당뇨병
# ================================================================
# 기준: ADA 2023 / 대한당뇨병학회
# 단계:
#   0 정상          glu < 100  AND  hba1c < 5.7
#   1 공복혈당장애   100 ≤ glu ≤ 125  OR  5.7 ≤ hba1c ≤ 6.4
#   2 당뇨병        glu ≥ 126        OR  hba1c ≥ 6.5
# 판정 원칙: glu / hba1c 각각 단계 계산 후 높은 쪽 채택
#            둘 중 하나만 있어도 판정 가능 (단, 신뢰도 낮음 명시)
# ================================================================
DM_LABELS = {
    0: "정상",
    1: "공복혈당장애 (당뇨 전단계)",
    2: "당뇨병 의심",
}

def classify_dm(
    glu:   Optional[float] = None,
    hba1c: Optional[float] = None,
) -> StageResult:
    missing = []
    if glu   is None: missing.append("공복혈당(glu)")
    if hba1c is None: missing.append("당화혈색소(HbA1c)")

    if glu is None and hba1c is None:
        return StageResult("DM", None, "판정 불가", "필수 수치 누락", missing)

    stages, details = [], []

    if glu is not None:
        if glu >= 126:
            stages.append(2); details.append(f"공복혈당 {glu} mg/dL (≥126 당뇨 범위)")
        elif glu >= 100:
            stages.append(1); details.append(f"공복혈당 {glu} mg/dL (100~125 장애 범위)")
        else:
            stages.append(0); details.append(f"공복혈당 {glu} mg/dL (정상)")

    if hba1c is not None:
        if hba1c >= 6.5:
            stages.append(2); details.append(f"HbA1c {hba1c}% (≥6.5 당뇨 범위)")
        elif hba1c >= 5.7:
            stages.append(1); details.append(f"HbA1c {hba1c}% (5.7~6.4 전단계)")
        else:
            stages.append(0); details.append(f"HbA1c {hba1c}% (정상)")

    stage = max(stages)
    detail = " / ".join(details)
    if missing:
        detail += f" ※ {', '.join(missing)} 미입력으로 단일 지표 판정"

    return StageResult("DM", stage, DM_LABELS[stage], detail, missing)


# ================================================================
# 3. DL — 이상지질혈증
# ================================================================
# 기준: 한국지질동맥경화학회 2022 이상지질혈증 치료지침
# 단계:
#   0 정상    LDL<100  총콜<200  TG<150  HDL≥60
#   1 경계    LDL 100~129  OR  총콜 200~239  OR  TG 150~199  OR  HDL 40~59
#   2 위험    LDL 130~159  OR  총콜 240~259  OR  TG 200~499  OR  HDL<40
#   3 고위험  LDL≥160      OR  총콜≥260      OR  TG≥500
# 판정 원칙: 4개 지표 각각 단계 계산 후 최고 단계 채택
#            1개 이상 있으면 판정 가능
# ================================================================
DL_LABELS = {
    0: "정상",
    1: "경계",
    2: "위험",
    3: "고위험",
}

def classify_dl(
    chol: Optional[float] = None,
    ldl:  Optional[float] = None,
    tg:   Optional[float] = None,
    hdl:  Optional[float] = None,
) -> StageResult:
    missing = []
    if chol is None: missing.append("총콜레스테롤(chol)")
    if ldl  is None: missing.append("LDL콜레스테롤(ldl)")
    if tg   is None: missing.append("중성지방(tg)")
    if hdl  is None: missing.append("HDL콜레스테롤(hdl)")

    if all(v is None for v in [chol, ldl, tg, hdl]):
        return StageResult("DL", None, "판정 불가", "필수 수치 누락", missing)

    stages, details = [], []

    if ldl is not None:
        if ldl >= 160:
            stages.append(3); details.append(f"LDL {ldl} mg/dL (고위험)")
        elif ldl >= 130:
            stages.append(2); details.append(f"LDL {ldl} mg/dL (위험)")
        elif ldl >= 100:
            stages.append(1); details.append(f"LDL {ldl} mg/dL (경계)")
        else:
            stages.append(0); details.append(f"LDL {ldl} mg/dL (정상)")

    if chol is not None:
        if chol >= 260:
            stages.append(3); details.append(f"총콜레스테롤 {chol} mg/dL (고위험)")
        elif chol >= 240:
            stages.append(2); details.append(f"총콜레스테롤 {chol} mg/dL (위험)")
        elif chol >= 200:
            stages.append(1); details.append(f"총콜레스테롤 {chol} mg/dL (경계)")
        else:
            stages.append(0); details.append(f"총콜레스테롤 {chol} mg/dL (정상)")

    if tg is not None:
        if tg >= 500:
            stages.append(3); details.append(f"중성지방 {tg} mg/dL (고위험)")
        elif tg >= 200:
            stages.append(2); details.append(f"중성지방 {tg} mg/dL (위험)")
        elif tg >= 150:
            stages.append(1); details.append(f"중성지방 {tg} mg/dL (경계)")
        else:
            stages.append(0); details.append(f"중성지방 {tg} mg/dL (정상)")

    if hdl is not None:
        if hdl < 40:
            stages.append(2); details.append(f"HDL {hdl} mg/dL (위험 — 낮음)")
        elif hdl < 60:
            stages.append(1); details.append(f"HDL {hdl} mg/dL (경계)")
        else:
            stages.append(0); details.append(f"HDL {hdl} mg/dL (정상)")

    stage = max(stages)
    detail = " / ".join(details)
    if missing:
        detail += f" ※ {', '.join(missing)} 미입력"

    return StageResult("DL", stage, DL_LABELS[stage], detail, missing)


# ================================================================
# 4. OBE — 비만
# ================================================================
# 기준: 대한비만학회 2022 (아시아-태평양 기준)
# 단계:
#   0 저체중     BMI < 18.5
#   1 정상       18.5 ≤ BMI < 23
#   2 비만전단계  23 ≤ BMI < 25
#   3 1단계비만  25 ≤ BMI < 30
#   4 2단계비만  30 ≤ BMI < 35
#   5 3단계비만  BMI ≥ 35
# ================================================================
OBE_LABELS = {
    0: "저체중",
    1: "정상",
    2: "비만 전단계 (과체중)",
    3: "비만 1단계",
    4: "비만 2단계",
    5: "비만 3단계 (고도비만)",
}

def classify_obe(
    bmi: Optional[float] = None,
    height_cm: Optional[float] = None,
    weight_kg: Optional[float] = None,
) -> StageResult:
    # BMI 없으면 키/몸무게로 계산
    if bmi is None and height_cm is not None and weight_kg is not None:
        bmi = round(weight_kg / (height_cm / 100) ** 2, 1)

    if bmi is None:
        missing = ["BMI"] if (height_cm is None or weight_kg is None) else []
        return StageResult("OBE", None, "판정 불가", "BMI 또는 키·몸무게 필요", missing)

    if bmi >= 35:
        stage = 5
    elif bmi >= 30:
        stage = 4
    elif bmi >= 25:
        stage = 3
    elif bmi >= 23:
        stage = 2
    elif bmi >= 18.5:
        stage = 1
    else:
        stage = 0

    detail = f"BMI {bmi:.1f} kg/m²"
    return StageResult("OBE", stage, OBE_LABELS[stage], detail)


# ================================================================
# 5. ANEM — 빈혈
# ================================================================
# 기준: WHO 기준 (헤모글로빈 기반)
# 성별 기준:
#   남성: Hb < 13.0 g/dL → 빈혈
#   여성: Hb < 12.0 g/dL → 빈혈
# 단계:
#   0 정상
#   1 경증빈혈   남 11~12.9 / 여 11~11.9
#   2 중등도빈혈  Hb 8~10.9
#   3 중증빈혈   Hb < 8
# ================================================================
ANEM_LABELS = {
    0: "정상",
    1: "경증 빈혈",
    2: "중등도 빈혈",
    3: "중증 빈혈",
}

def classify_anem(
    hb:  Optional[float] = None,
    sex: Optional[str]   = None,   # 'M' 또는 'F'
) -> StageResult:
    missing = []
    if hb  is None: missing.append("헤모글로빈(hb)")
    if sex is None: missing.append("성별(sex: M/F)")

    if hb is None:
        return StageResult("ANEM", None, "판정 불가", "헤모글로빈 수치 필요", missing)

    sex_str = str(sex).upper() if sex else ""
    is_male = sex_str in ("M", "1", "남", "MALE")

    normal_cutoff = 13.0 if is_male else 12.0
    sex_label = "남성" if is_male else "여성"

    if hb >= normal_cutoff:
        stage = 0
        detail = f"헤모글로빈 {hb} g/dL → 정상 ({sex_label} 기준 {normal_cutoff} 이상)"
    elif hb >= 11.0:
        stage = 1
        detail = f"헤모글로빈 {hb} g/dL → 경증 빈혈 ({sex_label} 기준)"
    elif hb >= 8.0:
        stage = 2
        detail = f"헤모글로빈 {hb} g/dL → 중등도 빈혈"
    else:
        stage = 3
        detail = f"헤모글로빈 {hb} g/dL → 중증 빈혈 (즉각 의료 확인 권장)"

    if missing:
        detail += f" ※ {', '.join(missing)} 미입력"

    return StageResult("ANEM", stage, ANEM_LABELS[stage], detail, missing)


# ================================================================
# 통합 판정 함수
# ================================================================
def classify_all(
    # HTN
    sbp:       Optional[float] = None,
    dbp:       Optional[float] = None,
    # DM
    glu:       Optional[float] = None,
    hba1c:     Optional[float] = None,
    # DL
    chol:      Optional[float] = None,
    ldl:       Optional[float] = None,
    tg:        Optional[float] = None,
    hdl:       Optional[float] = None,
    # OBE
    bmi:       Optional[float] = None,
    height_cm: Optional[float] = None,
    weight_kg: Optional[float] = None,
    # ANEM
    hb:        Optional[float] = None,
    sex:       Optional[str]   = None,   # 'M' or 'F'
) -> dict[str, StageResult]:
    """
    5개 질환 전체 판정.
    입력값 중 None인 항목은 해당 질환 판정에서 자동 처리.

    Returns:
        dict: {"HTN": StageResult, "DM": ..., "DL": ..., "OBE": ..., "ANEM": ...}
    """
    return {
        "HTN":  classify_htn(sbp=sbp, dbp=dbp),
        "DM":   classify_dm(glu=glu, hba1c=hba1c),
        "DL":   classify_dl(chol=chol, ldl=ldl, tg=tg, hdl=hdl),
        "OBE":  classify_obe(bmi=bmi, height_cm=height_cm, weight_kg=weight_kg),
        "ANEM": classify_anem(hb=hb, sex=sex),
    }


def print_results(results: dict[str, StageResult]):
    """결과 콘솔 출력 (확인용)"""
    print("=" * 60)
    print("건강 단계 판정 결과")
    print("=" * 60)
    DISEASE_KO = {
        "HTN": "고혈압", "DM": "당뇨병",
        "DL": "이상지질혈증", "OBE": "비만", "ANEM": "빈혈"
    }
    for key, r in results.items():
        status = f"[{r.stage}단계]" if r.stage is not None else "[판정불가]"
        print(f"\n  {DISEASE_KO.get(key, key):10s} {status} {r.label}")
        print(f"    근거: {r.detail}")
        if r.missing:
            print(f"    누락: {', '.join(r.missing)}")
    print("=" * 60)


# ================================================================
# 실행 예시
# ================================================================
if __name__ == "__main__":
    # 예시 1: 전체 수치 입력
    print("\n[예시 1] 전체 수치 입력")
    results = classify_all(
        sbp=135, dbp=85,
        glu=108, hba1c=5.9,
        chol=225, ldl=140, tg=180, hdl=48,
        bmi=26.5,
        hb=11.8, sex="F",
    )
    print_results(results)

    # 예시 2: 일부 수치만 입력
    print("\n[예시 2] 혈압 + BMI만 있는 경우")
    results2 = classify_all(
        sbp=155, dbp=95,
        height_cm=170, weight_kg=85,
        sex="M",
    )
    print_results(results2)

    # 예시 3: dict 형태로 결과 활용
    print("\n[예시 3] dict 활용")
    r = classify_all(sbp=118, dbp=76, glu=95, hba1c=5.4,
                     chol=185, ldl=95, tg=130, hdl=62,
                     bmi=22.1, hb=14.2, sex="M")
    for disease, result in r.items():
        print(f"  {disease}: stage={result.stage} / {result.label}")
