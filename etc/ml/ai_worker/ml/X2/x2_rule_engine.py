"""
X2 Rule Engine
==============
검진 수치 기반 리스크 점수 산출기

타겟: 당뇨위험 / 고혈압 / 이상지질혈증 / 간기능이상 / 신장단백뇨 / 비만
방식: 룰엔진 (ML 아님)
출처:
  - 당뇨위험    : 대한당뇨병학회 2023
  - 고혈압      : 대한고혈압학회 2022
  - 이상지질혈증 : 한국지질동맥경화학회 2022
  - 간기능이상   : 대한간학회 2023
  - 신장단백뇨   : 대한신장학회
  - 비만        : 대한비만학회 2022 (한국인 기준)

점수 체계 (타겟별로 단계 수가 다름, 공통 개념):
  0 = 정상
  1 = 주의
  2 = 전단계 / 경도이상
  3 = 위험
  4 = 고위험 (고혈압 2기만 해당)

결측값 처리:
  - 판정에 필요한 값이 없으면 해당 타겟/항목은 None 반환 (스킵)
"""

from __future__ import annotations
from typing import Any

# ---------------------------------------------------------------------------
# CLINICAL_BOUNDS: 입력값 유효범위 검증용
# 범위 초과 시 None 처리 (장비 오류 / 입력 오류로 간주)
# ---------------------------------------------------------------------------
CLINICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "sbp": (60, 300),  # 수축기혈압 mmHg
    "dbp": (30, 200),  # 이완기혈압 mmHg
    "fbs": (40, 600),  # 공복혈당 mg/dL
    "hba1c": (3.0, 20.0),  # 당화혈색소 %
    "tc": (50, 700),  # 총콜레스테롤 mg/dL
    "ldl": (10, 500),  # LDL mg/dL
    "hdl": (10, 200),  # HDL mg/dL
    "tg": (20, 3000),  # 중성지방 mg/dL
    "ast": (5, 2000),  # AST U/L
    "alt": (5, 2000),  # ALT U/L
    "ggt": (5, 2000),  # GGT U/L
    "waist": (40, 200),  # 허리둘레 cm
    "bmi": (10, 70),  # BMI kg/m²
    "urine_protein": (1, 5),  # 요단백 코드 (정수)
}

GGT_UPPER: dict[str, float] = {"M": 63.0, "F": 35.0}  # GGT 정상 상한 U/L


# ---------------------------------------------------------------------------
# 내부 유틸리티
# ---------------------------------------------------------------------------


def _validate(key: str, value: Any) -> float | None:
    """값이 CLINICAL_BOUNDS 범위 내에 있으면 float 반환, 아니면 None."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    lo, hi = CLINICAL_BOUNDS[key]
    return v if lo <= v <= hi else None


def _result(stage: str, score: int, values: dict, criteria: str) -> dict:
    return {
        "단계": stage,
        "점수": score,
        "수치": values,
        "기준": criteria,
    }


def _skip(reason: str = "필수 수치 없음") -> dict:
    return {
        "단계": None,
        "점수": None,
        "수치": {},
        "기준": reason,
    }


# ---------------------------------------------------------------------------
# 타겟별 룰엔진
# ---------------------------------------------------------------------------


def rule_diabetes(fbs: Any = None, hba1c: Any = None) -> dict:
    """
    당뇨위험 판정
      fbs   : 공복혈당 mg/dL (필수)
      hba1c : 당화혈색소 % (선택 — 있으면 정밀 판정 추가)

    점수:
      0 = 정상           (fbs < 100)
      1 = 주의           (해당 없음, 구조 통일용 예비)
      2 = 공복혈당장애   (100 ≤ fbs < 126)
      3 = 당뇨위험       (fbs ≥ 126)
      +hba1c ≥ 6.5% → 점수 max(현재, 3) + 'HbA1c 당뇨 확진' 플래그
    """
    fbs_v = _validate("fbs", fbs)
    hba1c_v = _validate("hba1c", hba1c)

    if fbs_v is None:
        return _skip("공복혈당 없음")

    values = {"공복혈당": fbs_v}
    if hba1c_v is not None:
        values["HbA1c"] = hba1c_v

    # 기본 판정
    if fbs_v < 100:
        stage, score, criteria = "정상", 0, "공복혈당 < 100 mg/dL"
    elif fbs_v < 126:
        stage, score, criteria = "공복혈당장애", 2, "100 ≤ 공복혈당 < 126 mg/dL"
    else:
        stage, score, criteria = "당뇨위험", 3, "공복혈당 ≥ 126 mg/dL"

    # HbA1c 정밀 판정
    hba1c_flag = None
    if hba1c_v is not None:
        if hba1c_v >= 6.5:
            hba1c_flag = "HbA1c 당뇨 확진 (≥6.5%)"
            score = max(score, 3)
            if stage == "정상":
                stage = "당뇨위험(HbA1c)"
        else:
            hba1c_flag = f"HbA1c 정상 범위 ({hba1c_v}%)"

    res = _result(stage, score, values, criteria)
    if hba1c_flag:
        res["HbA1c_판정"] = hba1c_flag
    return res


def rule_hypertension(sbp: Any = None, dbp: Any = None) -> dict:
    """
    고혈압 판정 (대한고혈압학회 2022)
      sbp : 수축기혈압 mmHg (필수)
      dbp : 이완기혈압 mmHg (필수)

    점수:
      0 = 정상          (SBP<120 AND DBP<80)
      1 = 주의혈압      (120≤SBP<130 AND DBP<80)
      2 = 고혈압 전단계 (130≤SBP<140 OR 80≤DBP<90)
      3 = 고혈압 1기    (140≤SBP<160 OR 90≤DBP<100)
      4 = 고혈압 2기    (SBP≥160 OR DBP≥100)
    """
    sbp_v = _validate("sbp", sbp)
    dbp_v = _validate("dbp", dbp)

    if sbp_v is None or dbp_v is None:
        return _skip("수축기/이완기혈압 중 하나 이상 없음")

    values = {"SBP": sbp_v, "DBP": dbp_v}

    if sbp_v >= 160 or dbp_v >= 100:
        return _result("고혈압 2기", 4, values, "SBP ≥ 160 OR DBP ≥ 100 mmHg")
    if sbp_v >= 140 or dbp_v >= 90:
        return _result("고혈압 1기", 3, values, "140 ≤ SBP < 160 OR 90 ≤ DBP < 100 mmHg")
    if sbp_v >= 130 or dbp_v >= 80:
        return _result("고혈압 전단계", 2, values, "130 ≤ SBP < 140 OR 80 ≤ DBP < 90 mmHg")
    if sbp_v >= 120:
        return _result("주의혈압", 1, values, "120 ≤ SBP < 130 AND DBP < 80 mmHg")
    return _result("정상", 0, values, "SBP < 120 AND DBP < 80 mmHg")


def rule_dyslipidemia(
    tc: Any = None,
    ldl: Any = None,
    hdl: Any = None,
    tg: Any = None,
    sex: str = "M",  # "M" or "F"
) -> dict:
    """
    이상지질혈증 — 항목별 개별 판정 후 각각 반환 (종합 단계 없음)
    출처: 한국지질동맥경화학회 2022

    반환 구조:
      {
        "TC":  {"단계": ..., "점수": ..., "수치": ..., "기준": ...},
        "LDL": {...},
        "HDL": {...},
        "TG":  {...},
      }

    점수(각 항목):
      0 = 정상
      1 = 경계
      2 = 이상
    """
    sex = sex.upper() if sex else "M"
    hdl_normal_cutoff = 40 if sex == "M" else 50
    hdl_border_cutoff = 60  # HDL은 높을수록 좋음 — 경계 상단 없음, 하단만 판정

    results: dict[str, dict] = {}

    # TC
    tc_v = _validate("tc", tc)
    if tc_v is not None:
        if tc_v < 200:
            results["TC"] = _result("정상", 0, {"TC": tc_v}, "TC < 200 mg/dL")
        elif tc_v < 240:
            results["TC"] = _result("경계", 1, {"TC": tc_v}, "200 ≤ TC < 240 mg/dL")
        else:
            results["TC"] = _result("이상", 2, {"TC": tc_v}, "TC ≥ 240 mg/dL")
    else:
        results["TC"] = _skip("총콜레스테롤 없음")

    # LDL
    ldl_v = _validate("ldl", ldl)
    if ldl_v is not None:
        if ldl_v < 130:
            results["LDL"] = _result("정상", 0, {"LDL": ldl_v}, "LDL < 130 mg/dL")
        elif ldl_v < 160:
            results["LDL"] = _result("경계", 1, {"LDL": ldl_v}, "130 ≤ LDL < 160 mg/dL")
        else:
            results["LDL"] = _result("이상", 2, {"LDL": ldl_v}, "LDL ≥ 160 mg/dL")
    else:
        results["LDL"] = _skip("LDL콜레스테롤 없음")

    # HDL (낮을수록 나쁨)
    hdl_v = _validate("hdl", hdl)
    if hdl_v is not None:
        if hdl_v < hdl_normal_cutoff:
            results["HDL"] = _result(
                "이상", 2, {"HDL": hdl_v}, f"HDL < {hdl_normal_cutoff} mg/dL ({'남' if sex == 'M' else '여'})"
            )
        elif hdl_v < hdl_border_cutoff:
            results["HDL"] = _result(
                "경계", 1, {"HDL": hdl_v}, f"{hdl_normal_cutoff} ≤ HDL < {hdl_border_cutoff} mg/dL"
            )
        else:
            results["HDL"] = _result("정상", 0, {"HDL": hdl_v}, f"HDL ≥ {hdl_border_cutoff} mg/dL")
    else:
        results["HDL"] = _skip("HDL콜레스테롤 없음")

    # TG
    tg_v = _validate("tg", tg)
    if tg_v is not None:
        if tg_v < 150:
            results["TG"] = _result("정상", 0, {"TG": tg_v}, "TG < 150 mg/dL")
        elif tg_v < 200:
            results["TG"] = _result("경계", 1, {"TG": tg_v}, "150 ≤ TG < 200 mg/dL")
        else:
            results["TG"] = _result("이상", 2, {"TG": tg_v}, "TG ≥ 200 mg/dL")
    else:
        results["TG"] = _skip("중성지방 없음")

    return results


def rule_liver(
    ast: Any = None,
    alt: Any = None,
    ggt: Any = None,
    sex: str = "M",
) -> dict:
    """
    간기능이상 판정 (대한간학회 2023)
      GGT 정상 상한: 남 63 / 여 35 U/L

    점수:
      0 = 정상       (AST/ALT ≤ 40 AND GGT ≤ 상한)
      1 = 경도이상   (AST/ALT 40~80 OR GGT 1.0~1.5× 상한)
      2 = 이상       (AST/ALT > 80 OR GGT > 1.5× 상한)

    AST/ALT 중 하나라도 있으면 판정 가능.
    GGT는 선택 — 없으면 AST/ALT만으로 판정.
    """
    sex = sex.upper() if sex else "M"
    ggt_limit = GGT_UPPER.get(sex, 63.0)

    ast_v = _validate("ast", ast)
    alt_v = _validate("alt", alt)
    ggt_v = _validate("ggt", ggt)

    if ast_v is None and alt_v is None:
        return _skip("AST/ALT 모두 없음")

    values: dict = {}
    if ast_v is not None:
        values["AST"] = ast_v
    if alt_v is not None:
        values["ALT"] = alt_v
    if ggt_v is not None:
        values["GGT"] = ggt_v

    # AST/ALT 최악값
    transaminase_vals = [v for v in [ast_v, alt_v] if v is not None]
    max_ta = max(transaminase_vals)

    # GGT 배수
    ggt_ratio = (ggt_v / ggt_limit) if ggt_v is not None else 0.0

    if max_ta > 80 or ggt_ratio > 1.5:
        return _result("이상", 2, values, f"AST/ALT > 80 OR GGT > 1.5× 상한({ggt_limit} U/L)")
    if max_ta > 40 or ggt_ratio > 1.0:
        return _result("경도이상", 1, values, f"40 < AST/ALT ≤ 80 OR 1.0× < GGT ≤ 1.5× 상한({ggt_limit} U/L)")
    return _result("정상", 0, values, f"AST/ALT ≤ 40 AND GGT ≤ {ggt_limit} U/L")


def rule_proteinuria(urine_protein: Any = None) -> dict:
    """
    신장단백뇨 판정 (대한신장학회)
      코드 체계:
        1 = 음성(-)
        2 = 미량(trace)  ← 일부 공단 데이터에서 trace를 2로 코딩
        3 = 1+
        4 = 2+
        5 = 3+ 이상

    점수:
      0 = 정상        (코드 1)
      1 = 단백뇨 의심 (코드 2~3, trace~1+)
      2 = 단백뇨      (코드 4~5, 2+ 이상)
    """
    up_v = _validate("urine_protein", urine_protein)
    if up_v is None:
        return _skip("요단백 없음")

    code = int(round(up_v))
    values = {"요단백_코드": code}

    if code <= 1:
        return _result("정상", 0, values, "요단백 음성 (코드 1)")
    if code <= 3:
        return _result("단백뇨 의심", 1, values, "요단백 trace~1+ (코드 2~3)")
    return _result("단백뇨", 2, values, "요단백 2+ 이상 (코드 4~5)")


def rule_obesity(bmi: Any = None) -> dict:
    """
    비만 판정 (대한비만학회 2022 한국인 기준)

    점수:
      0 = 저체중 / 정상
      1 = 과체중(비만전단계)  BMI 23.0~24.9
      2 = 1단계 비만          BMI 25.0~29.9
      3 = 2단계 비만          BMI 30.0~34.9
      4 = 3단계 비만(고도비만) BMI ≥ 35.0
    """
    bmi_v = _validate("bmi", bmi)
    if bmi_v is None:
        return _skip("BMI 없음")

    values = {"BMI": bmi_v}

    if bmi_v < 18.5:
        return _result("저체중", 0, values, "BMI < 18.5")
    if bmi_v < 23.0:
        return _result("정상", 0, values, "18.5 ≤ BMI < 23.0")
    if bmi_v < 25.0:
        return _result("과체중(비만전단계)", 1, values, "23.0 ≤ BMI < 25.0")
    if bmi_v < 30.0:
        return _result("1단계 비만", 2, values, "25.0 ≤ BMI < 30.0")
    if bmi_v < 35.0:
        return _result("2단계 비만", 3, values, "30.0 ≤ BMI < 35.0")
    return _result("3단계 비만(고도비만)", 4, values, "BMI ≥ 35.0")


# ---------------------------------------------------------------------------
# 통합 엔트리포인트
# ---------------------------------------------------------------------------


def run_x2(
    *,
    # 혈압
    sbp: Any = None,
    dbp: Any = None,
    # 혈당
    fbs: Any = None,
    hba1c: Any = None,
    # 지질
    tc: Any = None,
    ldl: Any = None,
    hdl: Any = None,
    tg: Any = None,
    # 간
    ast: Any = None,
    alt: Any = None,
    ggt: Any = None,
    # 신장
    urine_protein: Any = None,
    # 체형
    bmi: Any = None,
    # 공통
    sex: str = "M",  # "M" or "F"
) -> dict:
    """
    X2 룰엔진 통합 실행

    Parameters
    ----------
    모든 검진 수치를 키워드 인수로 전달.
    없는 값은 None으로 두면 해당 항목/타겟 자동 스킵.

    Returns
    -------
    dict:
      {
        "당뇨위험":      {...},
        "고혈압":        {...},
        "이상지질혈증":  {"TC": {...}, "LDL": {...}, "HDL": {...}, "TG": {...}},
        "간기능이상":    {...},
        "신장단백뇨":    {...},
        "비만":          {...},
      }
    """
    return {
        "당뇨위험": rule_diabetes(fbs=fbs, hba1c=hba1c),
        "고혈압": rule_hypertension(sbp=sbp, dbp=dbp),
        "이상지질혈증": rule_dyslipidemia(tc=tc, ldl=ldl, hdl=hdl, tg=tg, sex=sex),
        "간기능이상": rule_liver(ast=ast, alt=alt, ggt=ggt, sex=sex),
        "신장단백뇨": rule_proteinuria(urine_protein=urine_protein),
        "비만": rule_obesity(bmi=bmi),
    }


# ---------------------------------------------------------------------------
# 간단 동작 확인용
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    sample = run_x2(
        sbp=145,
        dbp=88,
        fbs=112,
        hba1c=6.1,
        tc=235,
        ldl=148,
        hdl=45,
        tg=185,
        ast=55,
        alt=72,
        ggt=90,
        urine_protein=3,
        bmi=26.4,
        sex="M",
    )
    print(json.dumps(sample, ensure_ascii=False, indent=2))
