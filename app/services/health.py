from decimal import ROUND_HALF_UP, Decimal

from app.dtos.health import HealthRecordCreateRequest, HealthRecordUpdateRequest
from app.models.health import HealthRecord
from app.models.users import User
from app.repositories import health_repository

HEALTH_RECORD_SOURCE_MANUAL = "MANUAL"
HEALTH_RECORD_SOURCE_OCR = "OCR"
HEALTH_RECORD_SOURCE_PROFILE = "PROFILE"
HEALTH_RECORD_SOURCE_ANALYSIS_PREP = "ANALYSIS_PREP"
HEALTH_RECORD_SOURCES = {
    HEALTH_RECORD_SOURCE_MANUAL,
    HEALTH_RECORD_SOURCE_OCR,
    HEALTH_RECORD_SOURCE_PROFILE,
    HEALTH_RECORD_SOURCE_ANALYSIS_PREP,
}
HEALTH_RECORD_SNAPSHOT_FIELDS = (
    "height_cm",
    "weight_kg",
    "bmi",
    "waist_cm",
    "systolic_bp",
    "diastolic_bp",
    "fasting_glucose",
    "hba1c",
    "total_cholesterol",
    "ldl_cholesterol",
    "hdl_cholesterol",
    "triglyceride",
    "has_diabetes",
    "has_obesity",
    "has_dyslipidemia",
    "has_hypertension",
    "occupation_code",
    "family_htn",
    "family_dm",
    "family_dyslipidemia",
    "smoking_status",
    "drinking_frequency",
    "drinking_amount",
    "walking_days_per_week",
    "strength_days_per_week",
    "sleep_hours",
)

REQUIRED_USER_ANALYSIS_FIELDS = {
    "gender": "성별",
    "birthday": "나이",
}

REQUIRED_BASIC_ANALYSIS_FIELDS = {
    "height_cm": "키",
    "weight_kg": "몸무게",
    "bmi": "BMI",
    "occupation_code": "직업군",
    "family_htn": "고혈압 가족력 여부",
    "family_dm": "당뇨병 가족력 여부",
    "family_dyslipidemia": "이상지질혈증 가족력 여부",
    "smoking_status": "현재 흡연 여부",
    "drinking_frequency": "1년간 음주 빈도",
    "drinking_amount": "한 번 음주량",
    "walking_days_per_week": "1주일간 걷기 일수",
    "strength_days_per_week": "1주일간 근력운동 일수",
}

REQUIRED_PRECISION_ANALYSIS_FIELDS = {
    "systolic_bp": "수축기 혈압",
    "diastolic_bp": "이완기 혈압",
    "fasting_glucose": "공복혈당",
    "total_cholesterol": "총콜레스테롤",
    "ldl_cholesterol": "LDL 콜레스테롤",
    "hdl_cholesterol": "HDL 콜레스테롤",
    "triglyceride": "중성지방",
    "waist_cm": "허리둘레",
}

OPTIONAL_PRECISION_ANALYSIS_FIELDS = {
    "hba1c": "당화혈색소",
}


async def create_health_record(user_id: int, request: HealthRecordCreateRequest) -> HealthRecord:
    data = _with_calculated_bmi(request.model_dump())
    data["source"] = normalize_health_record_source(data.get("source"), HEALTH_RECORD_SOURCE_MANUAL)
    return await health_repository.create_health_record(user_id, data)


async def get_health_record(record_id: int) -> HealthRecord | None:
    return await health_repository.get_health_record_by_id(record_id)


async def get_latest_health_record(user_id: int) -> HealthRecord | None:
    return await health_repository.get_latest_health_record_by_user(user_id)


async def list_health_records(user_id: int, limit: int = 20, offset: int = 0) -> list[HealthRecord]:
    return await health_repository.list_health_records_by_user(user_id, limit=limit, offset=offset)


async def update_health_record(record_id: int, request: HealthRecordUpdateRequest) -> HealthRecord | None:
    data = request.model_dump(exclude_unset=True)
    data = _with_calculated_bmi(data)
    if "source" in data:
        data["source"] = normalize_health_record_source(data.get("source"), HEALTH_RECORD_SOURCE_MANUAL)
    return await health_repository.update_health_record(record_id, data)


async def delete_health_record(record_id: int) -> int:
    return await health_repository.delete_health_record(record_id)


async def get_analysis_readiness(user_id: int) -> dict[str, object]:
    user = await User.get_or_none(id=user_id)
    missing_user_fields = [
        label
        for field_name, label in REQUIRED_USER_ANALYSIS_FIELDS.items()
        if user is None or _is_missing_value(getattr(user, field_name))
    ]
    latest_record = await get_latest_health_record(user_id)
    if latest_record is None:
        missing_basic_fields = missing_user_fields + list(REQUIRED_BASIC_ANALYSIS_FIELDS.values())
        missing_precision_fields = list(REQUIRED_PRECISION_ANALYSIS_FIELDS.values())
        return {
            "is_ready": False,
            "basic_ready": False,
            "precision_ready": False,
            "latest_health_record_id": None,
            "missing_fields": missing_basic_fields,
            "missing_basic_fields": missing_basic_fields,
            "missing_precision_fields": missing_precision_fields,
            "message": "건강 분석을 시작하려면 먼저 건강 정보를 입력해 주세요.",
        }

    missing_basic_fields = missing_user_fields + [
        label
        for field_name, label in REQUIRED_BASIC_ANALYSIS_FIELDS.items()
        if _is_missing_health_record_field(latest_record, field_name)
    ]
    missing_precision_fields = [
        label
        for field_name, label in REQUIRED_PRECISION_ANALYSIS_FIELDS.items()
        if _is_missing_value(getattr(latest_record, field_name))
    ]
    basic_ready = not missing_basic_fields
    # PRECISION은 BASIC 결과를 기본으로 만들고 질환별로 가능한 X2 수치만 보정한다.
    # 따라서 검진 수치 일부가 없어도 실행 자체는 BASIC 최소 입력이 있으면 가능하다.
    precision_ready = basic_ready
    message = "건강 분석을 실행할 수 있습니다." if basic_ready else "기본 분석에 필요한 항목을 더 입력해 주세요."
    return {
        "is_ready": basic_ready,
        "basic_ready": basic_ready,
        "precision_ready": precision_ready,
        "latest_health_record_id": latest_record.id,
        "missing_fields": missing_basic_fields,
        "missing_basic_fields": missing_basic_fields,
        "missing_precision_fields": missing_precision_fields,
        "message": message,
    }


def _with_calculated_bmi(data: dict) -> dict:
    height = data.get("height_cm")
    weight = data.get("weight_kg")
    if height is None or weight is None:
        return data

    height_decimal = Decimal(str(height))
    weight_decimal = Decimal(str(weight))
    if height_decimal <= 0 or weight_decimal <= 0:
        return data

    height_m = height_decimal / Decimal("100")
    data["bmi"] = (weight_decimal / (height_m * height_m)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return data


def normalize_health_record_source(value: object, default: str = HEALTH_RECORD_SOURCE_MANUAL) -> str:
    normalized = str(value or "").strip().upper()
    return normalized if normalized in HEALTH_RECORD_SOURCES else default


def build_health_record_snapshot_data(record: HealthRecord) -> dict[str, object]:
    return {field_name: getattr(record, field_name, None) for field_name in HEALTH_RECORD_SNAPSHOT_FIELDS}


def _is_missing_value(value: object) -> bool:
    return value is None or value == ""


def _is_missing_health_record_field(record: HealthRecord, field_name: str) -> bool:
    if field_name == "bmi" and not _is_missing_value(record.height_cm) and not _is_missing_value(record.weight_kg):
        return False
    return _is_missing_value(getattr(record, field_name))
