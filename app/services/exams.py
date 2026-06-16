import logging
import re
from datetime import datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core import config
from app.dtos.exams import (
    ExamConfirmMeasurementRequest,
    ExamConfirmRequest,
    ExamMeasurementCreateRequest,
    ExamMeasurementUpdateRequest,
    ExamReportCreateRequest,
    ExamReportUpdateRequest,
)
from app.models.exams import ExamMeasurement, ExamReport, OCRStatus
from app.repositories import exam_repository, health_repository
from app.services.health import (
    HEALTH_RECORD_SOURCE_OCR,
    _with_calculated_bmi,
    build_health_record_snapshot_data,
)
from app.services.storage import get_storage_service, normalize_storage_key

EXAM_MEASUREMENT_METADATA = {
    "height": ("키", "cm"),
    "height_cm": ("키", "cm"),
    "weight": ("몸무게", "kg"),
    "weight_kg": ("몸무게", "kg"),
    "bmi": ("체질량지수", "kg/m2"),
    "waist": ("허리둘레", "cm"),
    "waist_cm": ("허리둘레", "cm"),
    "systolic_bp": ("수축기혈압", "mmHg"),
    "diastolic_bp": ("이완기혈압", "mmHg"),
    "fasting_glucose": ("공복혈당", "mg/dL"),
    "hba1c": ("당화혈색소", "%"),
    "hb": ("혈색소", "g/dL"),
    "hemoglobin": ("혈색소", "g/dL"),
    "total_cholesterol": ("총콜레스테롤", "mg/dL"),
    "ldl": ("LDL 콜레스테롤", "mg/dL"),
    "ldl_cholesterol": ("LDL 콜레스테롤", "mg/dL"),
    "hdl": ("HDL 콜레스테롤", "mg/dL"),
    "hdl_cholesterol": ("HDL 콜레스테롤", "mg/dL"),
    "triglyceride": ("중성지방", "mg/dL"),
    "ast": ("AST", "U/L"),
    "alt": ("ALT", "U/L"),
    "gamma_gtp": ("감마GTP", "U/L"),
    "ggt": ("감마GTP", "U/L"),
    "creatinine": ("크레아티닌", "mg/dL"),
    "egfr": ("eGFR", "mL/min/1.73m2"),
    "urine_protein": ("요단백", None),
}

EXAM_MEASUREMENT_TO_HEALTH_FIELD = {
    "height": "height_cm",
    "height_cm": "height_cm",
    "weight": "weight_kg",
    "weight_kg": "weight_kg",
    "bmi": "bmi",
    "waist": "waist_cm",
    "waist_cm": "waist_cm",
    "systolic_bp": "systolic_bp",
    "diastolic_bp": "diastolic_bp",
    "fasting_glucose": "fasting_glucose",
    "hba1c": "hba1c",
    "total_cholesterol": "total_cholesterol",
    "triglyceride": "triglyceride",
    "hdl": "hdl_cholesterol",
    "hdl_cholesterol": "hdl_cholesterol",
    "ldl": "ldl_cholesterol",
    "ldl_cholesterol": "ldl_cholesterol",
}

HEALTH_INT_FIELDS = {
    "systolic_bp",
    "diastolic_bp",
    "fasting_glucose",
    "total_cholesterol",
    "ldl_cholesterol",
    "hdl_cholesterol",
    "triglyceride",
}

NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
logger = logging.getLogger(__name__)
NON_NUMERIC_EXAM_VALUE_MARKERS = {
    "-",
    "비해당",
    "해당없음",
    "해당없슴",
    "없음",
    "미실시",
    "검사안함",
}
EXAM_UPLOAD_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}


async def create_exam_report(user_id: int, request: ExamReportCreateRequest) -> ExamReport:
    return await exam_repository.create_exam_report(user_id, request.model_dump())


async def get_exam_report(exam_report_id: int) -> ExamReport | None:
    return await exam_repository.get_exam_report_by_id(exam_report_id)


async def list_exam_reports(user_id: int, limit: int = 20, offset: int = 0) -> list[ExamReport]:
    return await exam_repository.list_exam_reports_by_user(user_id, limit=limit, offset=offset)


async def update_exam_report(exam_report_id: int, request: ExamReportUpdateRequest) -> ExamReport | None:
    data = request.model_dump(exclude_unset=True)
    return await exam_repository.update_exam_report(exam_report_id, data)


async def store_exam_ocr_upload(
    report: ExamReport,
    image_bytes: bytes,
    image_media_type: str | None,
    image_filename: str | None,
) -> ExamReport:
    storage_key = _build_exam_upload_key(report, image_media_type, image_filename)
    stored_key = get_storage_service().save_bytes(image_bytes, storage_key, content_type=image_media_type)
    updated = await update_exam_report(
        int(report.id),
        ExamReportUpdateRequest(
            file_path=stored_key,
            original_filename=image_filename or report.original_filename,
            ocr_status=OCRStatus.PENDING,
        ),
    )
    return updated or report


def _upload_storage_root() -> Path:
    root = Path(config.UPLOAD_STORAGE_DIR)
    if not root.is_absolute():
        root = Path.cwd() / root
    return root


def _build_exam_upload_key(
    report: ExamReport,
    image_media_type: str | None,
    image_filename: str | None,
) -> str:
    extension = _upload_extension(image_media_type, image_filename)
    return normalize_storage_key(f"exams/{report.user_id}/{report.id}/{uuid4().hex}/source{extension}")


def _build_exam_upload_path(
    report: ExamReport,
    image_media_type: str | None,
    image_filename: str | None,
) -> Path:
    extension = _upload_extension(image_media_type, image_filename)
    return _upload_storage_root() / "exams" / str(report.user_id) / str(report.id) / f"source{extension}"


def _stored_path_for_db(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _upload_extension(image_media_type: str | None, image_filename: str | None) -> str:
    media_type = (image_media_type or "").lower().split(";", 1)[0].strip()
    if media_type == "application/pdf":
        return ".pdf"
    if media_type == "image/png":
        return ".png"
    if media_type == "image/webp":
        return ".webp"
    if media_type in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    suffix = Path(image_filename or "").suffix.lower()
    return suffix if suffix in EXAM_UPLOAD_MEDIA_TYPES else ".jpg"


def _resolve_stored_upload_path(file_path: str) -> Path:
    path = Path(file_path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _media_type_from_upload_path(path: Path, filename: str | None = None) -> str:
    suffix = path.suffix.lower()
    if suffix in EXAM_UPLOAD_MEDIA_TYPES:
        return EXAM_UPLOAD_MEDIA_TYPES[suffix]
    filename_suffix = Path(filename or "").suffix.lower()
    return EXAM_UPLOAD_MEDIA_TYPES.get(filename_suffix, "image/jpeg")


async def create_exam_measurement(exam_report_id: int, request: ExamMeasurementCreateRequest) -> ExamMeasurement:
    return await exam_repository.create_exam_measurement(exam_report_id, request.model_dump())


async def get_exam_measurement(measurement_id: int) -> ExamMeasurement | None:
    return await exam_repository.get_exam_measurement_by_id(measurement_id)


async def create_exam_measurements(
    exam_report_id: int, measurements: list[ExamMeasurementCreateRequest]
) -> list[ExamMeasurement]:
    data = [measurement.model_dump() for measurement in measurements]
    return await exam_repository.create_exam_measurements(exam_report_id, data)


async def list_exam_measurements(exam_report_id: int) -> list[ExamMeasurement]:
    return await exam_repository.list_exam_measurements(exam_report_id)


async def get_latest_confirmed_exam_measurements_for_analysis(
    user_id: int,
) -> tuple[ExamReport | None, list[ExamMeasurement]]:
    report = (
        await ExamReport.filter(user_id=user_id, is_confirmed=True, ocr_status=OCRStatus.CONFIRMED)
        .order_by("-confirmed_at", "-updated_at", "-uploaded_at", "-id")
        .first()
    )
    if report is None:
        return None, []
    return report, await list_exam_measurements(int(report.id))


async def update_exam_measurement(measurement_id: int, request: ExamMeasurementUpdateRequest) -> ExamMeasurement | None:
    data = request.model_dump(exclude_unset=True)
    return await exam_repository.update_exam_measurement(measurement_id, data)


async def confirm_exam_report(exam_report_id: int, request: ExamConfirmRequest | None = None) -> ExamReport | None:
    report = await get_exam_report(exam_report_id)
    if report is None:
        return None

    if request is not None and request.measurements:
        await apply_confirm_measurement_updates(exam_report_id, request.measurements)

    measurements = await list_exam_measurements(exam_report_id)
    await sync_exam_measurements_to_latest_health_record(report, measurements)
    return await exam_repository.confirm_exam_report(exam_report_id)


async def apply_confirm_measurement_updates(
    exam_report_id: int, measurements: list[ExamConfirmMeasurementRequest]
) -> list[ExamMeasurement]:
    existing_measurements = await list_exam_measurements(exam_report_id)
    existing_by_key = {measurement.measurement_key: measurement for measurement in existing_measurements}
    saved_measurements: list[ExamMeasurement] = []

    for measurement in measurements:
        key = measurement.measurement_key or measurement.key
        if key is None or not key.strip():
            continue
        value = None if measurement.value is None else str(measurement.value)
        existing = existing_by_key.get(key)
        if existing is None:
            saved_measurements.append(
                await create_exam_measurement(
                    exam_report_id,
                    ExamMeasurementCreateRequest(
                        measurement_key=key,
                        measurement_name=measurement.measurement_name or key,
                        value=value,
                        unit=measurement.unit,
                        ocr_confidence=measurement.ocr_confidence,
                        is_user_confirmed=measurement.is_user_confirmed
                        if measurement.is_user_confirmed is not None
                        else True,
                    ),
                )
            )
            continue

        update_data: dict[str, Any] = {
            "measurement_key": key,
            "is_user_confirmed": measurement.is_user_confirmed if measurement.is_user_confirmed is not None else True,
        }
        if measurement.measurement_name is not None:
            update_data["measurement_name"] = measurement.measurement_name
        if measurement.value is not None:
            update_data["value"] = value
        if measurement.unit is not None:
            update_data["unit"] = measurement.unit
        if measurement.ocr_confidence is not None:
            update_data["ocr_confidence"] = measurement.ocr_confidence

        update_request = ExamMeasurementUpdateRequest(**update_data)
        updated = await update_exam_measurement(existing.id, update_request)
        saved_measurements.append(updated or existing)
    return saved_measurements


async def sync_exam_measurements_to_latest_health_record(
    report: ExamReport, measurements: list[ExamMeasurement]
) -> dict[str, Any]:
    data = build_health_record_update_from_exam_measurements(measurements)
    if not data:
        return {}

    measured_at = (
        datetime.combine(report.exam_date, time.min, tzinfo=config.TIMEZONE)
        if report.exam_date
        else datetime.now(config.TIMEZONE)
    )
    latest_record = await health_repository.get_latest_health_record_by_user(report.user_id)
    if latest_record is not None:
        existing_payload = {field_name: getattr(latest_record, field_name, None) for field_name in data}
        missing_only_data = merge_health_record_with_exam_missing_only(existing_payload, data)
        snapshot_data = {
            **build_health_record_snapshot_data(latest_record),
            **missing_only_data,
            "source": HEALTH_RECORD_SOURCE_OCR,
            "measured_at": measured_at,
        }
        await health_repository.create_health_record(report.user_id, snapshot_data)
        return missing_only_data

    await health_repository.create_health_record(
        report.user_id,
        {"measured_at": measured_at, "source": HEALTH_RECORD_SOURCE_OCR, **data},
    )
    return data


def merge_health_record_with_exam_missing_only(
    existing_payload: dict[str, Any], exam_payload: dict[str, Any]
) -> dict[str, Any]:
    return {
        field_name: exam_value
        for field_name, exam_value in exam_payload.items()
        if _is_missing_health_record_sync_value(existing_payload.get(field_name))
    }


def _is_missing_health_record_sync_value(value: Any) -> bool:
    return value is None or value == ""


def build_health_record_update_from_exam_measurements(measurements: list[ExamMeasurement]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for measurement in measurements:
        health_field = EXAM_MEASUREMENT_TO_HEALTH_FIELD.get(measurement.measurement_key)
        if health_field is None:
            continue
        parsed_value = parse_exam_measurement_number(measurement.value)
        if parsed_value is None:
            continue
        data[health_field] = _coerce_health_record_value(health_field, parsed_value)
    return _with_calculated_bmi(data)


def parse_exam_measurement_number(value: object) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int | float):
        return Decimal(str(value))

    text = str(value).strip()
    if not text:
        return None
    normalized_text = re.sub(r"\s+", "", text)
    if normalized_text == "-" or any(
        marker in normalized_text for marker in NON_NUMERIC_EXAM_VALUE_MARKERS if marker != "-"
    ):
        return None
    match = NUMBER_PATTERN.search(text.replace(",", ""))
    if match is None:
        return None
    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return None


def _coerce_health_record_value(field_name: str, value: Decimal) -> int | Decimal:
    if field_name in HEALTH_INT_FIELDS:
        return int(value)
    return value


def _read_stored_exam_upload(file_path: str | None) -> tuple[bytes, str, str] | None:
    if not file_path:
        return None

    storage = get_storage_service()
    try:
        if storage.exists(file_path):
            return storage.read_bytes(file_path), _media_type_from_storage_key(file_path), Path(file_path).name
    except Exception:
        logger.warning("Storage 기반 건강검진 업로드 읽기 실패: %s", file_path, exc_info=True)

    legacy_path = _resolve_stored_upload_path(file_path)
    if legacy_path.exists() and legacy_path.is_file():
        return legacy_path.read_bytes(), _media_type_from_upload_path(legacy_path), legacy_path.name
    return None


def _media_type_from_storage_key(key: str, filename: str | None = None) -> str:
    suffix = Path(key).suffix.lower()
    if suffix in EXAM_UPLOAD_MEDIA_TYPES:
        return EXAM_UPLOAD_MEDIA_TYPES[suffix]
    filename_suffix = Path(filename or "").suffix.lower()
    return EXAM_UPLOAD_MEDIA_TYPES.get(filename_suffix, "image/jpeg")


def _measurement_tuples_from_mapping(values: dict[str, Any]) -> list[tuple[str, str, str, str | None]]:
    measurements = []
    for key, (measurement_name, unit) in EXAM_MEASUREMENT_METADATA.items():
        if key not in values:
            continue
        value = values.get(key)
        if value is None or value == "":
            continue
        if key == "urine_protein":
            normalized_value = str(value).strip()
            if not normalized_value or normalized_value in NON_NUMERIC_EXAM_VALUE_MARKERS:
                continue
            measurements.append((key, measurement_name, normalized_value, unit))
            continue
        if parse_exam_measurement_number(value) is None:
            continue
        measurement_key = key
        measurements.append((measurement_key, measurement_name, str(value), unit))
    return measurements
