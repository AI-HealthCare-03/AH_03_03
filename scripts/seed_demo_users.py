"""Seed local demo users and related MVP data.

This script is for local MVP frontend testing only. It creates deterministic
demo accounts and screen-friendly data. Do not run it against production or
shared databases.
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")

from tortoise import Tortoise  # noqa: E402

from app.core import config  # noqa: E402
from app.core.db.databases import TORTOISE_ORM  # noqa: E402
from app.core.utils.security import hash_password  # noqa: E402
from app.models.analysis import (  # noqa: E402
    AnalysisResult,
    AnalysisResultFactor,
    AnalysisSnapshot,
    AnalysisType,
    FactorDirection,
    RiskLevel,
)
from app.models.challenges import Challenge, ChallengeLog, UserChallenge, UserChallengeStatus  # noqa: E402
from app.models.diets import DietPhotoResult, DietRecord  # noqa: E402
from app.models.exams import ExamMeasurement, ExamReport, OCRStatus  # noqa: E402
from app.models.health import HealthRecord  # noqa: E402
from app.models.medications import Medication, MedicationRecord  # noqa: E402
from app.models.notifications import Notification  # noqa: E402
from app.models.users import Gender, User, UserConsent  # noqa: E402
from app.services.exams import DUMMY_OCR_MEASUREMENTS  # noqa: E402

DEMO_PASSWORD = "Demo1234!"


@dataclass(frozen=True)
class DemoUserSeed:
    email: str
    login_id: str
    name: str
    nickname: str
    phone_number: str
    birthday: date
    gender: Gender
    risk_profile: str


DEMO_USERS = [
    DemoUserSeed(
        email="demo@example.com",
        login_id="demo_user",
        name="데모사용자",
        nickname="데모",
        phone_number="01010000001",
        birthday=date(1990, 5, 10),
        gender=Gender.FEMALE,
        risk_profile="medium",
    ),
    DemoUserSeed(
        email="demo_high@example.com",
        login_id="demo_high",
        name="고위험데모",
        nickname="고위험",
        phone_number="01010000002",
        birthday=date(1982, 8, 20),
        gender=Gender.MALE,
        risk_profile="high",
    ),
]


async def seed_demo_users() -> None:
    await Tortoise.init(config=TORTOISE_ORM)
    stats = {
        "users_created": 0,
        "users_skipped": 0,
        "health_records_created": 0,
        "analysis_sets_created": 0,
        "medications_created": 0,
        "diet_records_created": 0,
        "exam_reports_created": 0,
        "notifications_created": 0,
        "user_challenges_created": 0,
        "challenge_logs_created": 0,
    }
    try:
        for seed in DEMO_USERS:
            user, created = await _get_or_create_user(seed)
            stats["users_created" if created else "users_skipped"] += 1
            stats["health_records_created"] += await _seed_health_records(user, seed.risk_profile)
            stats["analysis_sets_created"] += await _seed_analysis_results(user)
            stats["medications_created"] += await _seed_medications(user, seed.risk_profile)
            stats["diet_records_created"] += await _seed_diets(user, seed.risk_profile)
            stats["exam_reports_created"] += await _seed_exam_report(user)
            stats["notifications_created"] += await _seed_notifications(user, seed.risk_profile)
            challenge_stats = await _seed_user_challenges(user)
            stats["user_challenges_created"] += challenge_stats["user_challenges_created"]
            stats["challenge_logs_created"] += challenge_stats["challenge_logs_created"]
    finally:
        await Tortoise.close_connections()

    print("===== MVP Demo User Seed =====")
    print("This seed is for local MVP demos only.")
    print(f"demo_password: {DEMO_PASSWORD}")
    for key, value in stats.items():
        print(f"{key}: {value}")


async def _get_or_create_user(seed: DemoUserSeed) -> tuple[User, bool]:
    existing = await User.get_or_none(email=seed.email)
    if existing is not None:
        return existing, False

    user = await User.create(
        login_id=seed.login_id,
        auth_provider="local",
        email=seed.email,
        hashed_password=hash_password(DEMO_PASSWORD),
        name=seed.name,
        nickname=seed.nickname,
        gender=seed.gender,
        birthday=seed.birthday,
        phone_number=seed.phone_number,
        role="USER",
        is_active=True,
        email_verified_at=datetime.now(config.TIMEZONE),
    )
    await UserConsent.get_or_create(
        user=user,
        defaults={
            "terms_agreed": True,
            "privacy_agreed": True,
            "sensitive_data_agreed": True,
            "marketing_agreed": False,
        },
    )
    return user, True


async def _seed_health_records(user: User, risk_profile: str) -> int:
    if await HealthRecord.filter(user=user).count() >= 3:
        return 0

    now = datetime.now(config.TIMEZONE)
    values = _health_values(risk_profile)
    created_count = 0
    for index, value in enumerate(values):
        measured_at = now - timedelta(days=(2 - index) * 7)
        existing = await HealthRecord.get_or_none(user=user, measured_at=measured_at)
        if existing is not None:
            continue
        await HealthRecord.create(user=user, measured_at=measured_at, **value)
        created_count += 1
    return created_count


def _health_values(risk_profile: str) -> list[dict[str, Any]]:
    if risk_profile == "high":
        return [
            _health_payload(170, 88, 30.4, 96, 148, 94, 138, "6.9", 258, 172, 36, 224, 1, 1, 2, "5.5"),
            _health_payload(170, 87, 30.1, 95, 144, 92, 132, "6.7", 248, 164, 38, 210, 1, 1, 2, "5.8"),
            _health_payload(170, 86, 29.8, 94, 142, 90, 128, "6.5", 242, 158, 39, 198, 1, 1, 3, "6.0"),
        ]
    return [
        _health_payload(164, 64, 23.8, 78, 124, 78, 96, "5.4", 192, 112, 52, 128, 0, 0, 4, "7.0"),
        _health_payload(164, 65, 24.2, 79, 128, 80, 101, "5.6", 202, 124, 48, 142, 0, 0, 3, "6.8"),
        _health_payload(164, 66, 24.5, 80, 132, 82, 108, "5.8", 210, 132, 46, 158, 0, 0, 3, "6.5"),
    ]


def _health_payload(
    height: int,
    weight: int,
    bmi: float,
    waist: int,
    systolic: int,
    diastolic: int,
    glucose: int,
    hba1c: str,
    total_cholesterol: int,
    ldl: int,
    hdl: int,
    triglyceride: int,
    is_smoker: int,
    drinks_alcohol: int,
    exercise_days: int,
    sleep_hours: str,
) -> dict[str, Any]:
    return {
        "height_cm": Decimal(str(height)),
        "weight_kg": Decimal(str(weight)),
        "bmi": Decimal(str(bmi)),
        "waist_cm": Decimal(str(waist)),
        "systolic_bp": systolic,
        "diastolic_bp": diastolic,
        "fasting_glucose": glucose,
        "hba1c": Decimal(hba1c),
        "total_cholesterol": total_cholesterol,
        "ldl_cholesterol": ldl,
        "hdl_cholesterol": hdl,
        "triglyceride": triglyceride,
        "has_diabetes": glucose >= 126,
        "has_obesity": bmi >= 25,
        "has_dyslipidemia": ldl >= 130 or triglyceride >= 150,
        "has_hypertension": systolic >= 130 or diastolic >= 80,
        "is_smoker": bool(is_smoker),
        "drinks_alcohol": bool(drinks_alcohol),
        "exercise_days_per_week": exercise_days,
        "sleep_hours": Decimal(sleep_hours),
    }


async def _seed_analysis_results(user: User) -> int:
    latest_record = await HealthRecord.filter(user=user).order_by("-measured_at").first()
    if latest_record is None:
        return 0

    created_count = 0
    for analysis_type in AnalysisType:
        existing = await AnalysisResult.get_or_none(health_record=latest_record, analysis_type=analysis_type)
        if existing is None:
            score = _demo_analysis_score(latest_record, analysis_type)
            risk_level = _demo_risk_level(score)
            existing = await AnalysisResult.create(
                user=user,
                health_record=latest_record,
                analysis_type=analysis_type,
                risk_score=score,
                risk_level=risk_level,
                summary=_demo_guide_message(analysis_type, risk_level),
                model_name="dummy_rule_based",
                model_version="mvp-demo-v1",
                analyzed_at=datetime.now(config.TIMEZONE),
            )
            created_count += 1

        await _ensure_analysis_factor(existing, latest_record)
        await _ensure_analysis_snapshot(existing, latest_record)
    return 1 if created_count > 0 else 0


def _demo_analysis_score(record: HealthRecord, analysis_type: AnalysisType) -> Decimal:
    if analysis_type == AnalysisType.DIABETES:
        return _demo_diabetes_score(record)
    if analysis_type == AnalysisType.OBESITY:
        return _demo_obesity_score(record)
    if analysis_type == AnalysisType.DYSLIPIDEMIA:
        return _demo_dyslipidemia_score(record)
    return _demo_hypertension_score(record)


def _demo_diabetes_score(record: HealthRecord) -> Decimal:
    if record.fasting_glucose is not None and record.fasting_glucose >= 126:
        return Decimal("0.78000")
    if record.hba1c is not None and record.hba1c >= Decimal("5.7"):
        return Decimal("0.56000")
    return Decimal("0.25000")


def _demo_obesity_score(record: HealthRecord) -> Decimal:
    if record.bmi is not None and record.bmi >= Decimal("30"):
        return Decimal("0.82000")
    if record.bmi is not None and record.bmi >= Decimal("25"):
        return Decimal("0.68000")
    return Decimal("0.22000")


def _demo_dyslipidemia_score(record: HealthRecord) -> Decimal:
    if record.ldl_cholesterol is not None and record.ldl_cholesterol >= 160:
        return Decimal("0.78000")
    if record.triglyceride is not None and record.triglyceride >= 150:
        return Decimal("0.52000")
    return Decimal("0.25000")


def _demo_hypertension_score(record: HealthRecord) -> Decimal:
    if (record.systolic_bp is not None and record.systolic_bp >= 140) or (
        record.diastolic_bp is not None and record.diastolic_bp >= 90
    ):
        return Decimal("0.78000")
    if (record.systolic_bp is not None and record.systolic_bp >= 130) or (
        record.diastolic_bp is not None and record.diastolic_bp >= 80
    ):
        return Decimal("0.52000")
    return Decimal("0.18000")


def _demo_risk_level(score: Decimal) -> RiskLevel:
    if score >= Decimal("0.70"):
        return RiskLevel.HIGH
    if score >= Decimal("0.40"):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _demo_guide_message(analysis_type: AnalysisType, risk_level: RiskLevel) -> str:
    label = {
        AnalysisType.DIABETES: "당뇨",
        AnalysisType.OBESITY: "비만",
        AnalysisType.DYSLIPIDEMIA: "이상지질혈증",
        AnalysisType.HYPERTENSION: "고혈압",
    }[analysis_type]
    return f"{label} {risk_level.value} 구간입니다. MVP 데모용 더미 룰이며 실제 의료 진단이 아닙니다."


async def _ensure_analysis_factor(result: AnalysisResult, record: HealthRecord) -> None:
    if await AnalysisResultFactor.filter(analysis_result=result).count() > 0:
        return

    factor_key, factor_name, factor_value = {
        AnalysisType.DIABETES: ("fasting_glucose", "공복혈당", record.fasting_glucose),
        AnalysisType.OBESITY: ("bmi", "BMI", record.bmi),
        AnalysisType.DYSLIPIDEMIA: ("ldl_cholesterol", "LDL 콜레스테롤", record.ldl_cholesterol),
        AnalysisType.HYPERTENSION: (
            "blood_pressure",
            "혈압",
            f"{record.systolic_bp or '-'}/{record.diastolic_bp or '-'}",
        ),
    }[result.analysis_type]

    await AnalysisResultFactor.create(
        analysis_result=result,
        factor_key="dummy_risk_score",
        factor_name="더미 위험도 점수",
        factor_value=str(result.risk_score),
        contribution_score=result.risk_score,
        direction=FactorDirection.NEUTRAL,
        display_order=0,
    )
    await AnalysisResultFactor.create(
        analysis_result=result,
        factor_key=factor_key,
        factor_name=factor_name,
        factor_value=str(factor_value) if factor_value is not None else None,
        contribution_score=Decimal("0.400000"),
        direction=FactorDirection.POSITIVE,
        display_order=1,
    )


async def _ensure_analysis_snapshot(result: AnalysisResult, record: HealthRecord) -> None:
    if await AnalysisSnapshot.filter(analysis_result=result).count() > 0:
        return

    analysis_type = result.analysis_type.value
    await AnalysisSnapshot.create(
        analysis_result=result,
        input_payload={
            "analysis_type": analysis_type,
            "input_features": {
                "bmi": _json_value(record.bmi),
                "systolic_bp": record.systolic_bp,
                "diastolic_bp": record.diastolic_bp,
                "fasting_glucose": record.fasting_glucose,
                "hba1c": _json_value(record.hba1c),
                "ldl_cholesterol": record.ldl_cholesterol,
                "hdl_cholesterol": record.hdl_cholesterol,
                "triglyceride": record.triglyceride,
            },
        },
        output_payload={
            "model_outputs": {
                analysis_type: {
                    "risk_score": _json_value(result.risk_score),
                    "risk_level": result.risk_level.value,
                }
            },
            "rule_outputs": {
                "rule_engine": "seed_dummy_rule_based",
                "note": "로컬 MVP 데모용 더미 룰이며 실제 의료 진단이 아닙니다.",
            },
            "final_outputs": {
                "risk_level": result.risk_level.value,
                "guide_message": result.summary,
            },
        },
        shap_payload={"note": "실제 SHAP 계산이 아닌 seed 데모용 factor입니다."},
        model_payload={"model_name": "dummy_rule_based", "model_version": "mvp-demo-v1"},
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    return value


async def _seed_medications(user: User, risk_profile: str) -> int:
    medication_names = ["메트포르민", "오메가3", "종합비타민"]
    if risk_profile == "high":
        medication_names.insert(0, "혈압약")

    created_count = 0
    for index, name in enumerate(medication_names):
        medication, created = await Medication.get_or_create(
            user=user,
            name=name,
            defaults={
                "medication_type": "MEDICINE" if name in {"혈압약", "메트포르민"} else "SUPPLEMENT",
                "dosage": "1정",
                "frequency": "매일 1회",
                "reminder_time": None,
                "is_active": True,
                "memo": "로컬 MVP 데모용 복약 데이터",
            },
        )
        if created:
            created_count += 1
        await _seed_medication_record(user, medication, index)
    return created_count


async def _seed_medication_record(user: User, medication: Medication, index: int) -> None:
    scheduled_at = datetime.now(config.TIMEZONE).replace(hour=8 + index, minute=0, second=0, microsecond=0)
    existing = await MedicationRecord.get_or_none(medication=medication, user=user, scheduled_at=scheduled_at)
    if existing is not None:
        return
    await MedicationRecord.create(
        medication=medication,
        user=user,
        scheduled_at=scheduled_at,
        taken_at=scheduled_at + timedelta(minutes=10) if index % 2 == 0 else None,
        is_taken=index % 2 == 0,
        status="TAKEN" if index % 2 == 0 else "PENDING",
        memo="데모 복약 기록",
    )


async def _seed_diets(user: User, risk_profile: str) -> int:
    if await DietRecord.filter(user=user).count() >= 5:
        return 0

    now = datetime.now(config.TIMEZONE)
    meals = [
        ("BREAKFAST", "현미밥, 달걀, 나물 반찬", 82.0),
        ("LUNCH", "닭가슴살 샐러드와 고구마", 88.0),
        ("DINNER", "잡곡밥, 생선구이, 된장국", 76.0),
        ("SNACK", "무가당 요거트와 견과류", 84.0),
        ("DINNER", "외식 메뉴와 탄산음료", 58.0 if risk_profile == "high" else 68.0),
    ]
    created_count = 0
    for index, (meal_type, description, score) in enumerate(meals):
        existing = await DietRecord.get_or_none(user=user, description=description)
        if existing is not None:
            continue
        record = await DietRecord.create(
            user=user,
            meal_type=meal_type,
            meal_time=now - timedelta(days=index),
            description=description,
            image_path=f"/demo/diets/{user.login_id}_{index + 1}.jpg",
            detected_foods=[{"name": food.strip(), "confidence": 0.92} for food in description.split(",")],
            nutrition_summary={"calories": 520 + index * 35, "carbs_g": 65, "protein_g": 28, "fat_g": 16},
            diet_score=score,
            diet_feedback="MVP 데모용 식단 분석 결과입니다.",
            analysis_method="DUMMY",
            memo="로컬 MVP 데모 식단",
        )
        await DietPhotoResult.create(
            diet_record=record,
            detected_foods=record.detected_foods,
            confidence_payload={"avg_confidence": 0.92, "is_demo": True},
            raw_output={"source": "seed_demo_users", "is_dummy": True},
            is_dummy=True,
        )
        created_count += 1
    return created_count


async def _seed_exam_report(user: User) -> int:
    original_filename = f"{user.login_id}_demo_exam.pdf"
    report, created = await ExamReport.get_or_create(
        user=user,
        original_filename=original_filename,
        defaults={
            "file_path": f"/demo/exams/{original_filename}",
            "exam_date": date.today() - timedelta(days=30),
            "ocr_status": OCRStatus.SUCCESS,
            "is_confirmed": True,
            "uploaded_at": datetime.now(config.TIMEZONE) - timedelta(days=1),
            "confirmed_at": datetime.now(config.TIMEZONE),
        },
    )
    for key, name, value, unit in DUMMY_OCR_MEASUREMENTS:
        await ExamMeasurement.get_or_create(
            exam_report=report,
            measurement_key=key,
            defaults={
                "measurement_name": name,
                "value": value,
                "unit": unit,
                "ocr_confidence": Decimal("0.9700"),
                "is_user_confirmed": True,
            },
        )
    return 1 if created else 0


async def _seed_notifications(user: User, risk_profile: str) -> int:
    notifications = [
        ("ANALYSIS", "건강 분석 완료", "4종 위험도 분석 결과가 생성되었습니다.", False),
        ("CHALLENGE", "챌린지 리마인더", "오늘 챌린지를 완료하고 수행률을 올려보세요.", False),
        ("MEDICATION", "복약 리마인더", "오늘 예정된 복약/영양제 기록을 확인해 주세요.", True),
        ("DIET", "식단 기록 안내", "오늘 식단을 기록하면 추천 결과가 더 풍부해집니다.", False),
        ("INQUIRY", "문의 답변 안내", "1:1 문의 답변 예시는 관리자 기능에서 확인할 수 있습니다.", True),
    ]
    if risk_profile == "high":
        notifications.append(("HEALTH", "고위험 지표 확인", "혈압과 혈당 지표가 높게 표시되었습니다.", False))

    created_count = 0
    for notification_type, title, message, is_read in notifications:
        existing = await Notification.get_or_none(user=user, title=title, message=message)
        if existing is not None:
            continue
        await Notification.create(
            user=user,
            notification_type=notification_type,
            title=title,
            message=message,
            is_read=is_read,
            read_at=datetime.now(config.TIMEZONE) if is_read else None,
        )
        created_count += 1
    return created_count


async def _seed_user_challenges(user: User) -> dict[str, int]:
    active_challenges = await Challenge.all().order_by("id").limit(3)
    stats = {"user_challenges_created": 0, "challenge_logs_created": 0}
    for challenge in active_challenges:
        user_challenge, created = await UserChallenge.get_or_create(
            user=user,
            challenge=challenge,
            defaults={
                "status": UserChallengeStatus.JOINED,
                "started_at": datetime.now(config.TIMEZONE) - timedelta(days=4),
            },
        )
        if created:
            stats["user_challenges_created"] += 1

        for day_offset in range(5):
            log_date = date.today() - timedelta(days=day_offset)
            _, log_created = await ChallengeLog.get_or_create(
                user_challenge=user_challenge,
                log_date=log_date,
                defaults={
                    "is_completed": day_offset % 2 == 0,
                    "memo": "로컬 MVP 데모 챌린지 로그",
                },
            )
            if log_created:
                stats["challenge_logs_created"] += 1
    return stats


if __name__ == "__main__":
    asyncio.run(seed_demo_users())
