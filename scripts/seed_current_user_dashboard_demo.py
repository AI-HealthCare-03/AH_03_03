"""Seed rich local MVP demo data for one existing user.

This script is for local MVP demos only. Do not run it against production or
shared databases. It does not replace Aerich migrations and does not change
database schema.

Usage:
    DB_HOST=localhost uv run python scripts/seed_current_user_dashboard_demo.py --email aszx91@gmail.com
"""

import argparse
import asyncio
import os
import sys
from datetime import date, datetime, time, timedelta
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
from app.models.analysis import (  # noqa: E402
    AnalysisResult,
    AnalysisResultFactor,
    AnalysisSnapshot,
    AnalysisType,
    FactorDirection,
    RiskLevel,
)
from app.models.challenges import (  # noqa: E402
    Challenge,
    ChallengeCategory,
    ChallengeLog,
    ChallengeStatus,
    UserChallenge,
    UserChallengeStatus,
)
from app.models.diets import DietPhotoResult, DietRecord  # noqa: E402
from app.models.exams import ExamMeasurement, ExamReport, OCRStatus  # noqa: E402
from app.models.health import HealthRecord  # noqa: E402
from app.models.medications import Medication, MedicationRecord  # noqa: E402
from app.models.notifications import Notification  # noqa: E402
from app.models.users import User  # noqa: E402

DEMO_CHALLENGES = [
    {
        "title": "식후 10분 산책",
        "description": "식사 후 가볍게 걷고 혈당 관리 습관을 만들어보세요.",
        "category": ChallengeCategory.EXERCISE,
        "target_metric": "walking",
        "target_value": "10분",
        "duration_days": 7,
    },
    {
        "title": "단 음료 대신 물 마시기",
        "description": "하루 한 번 단 음료를 물로 바꾸는 챌린지입니다.",
        "category": ChallengeCategory.HABIT,
        "target_metric": "water",
        "target_value": "1회",
        "duration_days": 7,
    },
    {
        "title": "저녁 저염 식단 실천",
        "description": "저녁 식사에서 나트륨을 줄이고 균형 잡힌 식단을 기록해보세요.",
        "category": ChallengeCategory.DIET,
        "target_metric": "diet",
        "target_value": "저염",
        "duration_days": 14,
    },
]


async def seed_current_user(email: str) -> dict[str, int]:
    await Tortoise.init(config=TORTOISE_ORM)
    stats = {
        "health_records_created": 0,
        "health_records_skipped": 0,
        "analysis_results_created": 0,
        "analysis_results_skipped": 0,
        "analysis_factors_created": 0,
        "analysis_snapshots_created": 0,
        "challenges_created": 0,
        "user_challenges_created": 0,
        "user_challenges_skipped": 0,
        "challenge_logs_created": 0,
        "challenge_logs_skipped": 0,
        "diet_records_created": 0,
        "diet_records_skipped": 0,
        "diet_photo_results_created": 0,
        "medications_created": 0,
        "medications_skipped": 0,
        "medication_records_created": 0,
        "medication_records_skipped": 0,
        "exam_reports_created": 0,
        "exam_reports_skipped": 0,
        "exam_measurements_created": 0,
        "notifications_created": 0,
        "notifications_skipped": 0,
    }
    try:
        user = await User.get_or_none(email=email)
        if user is None:
            raise RuntimeError(f"User not found: {email}")

        health_stats = await _seed_health_records(user)
        stats.update(_merge_stats(stats, health_stats))
        analysis_stats = await _seed_analysis(user)
        stats.update(_merge_stats(stats, analysis_stats))
        challenge_stats = await _seed_challenges(user)
        stats.update(_merge_stats(stats, challenge_stats))
        diet_stats = await _seed_diets(user)
        stats.update(_merge_stats(stats, diet_stats))
        medication_stats = await _seed_medications(user)
        stats.update(_merge_stats(stats, medication_stats))
        exam_stats = await _seed_exam(user)
        stats.update(_merge_stats(stats, exam_stats))
        notification_stats = await _seed_notifications(user)
        stats.update(_merge_stats(stats, notification_stats))
    finally:
        await Tortoise.close_connections()
    return stats


def _merge_stats(base: dict[str, int], patch: dict[str, int]) -> dict[str, int]:
    merged = dict(base)
    for key, value in patch.items():
        merged[key] = merged.get(key, 0) + value
    return merged


async def _seed_health_records(user: User) -> dict[str, int]:
    stats = {"health_records_created": 0, "health_records_skipped": 0}
    today = date.today()
    for index in range(10):
        record_date = today - timedelta(days=9 - index)
        start_at = _at_time(record_date, 0)
        end_at = start_at + timedelta(days=1)
        existing = await HealthRecord.filter(user=user, measured_at__gte=start_at, measured_at__lt=end_at).first()
        if existing is not None:
            stats["health_records_skipped"] += 1
            continue

        weight = Decimal("80.0") - Decimal(str(index)) * Decimal("0.09")
        height = Decimal("172.0")
        bmi = weight / ((height / Decimal("100")) ** 2)
        await HealthRecord.create(
            user=user,
            measured_at=_at_time(record_date, 8),
            height_cm=height,
            weight_kg=weight.quantize(Decimal("0.01")),
            bmi=bmi.quantize(Decimal("0.01")),
            systolic_bp=132 - min(index, 6),
            diastolic_bp=84 - min(index, 6),
            fasting_glucose=108 - min(index, 7),
            hba1c=Decimal("5.70"),
            total_cholesterol=186,
            triglyceride=130,
            hdl_cholesterol=52,
            ldl_cholesterol=112,
            waist_cm=Decimal("86.00"),
            has_diabetes=False,
            has_obesity=True,
            has_dyslipidemia=False,
            has_hypertension=True,
            occupation_code="OFFICE",
            family_htn="YES",
            family_dm="NO",
            family_dyslipidemia="UNKNOWN",
            smoking_status="NON_SMOKER",
            drinking_frequency="MONTHLY_2_4",
            drinking_amount="LIGHT",
            walking_days_per_week=4,
            strength_days_per_week=2,
            sleep_hours=Decimal("7.00"),
        )
        stats["health_records_created"] += 1
    return stats


async def _seed_analysis(user: User) -> dict[str, int]:
    stats = {
        "analysis_results_created": 0,
        "analysis_results_skipped": 0,
        "analysis_factors_created": 0,
        "analysis_snapshots_created": 0,
    }
    record = await HealthRecord.filter(user=user).order_by("-measured_at").first()
    if record is None:
        return stats

    for analysis_type in AnalysisType:
        result = await AnalysisResult.get_or_none(health_record=record, analysis_type=analysis_type)
        if result is None:
            score = _analysis_score(record, analysis_type)
            result = await AnalysisResult.create(
                user=user,
                health_record=record,
                analysis_type=analysis_type,
                risk_score=score,
                risk_level=_risk_level(score),
                summary=_summary(analysis_type, _risk_level(score)),
                model_name="dummy_rule_based",
                model_version="mvp-demo-v1",
                analyzed_at=datetime.now(config.TIMEZONE),
            )
            stats["analysis_results_created"] += 1
        else:
            stats["analysis_results_skipped"] += 1

        stats["analysis_factors_created"] += await _ensure_factors(result, record)
        stats["analysis_snapshots_created"] += await _ensure_snapshot(result, record)
    return stats


def _analysis_score(record: HealthRecord, analysis_type: AnalysisType) -> Decimal:
    if analysis_type == AnalysisType.DIABETES:
        return Decimal("0.56000") if record.fasting_glucose and record.fasting_glucose >= 100 else Decimal("0.24000")
    if analysis_type == AnalysisType.HYPERTENSION:
        return Decimal("0.52000") if record.systolic_bp and record.systolic_bp >= 130 else Decimal("0.25000")
    if analysis_type == AnalysisType.DYSLIPIDEMIA:
        return Decimal("0.38000")
    return Decimal("0.62000") if record.bmi and record.bmi >= Decimal("25") else Decimal("0.22000")


def _risk_level(score: Decimal) -> RiskLevel:
    if score >= Decimal("0.70"):
        return RiskLevel.HIGH
    if score >= Decimal("0.40"):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _summary(analysis_type: AnalysisType, risk_level: RiskLevel) -> str:
    label = {
        AnalysisType.DIABETES: "당뇨",
        AnalysisType.HYPERTENSION: "고혈압",
        AnalysisType.DYSLIPIDEMIA: "이상지질혈증",
        AnalysisType.OBESITY: "비만",
    }[analysis_type]
    level = {"LOW": "낮음", "MEDIUM": "관리 필요", "HIGH": "높음"}[risk_level.value]
    return f"{label} 위험도는 {level} 구간입니다. 생활습관 기록을 이어가면 변화 추적에 도움이 됩니다."


async def _ensure_factors(result: AnalysisResult, record: HealthRecord) -> int:
    if await AnalysisResultFactor.filter(analysis_result=result).exists():
        return 0

    factor_key, factor_name, factor_value = {
        AnalysisType.DIABETES: ("fasting_glucose", "공복혈당", record.fasting_glucose),
        AnalysisType.HYPERTENSION: ("blood_pressure", "혈압", f"{record.systolic_bp}/{record.diastolic_bp}"),
        AnalysisType.DYSLIPIDEMIA: ("ldl_cholesterol", "LDL 콜레스테롤", record.ldl_cholesterol),
        AnalysisType.OBESITY: ("bmi", "BMI", record.bmi),
    }[result.analysis_type]

    await AnalysisResultFactor.create(
        analysis_result=result,
        factor_key=factor_key,
        factor_name=factor_name,
        factor_value=str(factor_value),
        contribution_score=Decimal("0.420000"),
        direction=FactorDirection.POSITIVE,
        display_order=1,
    )
    await AnalysisResultFactor.create(
        analysis_result=result,
        factor_key="lifestyle",
        factor_name="생활습관",
        factor_value="걷기 4일, 근력운동 2일",
        contribution_score=Decimal("0.180000"),
        direction=FactorDirection.NEUTRAL,
        display_order=2,
    )
    return 2


async def _ensure_snapshot(result: AnalysisResult, record: HealthRecord) -> int:
    if await AnalysisSnapshot.filter(analysis_result=result).exists():
        return 0
    analysis_type = result.analysis_type.value
    await AnalysisSnapshot.create(
        analysis_result=result,
        input_payload={
            "analysis_type": analysis_type,
            "input_features": {
                "height_cm": _json_value(record.height_cm),
                "weight_kg": _json_value(record.weight_kg),
                "bmi": _json_value(record.bmi),
                "systolic_bp": record.systolic_bp,
                "diastolic_bp": record.diastolic_bp,
                "fasting_glucose": record.fasting_glucose,
                "hba1c": _json_value(record.hba1c),
                "total_cholesterol": record.total_cholesterol,
                "triglyceride": record.triglyceride,
                "hdl_cholesterol": record.hdl_cholesterol,
                "ldl_cholesterol": record.ldl_cholesterol,
            },
        },
        output_payload={
            "model_outputs": {analysis_type: {"risk_score": _json_value(result.risk_score)}},
            "final_outputs": {"risk_level": result.risk_level.value, "guide_message": result.summary},
        },
        shap_payload={"top_factors": ["혈압", "공복혈당", "BMI"]},
        model_payload={"model_name": "dummy_rule_based", "model_version": "mvp-demo-v1"},
    )
    return 1


async def _seed_challenges(user: User) -> dict[str, int]:
    stats = {
        "challenges_created": 0,
        "user_challenges_created": 0,
        "user_challenges_skipped": 0,
        "challenge_logs_created": 0,
        "challenge_logs_skipped": 0,
    }
    challenges = []
    for seed in DEMO_CHALLENGES:
        challenge, created = await Challenge.get_or_create(
            title=seed["title"],
            defaults={
                "description": seed["description"],
                "category": seed["category"],
                "target_metric": seed["target_metric"],
                "target_value": seed["target_value"],
                "duration_days": seed["duration_days"],
                "status": ChallengeStatus.ACTIVE,
            },
        )
        challenges.append(challenge)
        if created:
            stats["challenges_created"] += 1

    for challenge in challenges[:3]:
        user_challenge, created = await UserChallenge.get_or_create(
            user=user,
            challenge=challenge,
            defaults={
                "status": UserChallengeStatus.JOINED,
                "started_at": datetime.now(config.TIMEZONE) - timedelta(days=6),
            },
        )
        stats["user_challenges_created" if created else "user_challenges_skipped"] += 1
        for offset in range(7):
            log_date = date.today() - timedelta(days=6 - offset)
            _, log_created = await ChallengeLog.get_or_create(
                user_challenge=user_challenge,
                log_date=log_date,
                defaults={
                    "is_completed": offset not in {1, 5},
                    "memo": "시연용 챌린지 수행 기록",
                },
            )
            stats["challenge_logs_created" if log_created else "challenge_logs_skipped"] += 1
    return stats


async def _seed_diets(user: User) -> dict[str, int]:
    stats = {"diet_records_created": 0, "diet_records_skipped": 0, "diet_photo_results_created": 0}
    meals = [
        ("BREAKFAST", "오트밀, 블루베리, 삶은 달걀", 86.0),
        ("LUNCH", "현미밥, 닭가슴살 샐러드, 된장국", 82.0),
        ("DINNER", "잡곡밥, 고등어구이, 나물 반찬", 78.0),
        ("SNACK", "무가당 요거트와 견과류", 88.0),
        ("DINNER", "채소 비빔밥과 두부", 84.0),
    ]
    now = datetime.now(config.TIMEZONE)
    for index, (meal_type, description, score) in enumerate(meals):
        meal_time = now - timedelta(days=index)
        meal_start = _at_time(meal_time.date(), 0)
        meal_end = meal_start + timedelta(days=1)
        record = await DietRecord.filter(
            user=user,
            meal_time__gte=meal_start,
            meal_time__lt=meal_end,
            meal_type=meal_type,
        ).first()
        if record is not None:
            stats["diet_records_skipped"] += 1
        else:
            record = await DietRecord.create(
                user=user,
                meal_type=meal_type,
                meal_time=meal_time,
                description=description,
                image_path=f"/demo/diets/current_user_{index + 1}.jpg",
                detected_foods=[{"name": food.strip(), "confidence": 0.91} for food in description.split(",")],
                nutrition_summary={
                    "calories": 480 + index * 35,
                    "carbs_g": 58,
                    "protein_g": 31,
                    "fat_g": 14,
                    "sodium_mg": 690,
                },
                diet_score=score,
                diet_feedback="채소와 단백질 비율이 좋아 혈당 변동 관리에 도움이 되는 구성입니다.",
                analysis_method="DEMO",
                memo="로컬 MVP 시연용 식단 기록",
            )
            stats["diet_records_created"] += 1

        if not await DietPhotoResult.filter(diet_record=record).exists():
            await DietPhotoResult.create(
                diet_record=record,
                detected_foods=record.detected_foods,
                confidence_payload={"avg_confidence": 0.91},
                raw_output={"source": "local_mvp_seed"},
                is_dummy=True,
            )
            stats["diet_photo_results_created"] += 1
    return stats


async def _seed_medications(user: User) -> dict[str, int]:
    stats = {
        "medications_created": 0,
        "medications_skipped": 0,
        "medication_records_created": 0,
        "medication_records_skipped": 0,
    }
    seeds = [
        ("혈압약", "MEDICINE", "5mg", "매일 아침 1회", 8, "매일 같은 시간 복용"),
        ("오메가3", "SUPPLEMENT", "1000mg", "매일 저녁 1회", 19, "식후 복용"),
        ("종합비타민", "SUPPLEMENT", "1정", "매일 아침 1회", 9, "아침 식후 복용"),
    ]
    for name, medication_type, dosage, frequency, reminder_hour, memo in seeds:
        medication, created = await Medication.get_or_create(
            user=user,
            name=name,
            defaults={
                "medication_type": medication_type,
                "dosage": dosage,
                "frequency": frequency,
                "reminder_time": None,
                "is_active": True,
                "memo": memo,
            },
        )
        stats["medications_created" if created else "medications_skipped"] += 1
        for offset in range(3):
            scheduled_at = _at_time(date.today() - timedelta(days=2 - offset), reminder_hour)
            _, record_created = await MedicationRecord.get_or_create(
                medication=medication,
                user=user,
                scheduled_at=scheduled_at,
                defaults={
                    "taken_at": scheduled_at + timedelta(minutes=12) if offset != 1 else None,
                    "is_taken": offset != 1,
                    "status": "TAKEN" if offset != 1 else "PENDING",
                    "memo": "시연용 복약 기록",
                },
            )
            stats["medication_records_created" if record_created else "medication_records_skipped"] += 1
    return stats


async def _seed_exam(user: User) -> dict[str, int]:
    stats = {"exam_reports_created": 0, "exam_reports_skipped": 0, "exam_measurements_created": 0}
    filename = "current_user_demo_exam.pdf"
    report, created = await ExamReport.get_or_create(
        user=user,
        original_filename=filename,
        defaults={
            "file_path": f"/demo/exams/{filename}",
            "exam_date": date.today() - timedelta(days=21),
            "ocr_status": OCRStatus.CONFIRMED,
            "is_confirmed": True,
            "uploaded_at": datetime.now(config.TIMEZONE) - timedelta(days=2),
            "confirmed_at": datetime.now(config.TIMEZONE) - timedelta(days=2),
        },
    )
    stats["exam_reports_created" if created else "exam_reports_skipped"] += 1
    measurements = [
        ("height_cm", "키", "172", "cm"),
        ("weight_kg", "몸무게", "79.2", "kg"),
        ("bmi", "체질량지수", "26.8", "kg/m2"),
        ("systolic_bp", "수축기 혈압", "126", "mmHg"),
        ("diastolic_bp", "이완기 혈압", "78", "mmHg"),
        ("fasting_glucose", "공복혈당", "101", "mg/dL"),
        ("hba1c", "당화혈색소", "5.7", "%"),
        ("total_cholesterol", "총콜레스테롤", "186", "mg/dL"),
        ("triglyceride", "중성지방", "130", "mg/dL"),
        ("hdl_cholesterol", "HDL", "52", "mg/dL"),
        ("ldl_cholesterol", "LDL", "112", "mg/dL"),
        ("waist_cm", "허리둘레", "86", "cm"),
    ]
    for key, name, value, unit in measurements:
        _, measurement_created = await ExamMeasurement.get_or_create(
            exam_report=report,
            measurement_key=key,
            defaults={
                "measurement_name": name,
                "value": value,
                "unit": unit,
                "ocr_confidence": Decimal("0.9600"),
                "is_user_confirmed": True,
            },
        )
        if measurement_created:
            stats["exam_measurements_created"] += 1
    return stats


async def _seed_notifications(user: User) -> dict[str, int]:
    stats = {"notifications_created": 0, "notifications_skipped": 0}
    notifications = [
        ("ANALYSIS", "건강 분석 완료", "4종 위험도 분석 결과가 준비되었습니다.", False),
        ("CHALLENGE", "오늘의 챌린지", "식후 10분 산책 챌린지를 완료해보세요.", False),
        ("MEDICATION", "복약 확인", "오늘 아침 복약 기록을 확인해주세요.", False),
        ("DIET", "식단 기록 안내", "오늘 점심 식단을 기록하면 추적 그래프가 더 풍부해집니다.", True),
        ("OCR", "검진표 인식 완료", "최근 검진표 측정값이 확인되었습니다.", True),
        ("CHATBOT", "AI 상담 추천", "혈압 관리와 식단 조절에 대해 질문해보세요.", False),
    ]
    for notification_type, title, message, is_read in notifications:
        _, created = await Notification.get_or_create(
            user=user,
            title=title,
            defaults={
                "notification_type": notification_type,
                "message": message,
                "is_read": is_read,
                "read_at": datetime.now(config.TIMEZONE) if is_read else None,
            },
        )
        stats["notifications_created" if created else "notifications_skipped"] += 1
    return stats


def _at_time(day: date, hour: int) -> datetime:
    return datetime.combine(day, time(hour, 0), tzinfo=config.TIMEZONE)


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed current local MVP user dashboard demo data.")
    parser.add_argument("--email", required=True, help="Target user email, e.g. aszx91@gmail.com")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    stats = await seed_current_user(args.email)
    print("===== Current User Dashboard Demo Seed =====")
    print("This seed is for local MVP demos only. Do not run against production.")
    print(f"target_email: {args.email}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
