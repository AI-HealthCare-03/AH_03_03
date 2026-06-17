"""Seed FAQ records for MVP frontend demos.

FAQ rows are created idempotently by question. For production/shared databases,
run only through an explicitly approved danger target.
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")

from tortoise import Tortoise  # noqa: E402

from app.core.db.databases import TORTOISE_ORM  # noqa: E402
from app.models.faqs import FAQ  # noqa: E402


@dataclass(frozen=True)
class FAQSeed:
    category: str
    question: str
    answer: str
    display_order: int


FAQ_SEEDS = [
    FAQSeed(
        "회원/로그인",
        "비밀번호를 잊어버렸어요.",
        "로그인 화면에서 '비밀번호 재설정'을 선택하면 이메일로 재설정 링크를 받을 수 있습니다.",
        1,
    ),
    FAQSeed(
        "회원/로그인",
        "회원가입에 필요한 정보는 무엇인가요?",
        "아이디, 이메일, 비밀번호, 이름, 생년월일, 성별, 휴대폰 번호가 필요합니다.",
        2,
    ),
    FAQSeed(
        "회원/로그인",
        "로그인이 되지 않아요.",
        "이메일과 비밀번호를 다시 확인해 주세요. 문제가 계속되면 비밀번호 재설정을 시도해 보세요.",
        3,
    ),
    FAQSeed(
        "건강분석",
        "분석 결과는 의료 진단인가요?",
        "아닙니다. 분석 결과는 건강관리 참고용이며, 의료 진단이나 처방을 대체하지 않습니다.",
        4,
    ),
    FAQSeed(
        "건강분석",
        "건강 위험도 점수는 어떻게 계산되나요?",
        "간편 분석은 입력하신 건강정보를 기준으로 AI 모델이 점수를 산출하고, 정밀 분석은 건강검진 수치를 바탕으로 보다 세밀한 기준을 적용해 결과를 제공합니다.",
        5,
    ),
    FAQSeed(
        "건강분석",
        "고혈압 분석도 제공하나요?",
        "네. 간편 분석에서는 생활습관 정보를 바탕으로 고혈압 위험도를 확인할 수 있고, 정밀 분석에서는 혈압 등 건강검진 수치를 추가로 반영해 더 정확한 결과를 제공합니다.",
        6,
    ),
    FAQSeed(
        "건강분석",
        "건강검진표 OCR 결과를 수정할 수 있나요?",
        "네. 인식된 결과를 항목별로 직접 확인하고 수정할 수 있습니다.",
        7,
    ),
    FAQSeed(
        "챌린지",
        "챌린지는 어떻게 참여하나요?",
        "챌린지 목록에서 원하는 항목을 선택하고 참여하기 버튼을 누르면 됩니다.",
        8,
    ),
    FAQSeed(
        "챌린지",
        "챌린지를 중도 포기할 수 있나요?",
        "네. 내 챌린지 화면에서 포기 버튼을 눌러 중도 종료할 수 있습니다.",
        9,
    ),
    FAQSeed(
        "챌린지",
        "오늘 완료 기록은 어디서 하나요?",
        "내 챌린지 목록에서 오늘 완료 버튼을 눌러 수행 여부를 기록합니다.",
        10,
    ),
    FAQSeed(
        "식단/복약",
        "식단 분석은 실제 영양 진단인가요?",
        "아닙니다. 식단 분석은 입력된 음식 정보를 바탕으로 한 참고용 결과이며, 전문 영양 진단을 대체하지 않습니다.",
        11,
    ),
    FAQSeed(
        "식단/복약",
        "식단 이미지를 꼭 업로드해야 하나요?",
        "사진 촬영 또는 이미지 파일 선택으로 분석할 수 있습니다. 사진 없이 음식명을 직접 입력해도 분석이 가능합니다.",
        12,
    ),
    FAQSeed(
        "식단/복약",
        "복약 알림은 어떻게 설정하나요?",
        "복약/영양제 화면에서 항목을 등록하고 설정 화면에서 알림 여부를 조정할 수 있습니다.",
        13,
    ),
    FAQSeed(
        "식단/복약",
        "영양제도 복약 관리에 포함되나요?",
        "네. 약과 영양제를 같은 복약/영양제 관리 화면에서 기록할 수 있습니다.",
        14,
    ),
    FAQSeed(
        "개인정보/보안",
        "내 건강정보는 안전하게 보관되나요?",
        "개인 건강정보는 암호화되어 안전하게 보관되며, 본인만 조회할 수 있습니다.",
        15,
    ),
    FAQSeed(
        "개인정보/보안",
        "소셜 로그인은 지원하나요?",
        "현재는 이메일 기반 로그인을 지원합니다. 소셜 로그인은 추후 지원 예정입니다.",
        16,
    ),
]


async def seed_faqs() -> None:
    await Tortoise.init(config=TORTOISE_ORM)
    created_count = 0
    skipped_count = 0
    try:
        for seed in FAQ_SEEDS:
            existing = await FAQ.get_or_none(question=seed.question)
            if existing is not None:
                skipped_count += 1
                continue

            await FAQ.create(
                category=seed.category,
                question=seed.question,
                answer=seed.answer,
                display_order=seed.display_order,
                is_active=True,
            )
            created_count += 1
    finally:
        await Tortoise.close_connections()

    print("===== MVP FAQ Seed =====")
    print("FAQ rows are created idempotently by question.")
    print(f"created_count: {created_count}")
    print(f"skipped_count: {skipped_count}")


if __name__ == "__main__":
    asyncio.run(seed_faqs())
