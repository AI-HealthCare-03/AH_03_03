"""Seed FAQ records for local MVP frontend demos.

This script is for local MVP testing only. It is not intended for production or
shared databases. FAQ rows are created idempotently by question.
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
        "로그인 화면의 비밀번호 재설정 기능에서 임시 토큰을 발급받을 수 있습니다.",
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
        "이메일과 비밀번호를 다시 확인하고, 로컬 테스트 DB에 데모 계정이 생성되어 있는지 확인해 주세요.",
        3,
    ),
    FAQSeed(
        "건강분석",
        "분석 결과는 의료 진단인가요?",
        "아닙니다. MVP 분석 결과는 건강관리 참고용이며 진단이나 처방이 아닙니다.",
        4,
    ),
    FAQSeed(
        "건강분석",
        "건강 위험도 점수는 어떻게 계산되나요?",
        "현재 MVP는 혈당, BMI, 지질, 혈압 기준의 더미 룰로 점수를 계산합니다.",
        5,
    ),
    FAQSeed(
        "건강분석", "고혈압 분석도 제공하나요?", "네. 수축기/이완기 혈압 입력값을 기준으로 더미 위험도를 표시합니다.", 6
    ),
    FAQSeed(
        "건강분석",
        "건강검진표 OCR 결과를 수정할 수 있나요?",
        "OCR 결과는 측정값 단위로 확인하고 수정하는 흐름을 전제로 설계되어 있습니다.",
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
        "식단/복약", "식단 분석은 실제 영양 진단인가요?", "아닙니다. 현재 식단 분석은 MVP 시연용 더미 결과입니다.", 11
    ),
    FAQSeed(
        "식단/복약",
        "식단 이미지를 꼭 업로드해야 하나요?",
        "현재는 이미지 업로드 없이 메모 기반 더미 분석을 실행할 수 있습니다.",
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
        "로컬 MVP에서는 개발 DB에 저장되며, 운영 환경에서는 접근 권한과 보안 설정을 강화해야 합니다.",
        15,
    ),
    FAQSeed(
        "개인정보/보안",
        "Firebase 인증은 꼭 필요한가요?",
        "아닙니다. 현재 MVP 기본 인증은 FastAPI JWT이며 Firebase는 optional provider로 유지됩니다.",
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
    print("This seed is for local MVP demos only.")
    print(f"created_count: {created_count}")
    print(f"skipped_count: {skipped_count}")


if __name__ == "__main__":
    asyncio.run(seed_faqs())
