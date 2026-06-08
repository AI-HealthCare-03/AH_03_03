from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        -- 기존 BLOOD_PRESSURE / BLOOD_GLUCOSE 데이터를 MONITORING 으로 일괄 변경
        UPDATE challenges
           SET category = 'MONITORING'
         WHERE category IN ('BLOOD_PRESSURE', 'BLOOD_GLUCOSE');

        -- category 컬럼 주석 업데이트
        COMMENT ON COLUMN "challenges"."category" IS
            'EXERCISE: EXERCISE\nDIET: DIET\nSLEEP: SLEEP\nMEDICATION: MEDICATION\nWATER: WATER\nMONITORING: MONITORING\nWEIGHT: WEIGHT\nHABIT: HABIT';
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        -- MONITORING → BLOOD_GLUCOSE 로 롤백 (원복 시 BLOOD_PRESSURE 구분 불가)
        UPDATE challenges
           SET category = 'BLOOD_GLUCOSE'
         WHERE category = 'MONITORING';

        COMMENT ON COLUMN "challenges"."category" IS
            'EXERCISE: EXERCISE\nDIET: DIET\nSLEEP: SLEEP\nMEDICATION: MEDICATION\nWATER: WATER\nBLOOD_PRESSURE: BLOOD_PRESSURE\nBLOOD_GLUCOSE: BLOOD_GLUCOSE\nWEIGHT: WEIGHT\nHABIT: HABIT';
    """
