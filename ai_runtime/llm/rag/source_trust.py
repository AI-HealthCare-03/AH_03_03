from __future__ import annotations

from typing import Any

OFFICIAL_GUIDELINE_TYPES = {"clinical_guideline", "nutrition_guideline", "official_guideline"}
PUBLIC_HEALTH_AGENCY_TYPES = {"official_society", "public_data_api", "public_health_agency"}
INTERNAL_POLICY_TYPES = {"safety_policy", "internal_policy"}
SERVICE_FAQ_TYPES = {"service_faq"}
CHALLENGE_POLICY_TYPES = {"challenge_policy"}

TRUST_LEVEL_ORDER = {
    "official_guideline": 5,
    "public_health_agency": 4,
    "internal_policy": 3,
    "service_faq": 2,
    "challenge_policy": 2,
    "unknown": 1,
}


def source_trust_level_for_type(source_type: str | None) -> str:
    normalized = str(source_type or "").strip().lower()
    if normalized in OFFICIAL_GUIDELINE_TYPES:
        return "official_guideline"
    if normalized in PUBLIC_HEALTH_AGENCY_TYPES:
        return "public_health_agency"
    if normalized in INTERNAL_POLICY_TYPES:
        return "internal_policy"
    if normalized in SERVICE_FAQ_TYPES:
        return "service_faq"
    if normalized in CHALLENGE_POLICY_TYPES:
        return "challenge_policy"
    return "unknown"


def source_trust_level_for_metadata(metadata: dict[str, Any]) -> str:
    return source_trust_level_for_type(metadata.get("source_type"))


def lowest_source_trust_level(levels: list[str]) -> str:
    if not levels:
        return "unknown"
    return min(levels, key=lambda level: TRUST_LEVEL_ORDER.get(level, 0))


def is_low_trust_level(level: str | None) -> bool:
    return TRUST_LEVEL_ORDER.get(str(level or "unknown"), 0) <= TRUST_LEVEL_ORDER["unknown"]
