from __future__ import annotations

import json
from typing import Any

from ai_runtime.llm.llm_client import call_llm_json
from ai_runtime.llm.prompt_templates import (
    DIET_RECOMMENDATION_REWRITE_PROMPT,
    DIET_RECOMMENDATION_REWRITE_PROMPT_VERSION,
)

DIET_RECOMMENDATION_REWRITE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "disease_comments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "disease_code": {"type": "string"},
                    "label": {"type": "string"},
                    "comment": {"type": "string"},
                    "basis": {"type": "string"},
                },
                "required": ["disease_code", "label", "comment", "basis"],
            },
        },
    },
    "required": ["summary", "disease_comments"],
}

FORBIDDEN_DIET_REWRITE_PHRASES = (
    "나트륨 과다입니다",
    "단백질이 부족합니다",
    "당뇨 식단으로 부적절합니다",
    "고혈압 식단입니다",
    "이 음식을 먹으면 안 됩니다",
    "단백질 제한하세요",
    "칼륨 제한하세요",
    "인 제한하세요",
    "치료하세요",
    "처방받으세요",
)


def rewrite_diet_rag_comment(
    *,
    rag_comment: dict[str, Any] | None,
    recommendation_payload: dict[str, Any],
    use_real_llm: bool,
) -> dict[str, Any] | None:
    if rag_comment is None:
        return None
    if not use_real_llm:
        return _with_rewrite_metadata(rag_comment, rewrite_used=False, fallback_reason="rewrite_disabled")

    try:
        prompt = _build_rewrite_prompt(rag_comment=rag_comment, recommendation_payload=recommendation_payload)
        raw_response = call_llm_json(
            prompt,
            schema=DIET_RECOMMENDATION_REWRITE_SCHEMA,
            schema_name="diet_recommendation_rewrite",
            metadata={
                "prompt_version": DIET_RECOMMENDATION_REWRITE_PROMPT_VERSION,
                "source": "diet_recommendation_rewrite",
                "use_real_llm": True,
            },
        )
        rewritten = _parse_rewrite_response(raw_response, original=rag_comment)
        if not _is_safe_rewrite(rewritten):
            return _with_rewrite_metadata(rag_comment, rewrite_used=False, fallback_reason="safety_failed")
        return _with_rewrite_metadata(rewritten, rewrite_used=True, fallback_reason=None)
    except Exception:
        return _with_rewrite_metadata(rag_comment, rewrite_used=False, fallback_reason="llm_rewrite_failed")


def _build_rewrite_prompt(*, rag_comment: dict[str, Any], recommendation_payload: dict[str, Any]) -> str:
    payload = {
        "nutrition_findings": recommendation_payload.get("nutrition_findings", []),
        "disease_context": recommendation_payload.get("disease_context", []),
        "recommended_foods": recommendation_payload.get("recommended_foods", []),
        "caution_foods": recommendation_payload.get("caution_foods", []),
        "recommended_challenges": recommendation_payload.get("recommended_challenges", []),
        "rag_comment": rag_comment,
        "safety_notice": recommendation_payload.get("safety_notice") or rag_comment.get("safety_notice"),
    }
    return DIET_RECOMMENDATION_REWRITE_PROMPT.format(payload=json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _parse_rewrite_response(raw_response: str, *, original: dict[str, Any]) -> dict[str, Any]:
    parsed = json.loads(raw_response)
    summary = str(parsed.get("summary") or "").strip()
    if not summary:
        raise ValueError("Diet recommendation rewrite response must include summary.")

    allowed_codes = {
        str(comment.get("disease_code") or "")
        for comment in _list_of_dicts(original.get("disease_comments"))
        if comment.get("disease_code")
    }
    comments = []
    for comment in _list_of_dicts(parsed.get("disease_comments")):
        disease_code = str(comment.get("disease_code") or "").strip()
        if disease_code not in allowed_codes:
            continue
        comments.append(
            {
                "disease_code": disease_code,
                "label": str(comment.get("label") or "").strip(),
                "comment": str(comment.get("comment") or "").strip(),
                "basis": str(comment.get("basis") or "").strip(),
            }
        )
    if any(not item["label"] or not item["comment"] or not item["basis"] for item in comments):
        raise ValueError("Diet recommendation rewrite disease comments must be complete.")

    return {
        **original,
        "summary": summary,
        "disease_comments": comments,
        "safety_notice": str(original.get("safety_notice") or ""),
    }


def _is_safe_rewrite(rag_comment: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(rag_comment.get("summary") or ""),
            str(rag_comment.get("safety_notice") or ""),
            *[str(comment.get("comment") or "") for comment in _list_of_dicts(rag_comment.get("disease_comments"))],
        ]
    )
    if any(phrase in text for phrase in FORBIDDEN_DIET_REWRITE_PHRASES):
        return False
    if "실제 섭취량" not in text or "참고" not in text or "진단" not in text:
        return False
    return True


def _with_rewrite_metadata(
    rag_comment: dict[str, Any],
    *,
    rewrite_used: bool,
    fallback_reason: str | None,
) -> dict[str, Any]:
    return {
        **rag_comment,
        "rewrite_used": rewrite_used,
        "fallback_reason": fallback_reason,
    }


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
