from __future__ import annotations

import json

import pytest

from scripts.qa import generate_diet_recommendation_openai_smoke_outputs as smoke


def _fake_rewrite(*, rag_comment, recommendation_payload, use_real_llm):
    assert use_real_llm is True
    assert recommendation_payload["recommended_challenges"]
    return {
        **rag_comment,
        "rewrite_used": True,
        "fallback_reason": None,
        "summary": "실제 섭취량이 확정되지 않아 참고용입니다. 생활관리 관점에서 식사 기록을 이어가 보세요.",
    }


def test_openai_smoke_cases_are_limited_and_representative() -> None:
    cases = smoke.smoke_cases()

    assert len(cases) == 10
    titles = {case.title for case in cases}
    assert "HTN 관리 + 나트륨 주의" in titles
    assert "DM 관리 + 당류/탄수화물 주의" in titles
    assert "CKD + HTN + DM" in titles
    assert "hybrid keyword 0건 + vector fallback 경로" in titles


@pytest.mark.asyncio
async def test_openai_smoke_requires_explicit_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smoke, "has_openai_config", lambda config: True)

    with pytest.raises(smoke.OpenAISmokeConfigurationError):
        await smoke.generate_smoke_outputs(confirm_openai_call=False, rewrite_func=_fake_rewrite)


@pytest.mark.asyncio
async def test_openai_smoke_generates_outputs_with_fake_rewrite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smoke, "has_openai_config", lambda config: True)
    monkeypatch.setattr(smoke, "get_openai_model", lambda: "gpt-test")

    outputs = await smoke.generate_smoke_outputs(confirm_openai_call=True, rewrite_func=_fake_rewrite)

    assert len(outputs) == 10
    assert all(output["openai"]["called"] is True for output in outputs)
    assert all(output["openai"]["rewrite_used"] is True for output in outputs)
    assert all(output["checks"]["passed"] is True for output in outputs)
    serialized = json.dumps(outputs, ensure_ascii=False)
    for term in smoke.qa.INTERNAL_TERMS:
        assert term not in serialized


@pytest.mark.asyncio
async def test_openai_smoke_write_outputs_include_original_and_rewrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(smoke, "has_openai_config", lambda config: True)
    monkeypatch.setattr(smoke, "get_openai_model", lambda: "gpt-test")
    outputs = await smoke.generate_smoke_outputs(confirm_openai_call=True, rewrite_func=_fake_rewrite)
    output_json = tmp_path / "smoke.json"
    output_md = tmp_path / "smoke.md"

    smoke.write_outputs(outputs, output_json, output_md)

    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    markdown = output_md.read_text(encoding="utf-8")
    assert len(loaded) == 10
    assert "rule/formatter 기반 원본 응답" in markdown
    assert "OpenAI rewrite 응답" in markdown
    assert "recommended_challenges" in markdown
    assert "API key" in markdown
    assert "candidate_unreviewed" not in markdown
    assert "chunk_key" not in markdown


def test_openai_smoke_checker_detects_forbidden_terms() -> None:
    response = {
        "diet_record_id": 1,
        "nutrition_findings": [{"message": "반드시 치료"}],
        "disease_context": [],
        "recommended_foods": [],
        "caution_foods": [],
        "recommended_challenges": [{"challenge_id": 1, "title": "테스트", "reason": "참고"}],
        "safety_notice": "생활관리 참고 정보",
        "rag_comment": {"summary": "chunk_key", "disease_comments": [], "evidence_sources": []},
    }

    checks = smoke.qa.run_response_checks(response)

    assert checks["passed"] is False
    assert "반드시" in checks["forbidden_phrases_found"]
    assert "chunk_key" in checks["internal_terms_found"]
