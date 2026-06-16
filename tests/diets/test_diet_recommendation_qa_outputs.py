from __future__ import annotations

import json

import pytest

from scripts.qa import generate_diet_recommendation_qa_outputs as qa


@pytest.mark.asyncio
async def test_diet_recommendation_qa_generator_builds_70_cases() -> None:
    cases = qa.build_qa_cases()
    outputs = await qa.generate_outputs(cases)

    assert len(cases) == 70
    assert len(outputs) == 70
    assert qa.summarize_outputs(outputs)["groups"] == {"A": 42, "B": 20, "C": 8}


def test_diet_recommendation_qa_case_coverage() -> None:
    cases = qa.build_qa_cases()

    disease_groups = {group for case in cases for group in case.disease_groups}
    diet_issues = {case.diet_issue for case in cases}
    composite_titles = {case.title for case in cases if case.group == "B"}

    assert disease_groups == set(qa.MANAGEMENT_GROUPS)
    assert diet_issues == set(qa.DIET_ISSUES)
    assert "HTN + DM" in composite_titles
    assert "CKD + HTN + DM" in composite_titles
    assert "HTN + DM + OBE + DL" in composite_titles


@pytest.mark.asyncio
async def test_diet_recommendation_qa_outputs_include_required_response_fields() -> None:
    outputs = await qa.generate_outputs(qa.build_qa_cases()[:3])

    required = {
        "diet_record_id",
        "nutrition_findings",
        "disease_context",
        "recommended_foods",
        "caution_foods",
        "recommended_challenges",
        "safety_notice",
        "rag_comment",
    }
    for output in outputs:
        assert required <= set(output["response"])
        assert output["checks"]["passed"] is True


def test_diet_recommendation_qa_forbidden_checkers_detect_terms() -> None:
    response = {
        "diet_record_id": 1,
        "nutrition_findings": [{"message": "반드시 치료하세요"}],
        "disease_context": [],
        "recommended_foods": [],
        "caution_foods": [],
        "recommended_challenges": [],
        "safety_notice": "생활관리 참고 정보",
        "rag_comment": {
            "summary": "candidate_unreviewed chunk_key score text-embedding",
            "disease_comments": [],
            "evidence_sources": [],
            "safety_notice": "생활관리 참고 정보",
        },
    }

    checks = qa.run_response_checks(response)

    assert checks["passed"] is False
    assert "반드시" in checks["forbidden_phrases_found"]
    assert "치료" in checks["forbidden_phrases_found"]
    assert "candidate_unreviewed" in checks["internal_terms_found"]
    assert "chunk_key" in checks["internal_terms_found"]
    assert "score" in checks["internal_terms_found"]
    assert "text-embedding" in checks["internal_terms_found"]


@pytest.mark.asyncio
async def test_diet_recommendation_qa_rag_edge_cases_are_present() -> None:
    outputs = await qa.generate_outputs([case for case in qa.build_qa_cases() if case.group == "C"])
    by_id = {output["case"]["case_id"]: output for output in outputs}

    assert by_id["C02"]["rag_trace"]["latest_metadata"]["fallback_reason"] == "keyword_result_sufficient"
    assert by_id["C02"]["rag_trace"]["vector_calls"] == 0
    assert by_id["C03"]["rag_trace"]["latest_metadata"]["fallback_used"] is True
    assert by_id["C03"]["rag_trace"]["latest_metadata"]["vector_returned_count"] >= 1
    assert by_id["C03"]["rag_trace"]["vector_calls"] >= 1
    assert by_id["C04"]["rag_trace"]["latest_metadata"]["fallback_reason"] == "vector_disabled"
    assert by_id["C05"]["rag_trace"]["latest_metadata"]["fallback_reason"] == "vector_gate_disabled"
    assert by_id["C06"]["rag_trace"]["event_count"] == 0
    assert by_id["C08"]["rag_trace"]["langfuse_available"] is False


@pytest.mark.asyncio
async def test_diet_recommendation_qa_outputs_do_not_expose_internal_values() -> None:
    outputs = await qa.generate_outputs(qa.build_qa_cases())

    assert all(output["checks"]["passed"] for output in outputs)
    serialized_responses = json.dumps([output["response"] for output in outputs], ensure_ascii=False)
    for term in qa.INTERNAL_TERMS:
        assert term not in serialized_responses


@pytest.mark.asyncio
async def test_diet_recommendation_qa_anem_caution_foods_are_not_split_characters() -> None:
    cases = [case for case in qa.build_qa_cases() if "ANEM" in case.disease_groups]
    outputs = await qa.generate_outputs(cases)

    for output in outputs:
        caution_foods = output["response"]["caution_foods"]
        serialized = ", ".join(caution_foods)
        assert "식, 사" not in serialized
        assert "직, 후" not in serialized
        assert all(len(item.strip()) > 1 for item in caution_foods)


@pytest.mark.asyncio
async def test_diet_recommendation_qa_ckd_food_recommendations_are_conservative() -> None:
    cases = [case for case in qa.build_qa_cases() if "CKD" in case.disease_groups]
    outputs = await qa.generate_outputs(cases)

    blocked_foods = {"잡곡밥", "해조류 반찬", "고단백 보충제"}
    blocked_messages = ("채소 반찬이나 잡곡밥처럼 쉽게", "단백질 반찬이나 채소를 함께")
    for output in outputs:
        recommended_foods = set(output["response"]["recommended_foods"])
        serialized_response = json.dumps(output["response"], ensure_ascii=False)
        assert recommended_foods.isdisjoint(blocked_foods)
        assert any(item in recommended_foods for item in {"식사일지", "조리법 기록", "상담 전 식단 기록"})
        for blocked_message in blocked_messages:
            assert blocked_message not in serialized_response


@pytest.mark.asyncio
async def test_diet_recommendation_qa_ckd_challenges_are_conservative() -> None:
    cases = [case for case in qa.build_qa_cases() if "CKD" in case.disease_groups]
    outputs = await qa.generate_outputs(cases)

    blocked_titles = {"식이섬유 먹어유 챌린지", "건강식탁 챌린지"}
    for output in outputs:
        challenge_titles = {item["title"] for item in output["response"]["recommended_challenges"]}
        assert challenge_titles.isdisjoint(blocked_titles)


@pytest.mark.asyncio
async def test_diet_recommendation_qa_htn_and_sodium_cases_do_not_recommend_seaweed() -> None:
    cases = [case for case in qa.build_qa_cases() if "HTN" in case.disease_groups or case.diet_issue == "나트륨 주의"]
    outputs = await qa.generate_outputs(cases)

    for output in outputs:
        assert "해조류 반찬" not in output["response"]["recommended_foods"]


@pytest.mark.asyncio
async def test_diet_recommendation_qa_challenge_reasons_are_deduplicated() -> None:
    outputs = await qa.generate_outputs(qa.build_qa_cases())

    for output in outputs:
        challenges = output["response"]["recommended_challenges"]
        ids = [item["challenge_id"] for item in challenges]
        titles = [item["title"] for item in challenges]
        reasons = [item["reason"] for item in challenges]
        assert len(ids) == len(set(ids))
        assert len(titles) == len(set(titles))
        assert len(reasons) == len(set(reasons))
        assert len(challenges) <= 3


@pytest.mark.asyncio
async def test_diet_recommendation_qa_composite_cases_have_diverse_challenge_focus() -> None:
    cases = [case for case in qa.build_qa_cases() if case.title in {"HTN + DM", "CKD + HTN + DM", "OBE + FL + DL"}]
    outputs = await qa.generate_outputs(cases)
    by_title = {output["case"]["title"]: output["response"]["recommended_challenges"] for output in outputs}

    htn_dm_titles = {item["title"] for item in by_title["HTN + DM"]}
    assert "염분 빼볼까염 챌린지" in htn_dm_titles
    assert any("탄수화물" in title or "채고밥" in title for title in htn_dm_titles)
    assert "식사일지 작성 챌린지" in htn_dm_titles

    ckd_combo_titles = {item["title"] for item in by_title["CKD + HTN + DM"]}
    assert "염분 빼볼까염 챌린지" in ckd_combo_titles
    assert any("탄수화물" in title or "채고밥" in title for title in ckd_combo_titles)
    assert "식사일지 작성 챌린지" in ckd_combo_titles

    metabolic_titles = {item["title"] for item in by_title["OBE + FL + DL"]}
    assert any("야식" in title or "폭식" in title or "2020" in title for title in metabolic_titles)
    assert any("기름" in title for title in metabolic_titles)
    assert any("식이섬유" in title or "건강식탁" in title for title in metabolic_titles)


@pytest.mark.asyncio
async def test_diet_recommendation_qa_rag_disabled_case_has_no_evidence_sources() -> None:
    case = next(case for case in qa.build_qa_cases() if case.case_id == "C06")
    output = (await qa.generate_outputs([case]))[0]

    assert "trace" not in case.expected_focus.lower()
    assert output["response"]["rag_comment"]["enabled"] is False
    assert output["response"]["rag_comment"]["evidence_sources"] == []


@pytest.mark.asyncio
async def test_diet_recommendation_qa_write_outputs(tmp_path) -> None:
    outputs = await qa.generate_outputs(qa.build_qa_cases()[:2])
    output_json = tmp_path / "diet_qa.json"
    output_md = tmp_path / "diet_qa.md"

    qa.write_outputs(outputs, output_json, output_md)

    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    markdown = output_md.read_text(encoding="utf-8")
    assert len(loaded) == 2
    assert "Case ID" in markdown
    assert "## A01" in markdown
    assert "nutrition_findings message" in markdown
    assert "recommended_challenges:" in markdown
    assert " | " in markdown


@pytest.mark.asyncio
async def test_diet_recommendation_qa_does_not_call_external_llm_or_db(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("external call or DB access must not be used by QA generator")

    monkeypatch.setattr("ai_runtime.llm.diet_recommendation_rewriter.call_llm_json", fail_if_called)
    monkeypatch.setattr(qa.service.diet_service, "get_diet_record", fail_if_called)
    monkeypatch.setattr(qa.service.analysis_repository, "list_analysis_results_by_user", fail_if_called)

    outputs = await qa.generate_outputs(qa.build_qa_cases()[:5])

    assert len(outputs) == 5
    assert all(output["checks"]["passed"] for output in outputs)
