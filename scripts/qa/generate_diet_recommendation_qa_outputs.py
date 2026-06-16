from __future__ import annotations

# ruff: noqa: E402,I001

import argparse
import asyncio
import json
import sys
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_runtime.llm.rag.retriever import RagRetrievalResult, RetrievedDocument
from app.models.analysis import AnalysisType, RiskLevel
from app.services import diet_recommendations as service


FORBIDDEN_PHRASES = (
    "반드시",
    "절대",
    "치료",
    "처방",
    "진단",
    "병이 있다",
    "먹으면 안 됩니다",
    "위험합니다",
)
INTERNAL_TERMS = (
    "candidate_unreviewed",
    "missing_source",
    "후보 지식",
    "chunk_key",
    "score",
    "embedding",
    "pgvector",
    "text-embedding",
    "vector retriever",
    "similarity",
)

MANAGEMENT_GROUPS = ("HTN", "DM", "DL", "OBE", "ANEM", "FL", "CKD")
DIET_ISSUES = (
    "식이섬유 보완",
    "나트륨 주의",
    "당류/탄수화물 주의",
    "포화지방/튀김/가공식품 주의",
    "열량/야식/과식",
    "이슈 없음 또는 균형 보완",
)

GROUP_ANALYSIS_TYPES: dict[str, AnalysisType] = {
    "HTN": AnalysisType.HYPERTENSION,
    "DM": AnalysisType.DIABETES,
    "DL": AnalysisType.DYSLIPIDEMIA,
    "OBE": AnalysisType.OBESITY,
    "ANEM": AnalysisType.ANEMIA,
    "FL": AnalysisType.FATTY_LIVER,
    "CKD": AnalysisType.CHRONIC_KIDNEY_DISEASE,
}

ACTIVE_CHALLENGE_TITLES = (
    "염분 빼볼까염 챌린지",
    "식사일지 작성 챌린지",
    "추가설탕 안녕이당 챌린지",
    "탄수화물 체인지 챌린지",
    "밥최고NO 채고밥YES 챌린지",
    "기름기 쫙빼기 챌린지",
    "건강식탁 챌린지",
    "Goodbye 야식 챌린지",
    "Goodbye 폭식 챌린지",
    "2020 식사 챌린지",
    "철분 반찬 추가 챌린지",
    "식이섬유 먹어유 챌린지",
    "비타민C 함께 먹기 챌린지",
    "철분 흡수 방해 식품 줄이기 챌린지",
    "30일, 금주 챌린지",
    "30일, 하루 두 잔 절주 챌린지",
    "30일 폭음 피하기 챌린지",
    "일정한 삼시세끼 챌린지",
)


@dataclass(frozen=True)
class DietQaCase:
    case_id: str
    group: str
    title: str
    user_context: str
    disease_groups: list[str]
    diet_issue: str
    meal_example: str
    rag_strategy: str
    expected_focus: str
    nutrition: dict[str, Any]
    food_name: str
    analysis_types: list[AnalysisType]
    description: str | None = None
    meal_type: str = "LUNCH"
    force_keyword_empty: bool = False
    fake_vector_documents: list[RetrievedDocument] = field(default_factory=list)
    rag_enabled: bool = True
    rag_embedding_enabled: bool = False
    rag_embedding_provider: str = "disabled"
    langfuse_available: bool = True


class FakeVectorRetriever:
    def __init__(self, documents: list[RetrievedDocument]) -> None:
        self.documents = documents
        self.calls: list[dict[str, Any]] = []

    async def retrieve(self, **kwargs: Any) -> RagRetrievalResult:
        self.calls.append(kwargs)
        return RagRetrievalResult(documents=self.documents, strategy="vector")


def build_qa_cases() -> list[DietQaCase]:
    cases: list[DietQaCase] = []

    for group in MANAGEMENT_GROUPS:
        for issue in DIET_ISSUES:
            case_number = len(cases) + 1
            cases.append(
                _case(
                    case_id=f"A{case_number:02d}",
                    group="A",
                    disease_groups=[group],
                    diet_issue=issue,
                    rag_strategy="keyword_only",
                    title=f"{group} 관리 + {issue}",
                    expected_focus=f"{group} 관리 맥락에서 {issue} 문구가 자연스러운지 확인",
                )
            )

    composite_groups = [
        ("HTN + DM", ["HTN", "DM"], "나트륨 주의"),
        ("HTN + DL", ["HTN", "DL"], "포화지방/튀김/가공식품 주의"),
        ("HTN + OBE", ["HTN", "OBE"], "열량/야식/과식"),
        ("DM + OBE", ["DM", "OBE"], "당류/탄수화물 주의"),
        ("DM + DL", ["DM", "DL"], "포화지방/튀김/가공식품 주의"),
        ("DM + CKD", ["DM", "CKD"], "당류/탄수화물 주의"),
        ("DL + OBE", ["DL", "OBE"], "포화지방/튀김/가공식품 주의"),
        ("DL + FL", ["DL", "FL"], "포화지방/튀김/가공식품 주의"),
        ("OBE + FL", ["OBE", "FL"], "열량/야식/과식"),
        ("OBE + ABDOMINAL_OBESITY 맥락", ["OBE"], "열량/야식/과식"),
        ("FL + LIVER_FUNCTION 맥락", ["FL"], "당류/탄수화물 주의"),
        ("CKD + KIDNEY_FUNCTION 맥락", ["CKD"], "이슈 없음 또는 균형 보완"),
        ("CKD + HTN", ["CKD", "HTN"], "나트륨 주의"),
        ("CKD + DM", ["CKD", "DM"], "당류/탄수화물 주의"),
        ("ANEM + OBE", ["ANEM", "OBE"], "식이섬유 보완"),
        ("ANEM + DM", ["ANEM", "DM"], "당류/탄수화물 주의"),
        ("HTN + DM + DL", ["HTN", "DM", "DL"], "나트륨 주의"),
        ("OBE + FL + DL", ["OBE", "FL", "DL"], "열량/야식/과식"),
        ("CKD + HTN + DM", ["CKD", "HTN", "DM"], "나트륨 주의"),
        ("HTN + DM + OBE + DL", ["HTN", "DM", "OBE", "DL"], "당류/탄수화물 주의"),
    ]
    for index, (title, groups, issue) in enumerate(composite_groups, start=1):
        cases.append(
            _case(
                case_id=f"B{index:02d}",
                group="B",
                disease_groups=groups,
                diet_issue=issue,
                rag_strategy="keyword_only",
                title=title,
                expected_focus="복합 관리 맥락에서 한쪽 질환에만 치우치지 않는지 확인",
                analysis_types=_analysis_types_for_groups(groups, title),
            )
        )

    edge_cases = [
        _case(
            case_id="C01",
            group="C",
            disease_groups=["HTN"],
            diet_issue="나트륨 주의",
            rag_strategy="keyword_only",
            title="keyword_only 기본 경로",
            expected_focus="keyword RAG evidence가 있을 때 기본 응답 확인",
        ),
        _case(
            case_id="C02",
            group="C",
            disease_groups=["HTN"],
            diet_issue="나트륨 주의",
            rag_strategy="keyword_first_vector_fallback",
            title="hybrid keyword 충분 경로",
            expected_focus="keyword 결과가 충분하면 vector fallback이 호출되지 않는지 확인",
            rag_embedding_enabled=True,
            rag_embedding_provider="openai",
        ),
        _case(
            case_id="C03",
            group="C",
            disease_groups=["HTN"],
            diet_issue="나트륨 주의",
            rag_strategy="keyword_first_vector_fallback",
            title="hybrid keyword 0건 + vector fallback 경로",
            expected_focus="keyword 0건일 때 vector 결과가 evidence로 변환되는지 확인",
            force_keyword_empty=True,
            fake_vector_documents=[_fake_vector_document("HTN", "고혈압 식생활 후보 지식")],
            rag_embedding_enabled=True,
            rag_embedding_provider="openai",
        ),
        _case(
            case_id="C04",
            group="C",
            disease_groups=["DM"],
            diet_issue="당류/탄수화물 주의",
            rag_strategy="vector_disabled",
            title="vector disabled 경로",
            expected_focus="vector disabled에서도 keyword 응답이 유지되는지 확인",
        ),
        _case(
            case_id="C05",
            group="C",
            disease_groups=["DL"],
            diet_issue="포화지방/튀김/가공식품 주의",
            rag_strategy="keyword_first_vector_fallback",
            title="벡터 설정 비활성 경로",
            expected_focus="벡터 설정이 꺼져 있으면 fallback 없이 안전한 응답인지 확인",
            force_keyword_empty=True,
            fake_vector_documents=[_fake_vector_document("DL", "이상지질혈증 식생활 후보 지식")],
            rag_embedding_enabled=False,
        ),
        _case(
            case_id="C06",
            group="C",
            disease_groups=["OBE"],
            diet_issue="열량/야식/과식",
            rag_strategy="keyword_only",
            title="RAG disabled 경로",
            expected_focus="RAG가 비활성화되어도 기본 식단 추천 응답이 안전하게 유지되는지 확인",
            rag_enabled=False,
        ),
        _case(
            case_id="C07",
            group="C",
            disease_groups=["HTN"],
            diet_issue="나트륨 주의",
            rag_strategy="keyword_first_vector_fallback",
            title="evidence source sanitizing 경로",
            expected_focus="내부 검토 상태와 내부 표현이 사용자 응답에서 제거되는지 확인",
            force_keyword_empty=True,
            fake_vector_documents=[_fake_vector_document("HTN", "고혈압/혈압/저염/나트륨 후보 지식")],
            rag_embedding_enabled=True,
            rag_embedding_provider="openai",
        ),
        _case(
            case_id="C08",
            group="C",
            disease_groups=["CKD"],
            diet_issue="이슈 없음 또는 균형 보완",
            rag_strategy="keyword_first_vector_fallback",
            title="Langfuse unavailable but API success",
            expected_focus="Langfuse 이벤트 전송이 비활성 상태여도 응답이 성공하는지 확인",
            langfuse_available=False,
            rag_embedding_enabled=True,
            rag_embedding_provider="openai",
        ),
    ]
    cases.extend(edge_cases)
    return cases


def _case(
    *,
    case_id: str,
    group: str,
    disease_groups: list[str],
    diet_issue: str,
    rag_strategy: str,
    title: str,
    expected_focus: str,
    analysis_types: list[AnalysisType] | None = None,
    force_keyword_empty: bool = False,
    fake_vector_documents: list[RetrievedDocument] | None = None,
    rag_enabled: bool = True,
    rag_embedding_enabled: bool = False,
    rag_embedding_provider: str = "disabled",
    langfuse_available: bool = True,
) -> DietQaCase:
    nutrition, food_name, meal_example, description, meal_type = _meal_for_issue(diet_issue)
    return DietQaCase(
        case_id=case_id,
        group=group,
        title=title,
        user_context=_user_context(disease_groups),
        disease_groups=disease_groups,
        diet_issue=diet_issue,
        meal_example=meal_example,
        rag_strategy=rag_strategy,
        expected_focus=expected_focus,
        nutrition=nutrition,
        food_name=food_name,
        analysis_types=analysis_types or _analysis_types_for_groups(disease_groups, title),
        description=description,
        meal_type=meal_type,
        force_keyword_empty=force_keyword_empty,
        fake_vector_documents=fake_vector_documents or [],
        rag_enabled=rag_enabled,
        rag_embedding_enabled=rag_embedding_enabled,
        rag_embedding_provider=rag_embedding_provider,
        langfuse_available=langfuse_available,
    )


def _analysis_types_for_groups(groups: list[str], title: str = "") -> list[AnalysisType]:
    analysis_types = [GROUP_ANALYSIS_TYPES[group] for group in groups if group in GROUP_ANALYSIS_TYPES]
    if "ABDOMINAL_OBESITY" in title:
        analysis_types.append(AnalysisType.ABDOMINAL_OBESITY)
    if "LIVER_FUNCTION" in title:
        analysis_types.append(AnalysisType.LIVER_FUNCTION)
    if "KIDNEY_FUNCTION" in title:
        analysis_types.append(AnalysisType.KIDNEY_FUNCTION)
    return _dedupe_analysis_types(analysis_types)


def _meal_for_issue(diet_issue: str) -> tuple[dict[str, Any], str, str, str | None, str]:
    if diet_issue == "식이섬유 보완":
        return ({"fiber_g": 1.0, "protein_g": 12, "sodium_mg": 260}, "흰밥과 계란", "흰밥과 계란", None, "LUNCH")
    if diet_issue == "나트륨 주의":
        return ({"sodium_mg": 920, "calories_kcal": 430, "fiber_g": 2.5}, "라면", "라면과 김치", None, "DINNER")
    if diet_issue == "당류/탄수화물 주의":
        return (
            {"carbohydrate_g": 82, "sugar_g": 20, "calories_kcal": 520, "fiber_g": 2.0},
            "달콤한 덮밥",
            "달콤한 소스 덮밥과 음료",
            None,
            "LUNCH",
        )
    if diet_issue == "포화지방/튀김/가공식품 주의":
        return (
            {"fat_g": 31, "saturated_fat_g": 10, "calories_kcal": 640, "fiber_g": 1.5},
            "치킨과 감자튀김",
            "치킨과 감자튀김",
            None,
            "DINNER",
        )
    if diet_issue == "열량/야식/과식":
        return (
            {"calories_kcal": 760, "fat_g": 18, "fiber_g": 2.0},
            "늦은 야식",
            "늦은 밤 피자와 간식",
            "늦은 야식",
            "NIGHT",
        )
    return (
        {"calories_kcal": 360, "protein_g": 18, "fiber_g": 4, "sodium_mg": 260},
        "잡곡밥과 채소 반찬",
        "잡곡밥, 두부, 채소 반찬",
        None,
        "LUNCH",
    )


def _user_context(disease_groups: list[str]) -> str:
    labels = {
        "HTN": "혈압 관리",
        "DM": "혈당 관리",
        "DL": "지질 관리",
        "OBE": "체중 관리",
        "ANEM": "철분 보완 참고",
        "FL": "간 건강 관리",
        "CKD": "신장 건강 관리",
    }
    return ", ".join(labels.get(group, group) for group in disease_groups)


async def generate_outputs(cases: list[DietQaCase] | None = None) -> list[dict[str, Any]]:
    return [await generate_case_output(case) for case in (cases or build_qa_cases())]


async def generate_case_output(case: DietQaCase) -> dict[str, Any]:
    trace_events: list[dict[str, Any]] = []
    vector_retriever = FakeVectorRetriever(case.fake_vector_documents)
    patches = [
        patch.object(service.config, "DIET_RECOMMENDATION_RAG_STRATEGY", case.rag_strategy),
        patch.object(service.config, "RAG_ENABLED", case.rag_enabled),
        patch.object(service.config, "RAG_EMBEDDING_ENABLED", case.rag_embedding_enabled),
        patch.object(service.config, "RAG_EMBEDDING_PROVIDER", case.rag_embedding_provider),
        patch.object(service.config, "OPENAI_API_KEY", "qa-fake-openai-key"),
        patch.object(service.config, "DIET_RECOMMENDATION_LLM_REWRITE_ENABLED", False),
    ]

    if case.force_keyword_empty:
        patches.append(patch.object(service, "retrieve_keyword_rag_matches", lambda *args, **kwargs: []))

    def fake_record_langfuse_event(**kwargs: Any) -> bool:
        if not case.langfuse_available:
            return False
        trace_events.append(kwargs)
        return True

    patches.append(patch.object(service, "record_langfuse_event", fake_record_langfuse_event))

    with ExitStack() as stack:
        for patcher in patches:
            stack.enter_context(patcher)
        response = await service.build_diet_health_recommendation_response_async(
            diet_record=_diet_record(case),
            analysis_results=_analysis_results(case),
            health_record=None,
            active_challenges=_active_challenges(),
            vector_retriever=vector_retriever,
        )

    checks = run_response_checks(response)
    return {
        "case": _case_payload(case),
        "response": response,
        "checks": checks,
        "rag_trace": {
            "event_count": len(trace_events),
            "latest_metadata": trace_events[-1].get("metadata") if trace_events else None,
            "vector_calls": len(vector_retriever.calls),
            "langfuse_available": case.langfuse_available,
        },
    }


def _case_payload(case: DietQaCase) -> dict[str, Any]:
    payload = asdict(case)
    payload["analysis_types"] = [analysis_type.value for analysis_type in case.analysis_types]
    payload.pop("fake_vector_documents")
    return payload


def _diet_record(case: DietQaCase) -> SimpleNamespace:
    return SimpleNamespace(
        id=1000,
        user_id=10,
        meal_type=case.meal_type,
        description=case.description,
        memo=None,
        detected_foods=[
            {
                "name": case.food_name,
                "original_name": case.food_name,
                "query_name": case.food_name,
                "match_metadata": {
                    "provider": "mfds",
                    "status": "matched",
                    "nutrition": {
                        **case.nutrition,
                        "basis_label": case.nutrition.get("basis_label", "100g 기준"),
                    },
                },
            }
        ],
    )


def _analysis_results(case: DietQaCase) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            id=index + 1,
            analysis_type=analysis_type,
            risk_level=RiskLevel.CAUTION,
            analyzed_at=datetime(2026, 6, 15, tzinfo=UTC),
        )
        for index, analysis_type in enumerate(case.analysis_types)
    ]


def _active_challenges() -> list[SimpleNamespace]:
    return [SimpleNamespace(id=index + 1, title=title) for index, title in enumerate(ACTIVE_CHALLENGE_TITLES)]


def _fake_vector_document(disease_code: str, title: str) -> RetrievedDocument:
    return RetrievedDocument(
        content=f"{disease_code} internal chunk content must not appear in user output",
        title=title,
        source_name="QA fake source",
        url="https://example.test/rag",
        metadata={
            "chunk_key": f"qa:{disease_code.lower()}:chunk:0000",
            "document_key": f"qa:{disease_code.lower()}.md",
            "source_key": disease_code.lower(),
            "disease_code": disease_code,
            "review_status": "candidate_unreviewed",
            "status": "candidate_unreviewed",
            "retriever_strategy": "vector",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
        },
        score=0.91,
    )


def run_response_checks(response: dict[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(response, ensure_ascii=False, sort_keys=True)
    forbidden_phrases = [term for term in FORBIDDEN_PHRASES if term in serialized]
    internal_terms = [term for term in INTERNAL_TERMS if term in serialized]
    required_fields = [
        "diet_record_id",
        "nutrition_findings",
        "disease_context",
        "recommended_foods",
        "caution_foods",
        "recommended_challenges",
        "safety_notice",
        "rag_comment",
    ]
    return {
        "passed": not forbidden_phrases and not internal_terms and all(field in response for field in required_fields),
        "forbidden_phrases_found": forbidden_phrases,
        "internal_terms_found": internal_terms,
        "required_fields_present": {field: field in response for field in required_fields},
    }


def write_outputs(outputs: list[dict[str, Any]], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown_report(outputs), encoding="utf-8")


def render_markdown_report(outputs: list[dict[str, Any]]) -> str:
    lines = [
        "# Diet Recommendation QA Outputs",
        "",
        "이 리포트는 식단 추천 API 사용자-facing 문구 검수용 샘플입니다.",
        "DB write, OpenAI/LLM/외부 벡터 호출 없이 fake/stub 기반으로 생성합니다.",
        "",
        f"- 총 케이스: {len(outputs)}",
        f"- 자동 검사 통과: {sum(1 for output in outputs if output['checks']['passed'])}/{len(outputs)}",
        "",
    ]
    for output in outputs:
        case = output["case"]
        response = output["response"]
        rag_comment = response.get("rag_comment") or {}
        lines.extend(
            [
                f"## {case['case_id']} {case['title']}",
                "",
                f"- Case ID: {case['case_id']}",
                f"- 입력 요약: {case['user_context']} / {case['expected_focus']}",
                f"- disease_groups: {', '.join(case['disease_groups'])}",
                f"- meal_example: {case['meal_example']}",
                f"- diet_issue: {case['diet_issue']}",
                f"- RAG 전략: {case['rag_strategy']}",
                f"- nutrition_findings message: {_messages(response.get('nutrition_findings'))}",
                f"- disease_context message: {_messages(response.get('disease_context'))}",
                f"- recommended_foods: {', '.join(response.get('recommended_foods') or []) or '-'}",
                f"- caution_foods: {', '.join(response.get('caution_foods') or []) or '-'}",
                f"- recommended_challenges: {_challenge_details(response.get('recommended_challenges'))}",
                f"- rag_comment.summary: {rag_comment.get('summary') or '-'}",
                f"- rag_comment.evidence_sources: {json.dumps(rag_comment.get('evidence_sources') or [], ensure_ascii=False)}",
                f"- safety_notice: {response.get('safety_notice')}",
                f"- 금지 표현 검사 결과: {output['checks']['forbidden_phrases_found'] or '통과'}",
                f"- 내부값 노출 검사 결과: {output['checks']['internal_terms_found'] or '통과'}",
                f"- 사람이 확인할 포인트: {case['expected_focus']}",
                "",
            ]
        )
    return "\n".join(lines)


def _messages(items: Any) -> str:
    if not isinstance(items, list):
        return "-"
    return (
        " / ".join(str(item.get("message")) for item in items if isinstance(item, dict) and item.get("message")) or "-"
    )


def _reasons(items: Any) -> str:
    if not isinstance(items, list):
        return "-"
    return " / ".join(str(item.get("reason")) for item in items if isinstance(item, dict) and item.get("reason")) or "-"


def _challenge_details(items: Any) -> str:
    if not isinstance(items, list):
        return "-"
    details = []
    for item in items:
        if not isinstance(item, dict):
            continue
        challenge_id = item.get("challenge_id", "-")
        title = item.get("title") or "-"
        reason = item.get("reason") or "-"
        details.append(f"{challenge_id} | {title} | {reason}")
    return " / ".join(details) or "-"


def summarize_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, int] = {}
    disease_groups: set[str] = set()
    diet_issues: set[str] = set()
    rag_strategies: set[str] = set()
    failed_cases: list[str] = []
    for output in outputs:
        case = output["case"]
        groups[case["group"]] = groups.get(case["group"], 0) + 1
        disease_groups.update(case["disease_groups"])
        diet_issues.add(case["diet_issue"])
        rag_strategies.add(case["rag_strategy"])
        if not output["checks"]["passed"]:
            failed_cases.append(case["case_id"])
    return {
        "total_cases": len(outputs),
        "groups": groups,
        "disease_groups": sorted(disease_groups),
        "diet_issues": sorted(diet_issues),
        "rag_strategies": sorted(rag_strategies),
        "failed_cases": failed_cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diet recommendation QA response samples.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/qa/diet_recommendation_qa_outputs.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/qa/diet_recommendation_qa_outputs.md"),
    )
    parser.add_argument("--summary-json", action="store_true", help="Print summary as JSON.")
    return parser.parse_args()


async def _main_async() -> None:
    args = parse_args()
    outputs = await generate_outputs()
    write_outputs(outputs, args.output_json, args.output_md)
    summary = summarize_outputs(outputs) | {
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
        "db_write_performed": False,
        "openai_called": False,
        "llm_called": False,
        "embedding_called": False,
    }
    if args.summary_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Generated {summary['total_cases']} diet QA outputs.")
        print(f"JSON: {args.output_json}")
        print(f"Markdown: {args.output_md}")
        print(f"Failed checks: {summary['failed_cases'] or 'none'}")


def _dedupe_analysis_types(items: list[AnalysisType]) -> list[AnalysisType]:
    seen: set[AnalysisType] = set()
    result: list[AnalysisType] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


if __name__ == "__main__":
    asyncio.run(_main_async())
