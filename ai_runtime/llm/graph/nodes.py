from __future__ import annotations

import re
from time import perf_counter
from typing import Any

from langchain_core.documents import Document

from ai_runtime.llm.health_chatbot import CAUTION_MESSAGE, infer_main_health_chatbot_intent
from ai_runtime.llm.llm_client import get_openai_model, record_langfuse_event
from ai_runtime.llm.rag import retrieve_keyword_rag_contexts
from ai_runtime.llm.rag.rag_context_builder import build_reference_sources, build_reference_summary
from ai_runtime.llm.rag_generator import generate_main_health_rag_response
from ai_runtime.llm.response_router import route_main_health_chatbot_response
from ai_runtime.llm.safety import check_medical_safety, detect_mental_health_safety
from ai_runtime.llm.schemas import MainHealthChatbotInput
from app.core import config

from .state import HealthChatbotGraphState

SOURCE_GRAPH = "langgraph_chatbot"


def normalize_input(state: HealthChatbotGraphState) -> HealthChatbotGraphState:
    message = str(state.get("user_message") or "").strip()
    metadata = {
        **state.get("metadata", {}),
        "graph_version": "health_chatbot_langgraph_v1",
        "message_length": len(message),
    }
    trace_metadata = {
        **state.get("trace_metadata", {}),
        "sanitized_message_preview": sanitize_for_trace(message),
    }
    next_state = {
        **state,
        "user_message": message,
        "user_context": state.get("user_context") or {},
        "metadata": metadata,
        "trace_metadata": trace_metadata,
        "intent": state.get("intent"),
        "safety_level": state.get("safety_level"),
        "safety_response": state.get("safety_response"),
        "should_bypass_llm": bool(state.get("should_bypass_llm", False)),
        "retrieved_docs": state.get("retrieved_docs") or [],
        "reference_sources": state.get("reference_sources") or [],
        "reference_summary": state.get("reference_summary"),
        "llm_answer": state.get("llm_answer"),
        "final_answer": state.get("final_answer"),
        "recommended_actions": state.get("recommended_actions") or [],
        "fallback_reason": state.get("fallback_reason"),
        "source": state.get("source") or SOURCE_GRAPH,
        "caution_message": state.get("caution_message") or CAUTION_MESSAGE,
        "is_safe": bool(state.get("is_safe", True)),
        "safety_result": state.get("safety_result") or {},
        "use_real_llm": bool(state.get("use_real_llm", False)),
        "use_rag": bool(state.get("use_rag", True)),
    }
    trace_graph_node("normalize_input", next_state, {"message_length": len(message)})
    return next_state


def check_mental_health_safety(state: HealthChatbotGraphState) -> HealthChatbotGraphState:
    result = detect_mental_health_safety(state["user_message"])
    if result is None:
        trace_graph_node("check_mental_health_safety", state, {"safety_level": None, "should_bypass_llm": False})
        return state

    should_bypass = result.level == "crisis"
    next_state = {
        **state,
        "intent": result.intent,
        "safety_level": result.level,
        "safety_response": result.response,
        "should_bypass_llm": should_bypass,
        "fallback_reason": "mental_health_crisis_bypass" if should_bypass else state.get("fallback_reason"),
        "source": "safety_policy",
    }
    trace_graph_node(
        "check_mental_health_safety",
        next_state,
        {
            "safety_level": result.level,
            "should_bypass_llm": should_bypass,
            "fallback_reason": next_state.get("fallback_reason"),
        },
    )
    return next_state


def classify_intent(state: HealthChatbotGraphState) -> HealthChatbotGraphState:
    if state.get("intent"):
        trace_graph_node("classify_intent", state, {"intent": state["intent"], "source": "safety_policy"})
        return state

    intent = infer_main_health_chatbot_intent(state["user_message"])
    next_state = {
        **state,
        "intent": intent,
    }
    trace_graph_node("classify_intent", next_state, {"intent": intent})
    return next_state


def retrieve_rag_context(state: HealthChatbotGraphState) -> HealthChatbotGraphState:
    if not state.get("use_rag") or not config.RAG_ENABLED:
        next_state = {
            **state,
            "fallback_reason": state.get("fallback_reason") or "rag_disabled",
            "trace_metadata": {
                **state.get("trace_metadata", {}),
                "retrieval": {"enabled": False, "reason": "rag_disabled"},
            },
        }
        trace_graph_node("retrieve_rag_context", next_state, {"enabled": False, "reason": "rag_disabled"})
        return next_state

    started_at = perf_counter()
    contexts = retrieve_keyword_rag_contexts(
        user_message=state["user_message"],
        top_k=2,
        include_safety_disclaimer=True,
    )
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
    documents = [_context_to_document(context) for context in contexts]
    reference_sources = build_reference_sources(contexts)
    reference_summary = build_reference_summary(contexts)
    next_state = {
        **state,
        "retrieved_docs": [_document_to_payload(document) for document in documents],
        "reference_sources": reference_sources,
        "reference_summary": reference_summary,
        "trace_metadata": {
            **state.get("trace_metadata", {}),
            "retrieval": {
                "enabled": True,
                "elapsed_ms": elapsed_ms,
                "document_ids": [source.get("id") for source in reference_sources],
                "source_types": [source.get("source_org") for source in reference_sources],
                "reference_summary": reference_summary,
            },
        },
    }
    trace_graph_node(
        "retrieve_rag_context",
        next_state,
        {
            "enabled": True,
            "elapsed_ms": elapsed_ms,
            "document_count": len(documents),
            "document_ids": [source.get("id") for source in reference_sources],
        },
    )
    return next_state


def generate_llm_answer(state: HealthChatbotGraphState) -> HealthChatbotGraphState:
    if state.get("safety_response"):
        final_answer = _with_caution(str(state["safety_response"]))
        safety_result = check_medical_safety(final_answer)
        next_state = {
            **state,
            "llm_answer": final_answer,
            "source": "safety_policy",
            "is_safe": safety_result["is_safe"],
            "safety_result": _with_graph_metadata(safety_result, state, node="generate_llm_answer"),
        }
        trace_graph_node(
            "generate_llm_answer",
            next_state,
            {"llm_used": False, "source": "safety_policy", "safety_level": state.get("safety_level")},
        )
        return next_state

    if state.get("retrieved_docs"):
        retrieved_context = _retrieved_docs_to_context_text(state["retrieved_docs"])
        started_at = perf_counter()
        output = generate_main_health_rag_response(
            user_message=state["user_message"],
            retrieved_context=retrieved_context,
            context_sources=state.get("reference_sources") or [],
            use_real_llm=state.get("use_real_llm", False),
        )
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
    else:
        started_at = perf_counter()
        output = route_main_health_chatbot_response(
            MainHealthChatbotInput(user_message=state["user_message"]),
            use_llm_fallback=False,
            use_llm_rewrite=state.get("use_real_llm", False),
            use_real_llm=state.get("use_real_llm", False),
        )
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)

    next_state = {
        **state,
        "intent": output.intent,
        "llm_answer": output.answer,
        "source": output.source,
        "caution_message": output.caution_message,
        "is_safe": output.is_safe,
        "safety_result": _with_graph_metadata(output.safety_result, state, node="generate_llm_answer"),
        "trace_metadata": {
            **state.get("trace_metadata", {}),
            "generation": {
                "elapsed_ms": elapsed_ms,
                "model_name": get_openai_model() if state.get("use_real_llm") else None,
                "prompt_version": _prompt_version_for_source(output.source),
                "llm_requested": state.get("use_real_llm", False),
                "llm_source": output.source,
            },
        },
    }
    trace_graph_node(
        "generate_llm_answer",
        next_state,
        {
            "elapsed_ms": elapsed_ms,
            "model_name": get_openai_model() if state.get("use_real_llm") else None,
            "prompt_version": _prompt_version_for_source(output.source),
            "llm_requested": state.get("use_real_llm", False),
            "source": output.source,
        },
    )
    return next_state


def check_grounding_or_fallback(state: HealthChatbotGraphState) -> HealthChatbotGraphState:
    answer = str(state.get("llm_answer") or "")
    safety_result = check_medical_safety(answer)
    if safety_result["is_safe"]:
        next_state = {
            **state,
            "is_safe": bool(state.get("is_safe", True)),
            "safety_result": _with_graph_metadata(
                {
                    **state.get("safety_result", {}),
                    "medical_safety": safety_result,
                },
                state,
                node="check_grounding_or_fallback",
            ),
        }
        trace_graph_node("check_grounding_or_fallback", next_state, {"fallback_used": False})
        return next_state

    fallback_answer = (
        "건강 관련 판단은 입력된 정보만으로 확정하기 어렵습니다. "
        "생활습관 관리 방향은 참고용으로만 활용해 주세요. "
        f"{CAUTION_MESSAGE}"
    )
    fallback_safety = check_medical_safety(fallback_answer)
    next_state = {
        **state,
        "llm_answer": fallback_answer,
        "fallback_reason": "medical_safety_check_failed",
        "source": "rule_based_graph_fallback",
        "is_safe": fallback_safety["is_safe"],
        "safety_result": _with_graph_metadata(fallback_safety, state, node="check_grounding_or_fallback"),
    }
    trace_graph_node(
        "check_grounding_or_fallback",
        next_state,
        {"fallback_used": True, "fallback_reason": "medical_safety_check_failed"},
    )
    return next_state


def build_recommended_actions(state: HealthChatbotGraphState) -> HealthChatbotGraphState:
    actions = _recommended_actions(
        message=state["user_message"],
        intent=state.get("intent"),
        safety_level=state.get("safety_level"),
        context_type=state.get("context_type"),
    )
    next_state = {
        **state,
        "recommended_actions": actions,
    }
    trace_graph_node("build_recommended_actions", next_state, {"action_count": len(actions)})
    return next_state


def format_final_response(state: HealthChatbotGraphState) -> HealthChatbotGraphState:
    answer = state.get("llm_answer") or state.get("safety_response") or ""
    final_answer = _with_caution(answer)
    safety_result = check_medical_safety(final_answer)
    actions = state.get("recommended_actions") or _recommended_actions(
        message=state["user_message"],
        intent=state.get("intent"),
        safety_level=state.get("safety_level"),
        context_type=state.get("context_type"),
    )
    next_state = {
        **state,
        "final_answer": final_answer,
        "recommended_actions": actions,
        "caution_message": CAUTION_MESSAGE,
        "is_safe": bool(state.get("is_safe", True) and safety_result["is_safe"]),
        "safety_result": _with_graph_metadata(
            {
                **state.get("safety_result", {}),
                "medical_safety": safety_result,
            },
            state,
            node="format_final_response",
        ),
    }
    trace_graph_node(
        "format_final_response",
        next_state,
        {"source": next_state["source"], "intent": next_state.get("intent"), "is_safe": next_state["is_safe"]},
    )
    return next_state


def should_bypass_llm(state: HealthChatbotGraphState) -> str:
    return "bypass" if state.get("should_bypass_llm") else "continue"


def trace_graph_node(
    node_name: str,
    state: HealthChatbotGraphState,
    metadata: dict[str, Any] | None = None,
) -> bool:
    trace_metadata = state.get("trace_metadata", {})
    return record_langfuse_event(
        name=f"health_chatbot.graph.{node_name}",
        input_payload={
            "message_length": len(state.get("user_message") or ""),
            "context_type": state.get("context_type"),
        },
        output_payload={
            "intent": state.get("intent"),
            "safety_level": state.get("safety_level"),
            "should_bypass_llm": state.get("should_bypass_llm"),
            "fallback_reason": state.get("fallback_reason"),
        },
        metadata={
            "graph_version": state.get("metadata", {}).get("graph_version"),
            "sanitized_message_preview": trace_metadata.get("sanitized_message_preview"),
            **(metadata or {}),
        },
    )


def sanitize_for_trace(value: str, limit: int = 80) -> str:
    sanitized = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[email]", value)
    sanitized = re.sub(r"01[016789][- ]?\d{3,4}[- ]?\d{4}", "[phone]", sanitized)
    sanitized = re.sub(r"\d{2,3}[- ]?\d{3,4}[- ]?\d{4}", "[phone]", sanitized)
    return sanitized[:limit]


def _context_to_document(context) -> Document:
    return Document(
        page_content=context.content,
        metadata={
            **context.metadata,
            "title": context.title,
            "source_name": context.source_name,
            "url": context.url,
        },
    )


def _document_to_payload(document: Document) -> dict[str, Any]:
    return {
        "content": document.page_content,
        "metadata": dict(document.metadata),
    }


def _retrieved_docs_to_context_text(documents: list[dict[str, Any]]) -> str:
    chunks = []
    for document in documents:
        metadata = document.get("metadata") or {}
        chunks.append(
            "\n".join(
                [
                    f"[{metadata.get('id')}] {metadata.get('title')}",
                    f"source_org: {metadata.get('source_org') or metadata.get('source_name')}",
                    f"status: {metadata.get('status')}",
                    str(document.get("content") or "").strip(),
                ]
            )
        )
    return "\n\n---\n\n".join(chunks)


def _with_caution(answer: str) -> str:
    final_answer = answer.strip()
    if "진단이 아니" not in final_answer or "의료진 상담" not in final_answer:
        final_answer = f"{final_answer} {CAUTION_MESSAGE}"
    return final_answer


def _with_graph_metadata(safety_result: dict[str, Any], state: HealthChatbotGraphState, *, node: str) -> dict[str, Any]:
    metadata = {
        **safety_result.get("metadata", {}),
        "graph_version": state.get("metadata", {}).get("graph_version"),
        "graph_node": node,
        "trace_metadata": state.get("trace_metadata", {}),
    }
    return {
        **safety_result,
        "metadata": metadata,
    }


def _prompt_version_for_source(source: str) -> str | None:
    if source == "rag_llm":
        return "main_health_rag_v1"
    if source == "openai_rewrite":
        return "main_rewrite_v1"
    return None


def _recommended_actions(
    *,
    message: str,
    intent: str | None,
    safety_level: str | None,
    context_type: str | None,
) -> list[str]:
    if intent == "mental_health_crisis_support" or safety_level == "crisis":
        return ["가까운 보호자에게 알리기", "119 또는 112에 도움 요청", "전문기관 상담 연결"]
    if safety_level == "professional_support":
        return ["수면과 식사 기록하기", "가벼운 자기관리 목표 세우기", "전문 상담 일정 확인"]
    if safety_level == "self_care":
        return ["수면 시간 기록하기", "짧은 호흡 챌린지", "가벼운 산책 목표 세우기"]

    normalized_message = message.lower()
    normalized_context = str(context_type or "").upper()
    if normalized_context == "DIET" or any(keyword in normalized_message for keyword in ["식단", "음식", "칼로리"]):
        return ["오늘 식단 기록하기", "식단 분석 확인하기", "단 음료 대신 물 선택하기"]
    if normalized_context == "CHALLENGE" or any(
        keyword in normalized_message for keyword in ["운동", "걷기", "챌린지", "습관"]
    ):
        return ["챌린지 목록 보기", "오늘 완료 체크하기", "주 3회 걷기 목표 세우기"]
    if any(keyword in normalized_message for keyword in ["혈당", "당뇨", "hba1c"]):
        return ["건강정보 입력", "건강 분석 실행하기", "식후 10분 산책 챌린지 참여"]
    if any(keyword in normalized_message for keyword in ["혈압", "고혈압"]):
        return ["혈압 기록하기", "나트륨 줄이기", "수면 시간 점검"]
    if any(keyword in normalized_message for keyword in ["약", "복약", "영양제"]):
        return ["복약 기록하기", "복약 알림 설정", "의료진 상담 메모 남기기"]
    return ["건강정보 입력", "분석 준비 상태 확인", "대시보드 보기"]
