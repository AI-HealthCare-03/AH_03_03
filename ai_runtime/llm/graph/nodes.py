from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from langchain_core.documents import Document

from ai_runtime.llm.health_chatbot import CAUTION_MESSAGE, infer_main_health_chatbot_intent
from ai_runtime.llm.llm_client import get_openai_model, record_langfuse_event
from ai_runtime.llm.prompt_templates import (
    FALLBACK_SAFE_RESPONSE_PROMPT_VERSION,
    MAIN_HEALTH_RAG_PROMPT_VERSION,
    MAIN_REWRITE_PROMPT_VERSION,
)
from ai_runtime.llm.rag import RetrievedDocument, disabled_rag_retrieval_result, get_default_rag_retriever
from ai_runtime.llm.rag.source_trust import (
    is_low_trust_level,
    lowest_source_trust_level,
    source_trust_level_for_metadata,
)
from ai_runtime.llm.rag_generator import generate_main_health_rag_response
from ai_runtime.llm.response_router import route_main_health_chatbot_response
from ai_runtime.llm.safety import check_medical_safety, detect_mental_health_safety
from ai_runtime.llm.schemas import MainHealthChatbotInput
from app.core import config
from app.core.providers import has_langfuse_config

from .state import HealthChatbotGraphState

SOURCE_GRAPH = "langgraph_chatbot"
_REAL_RECORD_LANGFUSE_EVENT = record_langfuse_event


@dataclass(frozen=True)
class RecommendedAction:
    action_type: str
    title: str
    description: str
    reason: str
    priority: int
    safety_level: str | None = None
    related_risk_factor: str | None = None
    expected_benefit: str | None = None
    requires_professional_help: bool = False

    def to_public_label(self) -> str:
        return self.title

    def to_trace_metadata(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "title": self.title,
            "reason": self.reason,
            "priority": self.priority,
            "safety_level": self.safety_level,
            "related_risk_factor": self.related_risk_factor,
            "expected_benefit": self.expected_benefit,
            "requires_professional_help": self.requires_professional_help,
        }


def normalize_input(state: HealthChatbotGraphState) -> HealthChatbotGraphState:
    node_started_at = perf_counter()
    message = str(state.get("user_message") or "").strip()
    metadata = _metadata_with_node(
        {
            **state.get("metadata", {}),
            "graph_version": "health_chatbot_langgraph_v1",
            "message_length": len(message),
        },
        "normalize_input",
        started_at=node_started_at,
    )
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
    node_started_at = perf_counter()
    result = detect_mental_health_safety(state["user_message"])
    if result is None:
        next_state = {
            **state,
            "metadata": _metadata_with_node(
                state.get("metadata", {}),
                "check_mental_health_safety",
                started_at=node_started_at,
            ),
        }
        trace_graph_node("check_mental_health_safety", next_state, {"safety_level": None, "should_bypass_llm": False})
        return next_state

    should_bypass = result.level == "crisis"
    next_state = {
        **state,
        "metadata": _metadata_with_node(
            state.get("metadata", {}),
            "check_mental_health_safety",
            started_at=node_started_at,
        ),
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
    node_started_at = perf_counter()
    if state.get("intent"):
        next_state = {
            **state,
            "metadata": _metadata_with_node(
                state.get("metadata", {}),
                "classify_intent",
                started_at=node_started_at,
            ),
        }
        trace_graph_node("classify_intent", next_state, {"intent": next_state["intent"], "source": "safety_policy"})
        return next_state

    intent = infer_main_health_chatbot_intent(state["user_message"])
    next_state = {
        **state,
        "metadata": _metadata_with_node(state.get("metadata", {}), "classify_intent", started_at=node_started_at),
        "intent": intent,
    }
    trace_graph_node("classify_intent", next_state, {"intent": intent})
    return next_state


def retrieve_rag_context(state: HealthChatbotGraphState) -> HealthChatbotGraphState:
    node_started_at = perf_counter()
    if not state.get("use_rag") or not config.RAG_ENABLED:
        result = disabled_rag_retrieval_result()
        next_state = {
            **state,
            "metadata": _metadata_with_node(
                state.get("metadata", {}),
                "retrieve_rag_context",
                started_at=node_started_at,
            ),
            "fallback_reason": state.get("fallback_reason") or result.fallback_reason,
            "trace_metadata": {
                **state.get("trace_metadata", {}),
                "retrieval": result.trace_metadata,
            },
        }
        trace_graph_node("retrieve_rag_context", next_state, {"enabled": False, "reason": "rag_disabled"})
        return next_state

    retrieval_started_at = perf_counter()
    result = get_default_rag_retriever().retrieve(
        query=state["user_message"] or "",
        top_k=2,
        include_safety_disclaimer=True,
    )
    elapsed_ms = round((perf_counter() - retrieval_started_at) * 1000, 2)
    documents = [_retrieved_document_to_langchain_document(document) for document in result.documents]
    next_state = {
        **state,
        "metadata": _metadata_with_node(
            state.get("metadata", {}),
            "retrieve_rag_context",
            started_at=node_started_at,
        ),
        "retrieved_docs": [_document_to_payload(document) for document in documents],
        "reference_sources": result.reference_sources,
        "reference_summary": result.reference_summary,
        "fallback_reason": state.get("fallback_reason") or result.fallback_reason,
        "trace_metadata": {
            **state.get("trace_metadata", {}),
            "retrieval": {
                "enabled": True,
                "elapsed_ms": elapsed_ms,
                "reference_summary": result.reference_summary,
                **result.trace_metadata,
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
            "document_ids": result.trace_metadata.get("document_ids", []),
            "strategy": result.strategy,
            "fallback_reason": result.fallback_reason,
        },
    )
    return next_state


def generate_llm_answer(state: HealthChatbotGraphState) -> HealthChatbotGraphState:
    node_started_at = perf_counter()
    if state.get("safety_response"):
        final_answer = _with_caution(str(state["safety_response"]))
        safety_result = check_medical_safety(final_answer)
        prompt_version = FALLBACK_SAFE_RESPONSE_PROMPT_VERSION
        next_state = {
            **state,
            "metadata": _metadata_with_node(
                state.get("metadata", {}),
                "generate_llm_answer",
                started_at=node_started_at,
            ),
            "llm_answer": final_answer,
            "source": "safety_policy",
            "is_safe": safety_result["is_safe"],
            "safety_result": _with_graph_metadata(
                {
                    **safety_result,
                    "metadata": {
                        **safety_result.get("metadata", {}),
                        "prompt_version": prompt_version,
                    },
                },
                state,
                node="generate_llm_answer",
            ),
            "trace_metadata": {
                **state.get("trace_metadata", {}),
                "generation": {
                    "elapsed_ms": 0,
                    "model_name": None,
                    "prompt_version": prompt_version,
                    "llm_requested": False,
                    "llm_source": "safety_policy",
                    "bypassed_rewrite": True,
                },
            },
        }
        trace_graph_node(
            "generate_llm_answer",
            next_state,
            {
                "llm_used": False,
                "source": "safety_policy",
                "safety_level": state.get("safety_level"),
                "prompt_version": prompt_version,
            },
        )
        return next_state

    if state.get("retrieved_docs"):
        retrieved_context = _retrieved_docs_to_context_text(state["retrieved_docs"])
        generation_started_at = perf_counter()
        output = generate_main_health_rag_response(
            user_message=state["user_message"],
            retrieved_context=retrieved_context,
            context_sources=state.get("reference_sources") or [],
            use_real_llm=state.get("use_real_llm", False),
        )
        elapsed_ms = round((perf_counter() - generation_started_at) * 1000, 2)
    else:
        generation_started_at = perf_counter()
        output = route_main_health_chatbot_response(
            MainHealthChatbotInput(user_message=state["user_message"]),
            use_llm_fallback=False,
            use_llm_rewrite=state.get("use_real_llm", False),
            use_real_llm=state.get("use_real_llm", False),
        )
        elapsed_ms = round((perf_counter() - generation_started_at) * 1000, 2)

    prompt_version = _prompt_version_for_output(output)
    next_state = {
        **state,
        "metadata": _metadata_with_node(state.get("metadata", {}), "generate_llm_answer", started_at=node_started_at),
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
                "prompt_version": prompt_version,
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
            "prompt_version": prompt_version,
            "llm_requested": state.get("use_real_llm", False),
            "source": output.source,
        },
    )
    return next_state


def check_grounding_or_fallback(state: HealthChatbotGraphState) -> HealthChatbotGraphState:
    node_started_at = perf_counter()
    answer = str(state.get("llm_answer") or "")
    safety_result = check_medical_safety(answer)
    grounding_metadata = _build_grounding_metadata(state, safety_result=safety_result)
    grounded_answer = _answer_with_grounding_guardrails(answer, grounding_metadata)
    if grounded_answer != answer:
        answer = grounded_answer
        safety_result = check_medical_safety(answer)
        grounding_metadata = _build_grounding_metadata(state, safety_result=safety_result)

    if safety_result["is_safe"]:
        next_state = {
            **state,
            "metadata": _metadata_with_node(
                state.get("metadata", {}),
                "check_grounding_or_fallback",
                started_at=node_started_at,
            ),
            "is_safe": bool(state.get("is_safe", True)),
            "safety_result": _with_graph_metadata(
                {
                    **state.get("safety_result", {}),
                    "medical_safety": safety_result,
                },
                state,
                node="check_grounding_or_fallback",
            ),
            "llm_answer": answer,
            "fallback_reason": state.get("fallback_reason") or grounding_metadata["fallback_reason"],
            "trace_metadata": {
                **state.get("trace_metadata", {}),
                "grounding": grounding_metadata,
            },
        }
        trace_graph_node(
            "check_grounding_or_fallback",
            next_state,
            {
                "fallback_used": grounding_metadata["grounding_status"] != "grounded",
                "retrieved_doc_count": grounding_metadata["retrieved_doc_count"],
                "source_trust_levels": grounding_metadata["source_trust_levels"],
                "grounding_status": grounding_metadata["grounding_status"],
                "fallback_reason": grounding_metadata["fallback_reason"],
                "reference_source_ids": grounding_metadata["reference_source_ids"],
            },
        )
        return next_state

    fallback_answer = (
        "건강 관련 판단은 입력된 정보만으로 확정하기 어렵습니다. "
        "생활습관 관리 방향은 참고용으로만 활용해 주세요. "
        f"{CAUTION_MESSAGE}"
    )
    fallback_safety = check_medical_safety(fallback_answer)
    grounding_metadata = {
        **grounding_metadata,
        "grounding_status": "medical_safety_fallback",
        "fallback_reason": "medical_safety_check_failed",
    }
    next_state = {
        **state,
        "metadata": _metadata_with_node(
            state.get("metadata", {}),
            "check_grounding_or_fallback",
            started_at=node_started_at,
        ),
        "llm_answer": fallback_answer,
        "fallback_reason": "medical_safety_check_failed",
        "source": "rule_based_graph_fallback",
        "is_safe": fallback_safety["is_safe"],
        "safety_result": _with_graph_metadata(fallback_safety, state, node="check_grounding_or_fallback"),
        "trace_metadata": {
            **state.get("trace_metadata", {}),
            "grounding": grounding_metadata,
        },
    }
    trace_graph_node(
        "check_grounding_or_fallback",
        next_state,
        {
            "fallback_used": True,
            "retrieved_doc_count": grounding_metadata["retrieved_doc_count"],
            "source_trust_levels": grounding_metadata["source_trust_levels"],
            "grounding_status": grounding_metadata["grounding_status"],
            "fallback_reason": "medical_safety_check_failed",
            "reference_source_ids": grounding_metadata["reference_source_ids"],
        },
    )
    return next_state


def build_recommended_actions(state: HealthChatbotGraphState) -> HealthChatbotGraphState:
    node_started_at = perf_counter()
    action_specs = _recommended_action_specs(
        message=state["user_message"],
        intent=state.get("intent"),
        safety_level=state.get("safety_level"),
        context_type=state.get("context_type"),
        user_context=state.get("user_context") or {},
        risk_factors=state.get("risk_factors") or [],
    )
    actions = [action.to_public_label() for action in action_specs]
    recommendation_trace = _recommendation_trace_metadata(action_specs, state.get("safety_level"))
    next_state = {
        **state,
        "metadata": _metadata_with_node(
            state.get("metadata", {}),
            "build_recommended_actions",
            started_at=node_started_at,
        ),
        "trace_metadata": {
            **state.get("trace_metadata", {}),
            "recommended_actions": recommendation_trace,
        },
        "recommended_actions": actions,
    }
    trace_graph_node(
        "build_recommended_actions",
        next_state,
        {
            "recommended_action_count": recommendation_trace["recommended_action_count"],
            "action_types": recommendation_trace["action_types"],
            "safety_level": recommendation_trace["safety_level"],
            "top_recommendation_reason": recommendation_trace["top_recommendation_reason"],
        },
    )
    return next_state


def format_final_response(state: HealthChatbotGraphState) -> HealthChatbotGraphState:
    node_started_at = perf_counter()
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
        "metadata": _metadata_with_node(state.get("metadata", {}), "format_final_response", started_at=node_started_at),
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
    record_event_is_test_double = record_langfuse_event is not _REAL_RECORD_LANGFUSE_EVENT
    if not (config.LANGFUSE_ENABLED and has_langfuse_config(config)) and not record_event_is_test_double:
        return False

    trace_metadata = state.get("trace_metadata", {})
    graph_metadata = state.get("metadata", {})
    node_durations_ms = graph_metadata.get("node_durations_ms") or {}
    node_duration_ms = node_durations_ms.get(node_name) if isinstance(node_durations_ms, dict) else None
    return record_langfuse_event(
        name=f"health_chatbot.graph.{node_name}",
        input_payload={
            "message_length": len(state.get("user_message") or ""),
            "context_type": state.get("context_type"),
            "graph_run_id": graph_metadata.get("graph_run_id"),
        },
        output_payload={
            "intent": state.get("intent"),
            "safety_level": state.get("safety_level"),
            "should_bypass_llm": state.get("should_bypass_llm"),
            "fallback_reason": state.get("fallback_reason"),
            "source": state.get("source"),
        },
        metadata={
            "graph_run_id": graph_metadata.get("graph_run_id"),
            "graph_version": graph_metadata.get("graph_version"),
            "node_path": graph_metadata.get("node_path", []),
            "node_duration_ms": node_duration_ms,
            "sanitized_message_preview": trace_metadata.get("sanitized_message_preview"),
            **(metadata or {}),
        },
    )


def sanitize_for_trace(value: str, limit: int = 80) -> str:
    sanitized = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[email]", value)
    sanitized = re.sub(r"\b\d{6}[- ]?[1-4]\d{6}\b", "[rrn]", sanitized)
    sanitized = re.sub(r"01[016789][- ]?\d{3,4}[- ]?\d{4}", "[phone]", sanitized)
    sanitized = re.sub(r"\d{2,3}[- ]?\d{3,4}[- ]?\d{4}", "[phone]", sanitized)
    sanitized = re.sub(
        r"(?i)\b(api[_-]?key|token|password|passwd|secret|authorization|bearer)\b\s*[:=]?\s*['\"]?[^\s'\",;]+",
        r"\1=[secret]",
        sanitized,
    )
    sanitized = re.sub(r"(?i)\b(?:sk|pk|rk|key|token)-[a-z0-9_-]{8,}\b", "[secret]", sanitized)
    sanitized = re.sub(r"\beyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b", "[secret]", sanitized)
    health_terms = (
        "공복혈당",
        "혈당",
        "혈압",
        "수축기",
        "이완기",
        "콜레스테롤",
        "중성지방",
        "hdl",
        "ldl",
        "bmi",
        "체중",
        "몸무게",
        "허리둘레",
        "키",
        "신장",
    )
    term_pattern = "|".join(re.escape(term) for term in health_terms)
    sanitized = re.sub(
        rf"({term_pattern})\s*[:=]?\s*\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?",
        r"\1 [health_value]",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(r"\b\d{4,}(?:\.\d+)?\b", "[number]", sanitized)
    return sanitized[:limit]


def _retrieved_document_to_langchain_document(document: RetrievedDocument) -> Document:
    return Document(
        page_content=document.content,
        metadata={
            **document.metadata,
            "title": document.title,
            "source_name": document.source_name,
            "url": document.url,
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


def _build_grounding_metadata(
    state: HealthChatbotGraphState,
    *,
    safety_result: dict[str, Any],
) -> dict[str, Any]:
    retrieved_docs = state.get("retrieved_docs") or []
    reference_sources = state.get("reference_sources") or []
    fallback_reason = state.get("fallback_reason")
    source = state.get("source")
    source_trust_levels = _source_trust_levels(reference_sources=reference_sources, retrieved_docs=retrieved_docs)
    grounding_status = _grounding_status(
        state=state,
        source_trust_levels=source_trust_levels,
        safety_result=safety_result,
    )
    return {
        "retrieved_doc_count": len(retrieved_docs),
        "reference_source_count": len(reference_sources),
        "source_trust_levels": source_trust_levels,
        "lowest_source_trust_level": lowest_source_trust_level(source_trust_levels),
        "grounding_status": grounding_status,
        "fallback_reason": fallback_reason or _fallback_reason_for_grounding_status(grounding_status),
        "reference_source_ids": _reference_source_ids(reference_sources),
        "reference_summary_present": bool(state.get("reference_summary")),
        "source": source,
    }


def _grounding_status(
    *,
    state: HealthChatbotGraphState,
    source_trust_levels: list[str],
    safety_result: dict[str, Any],
) -> str:
    if state.get("source") == "safety_policy":
        return "safety_policy_priority"
    if not safety_result.get("is_safe", True):
        return "unsafe_answer"
    if not state.get("use_rag") or not config.RAG_ENABLED:
        return "rag_disabled"
    fallback_reason = state.get("fallback_reason")
    if fallback_reason in {"no_result", "empty_retrieved_context"}:
        return "no_reference"
    if fallback_reason == "safety_disclaimer_only":
        return "weak_reference"
    if not state.get("retrieved_docs") or not state.get("reference_sources"):
        return "no_reference"
    if not state.get("reference_summary"):
        return "summary_missing"
    if any(is_low_trust_level(level) for level in source_trust_levels):
        return "weak_reference"
    return "grounded"


def _answer_with_grounding_guardrails(answer: str, grounding_metadata: dict[str, Any]) -> str:
    grounding_status = grounding_metadata["grounding_status"]
    if grounding_status in {"no_reference", "summary_missing"} and "근거 문서를 찾지 못했습니다" not in answer:
        return f"근거 문서를 찾지 못했습니다. 아래 내용은 일반적인 건강관리 참고 정보로만 확인해 주세요. {answer}"
    if grounding_status == "weak_reference" and "근거 수준이 제한적" not in answer:
        return f"근거 수준이 제한적이므로 단정하지 않고 참고용으로 안내드립니다. {answer}"
    return answer


def _fallback_reason_for_grounding_status(grounding_status: str) -> str | None:
    if grounding_status == "no_reference":
        return "rag_no_reference"
    if grounding_status == "summary_missing":
        return "rag_reference_summary_missing"
    if grounding_status == "weak_reference":
        return "rag_weak_reference"
    if grounding_status == "unsafe_answer":
        return "medical_safety_check_failed"
    return None


def _source_trust_levels(
    *,
    reference_sources: list[dict[str, Any]],
    retrieved_docs: list[dict[str, Any]],
) -> list[str]:
    levels: list[str] = []
    for source in reference_sources:
        level = source.get("source_trust_level")
        if isinstance(level, str) and level:
            levels.append(level)
        else:
            levels.append(source_trust_level_for_metadata(source))
    if levels:
        return levels
    for document in retrieved_docs:
        metadata = document.get("metadata") if isinstance(document, dict) else None
        if isinstance(metadata, dict):
            levels.append(source_trust_level_for_metadata(metadata))
    return levels


def _reference_source_ids(reference_sources: list[dict[str, Any]]) -> list[str]:
    source_ids = []
    for source in reference_sources:
        source_id = source.get("id")
        if source_id is not None:
            source_ids.append(str(source_id))
    return source_ids


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


def _metadata_with_node(
    metadata: dict[str, Any],
    node: str,
    *,
    started_at: float | None = None,
) -> dict[str, Any]:
    node_path = list(metadata.get("node_path") or [])
    duration_ms = round((perf_counter() - started_at) * 1000, 2) if started_at is not None else 0.0
    node_durations_ms = dict(metadata.get("node_durations_ms") or {})
    node_durations_ms[node] = duration_ms
    return {
        **metadata,
        "node_path": [*node_path, node],
        "node_durations_ms": node_durations_ms,
        "latest_node_duration_ms": duration_ms,
    }


def _prompt_version_for_source(source: str) -> str | None:
    if source == "rag_llm":
        return MAIN_HEALTH_RAG_PROMPT_VERSION
    if source == "openai_rewrite":
        return MAIN_REWRITE_PROMPT_VERSION
    if source == "safety_policy":
        return FALLBACK_SAFE_RESPONSE_PROMPT_VERSION
    return None


def _prompt_version_for_output(output) -> str | None:
    metadata = output.safety_result.get("metadata", {}) if isinstance(output.safety_result, dict) else {}
    prompt_version = metadata.get("prompt_version")
    if isinstance(prompt_version, str) and prompt_version:
        return prompt_version
    return _prompt_version_for_source(output.source)


def _recommended_actions(
    *,
    message: str,
    intent: str | None,
    safety_level: str | None,
    context_type: str | None,
    user_context: dict[str, Any] | None = None,
    risk_factors: list[dict[str, Any]] | None = None,
) -> list[str]:
    return [
        action.to_public_label()
        for action in _recommended_action_specs(
            message=message,
            intent=intent,
            safety_level=safety_level,
            context_type=context_type,
            user_context=user_context or {},
            risk_factors=risk_factors or [],
        )
    ]


def _recommended_action_specs(
    *,
    message: str,
    intent: str | None,
    safety_level: str | None,
    context_type: str | None,
    user_context: dict[str, Any],
    risk_factors: list[dict[str, Any]],
) -> list[RecommendedAction]:
    related_risk_factor = _primary_related_risk_factor(user_context=user_context, risk_factors=risk_factors)
    if intent == "mental_health_crisis_support" or safety_level == "crisis":
        return [
            RecommendedAction(
                action_type="emergency_support",
                title="119 또는 112에 도움 요청",
                description="지금 안전이 걱정된다면 즉시 도움을 요청합니다.",
                reason="위기 키워드에서는 일반 챌린지보다 즉시 안전 확보가 우선입니다.",
                priority=1,
                safety_level=safety_level,
                expected_benefit="즉각적인 안전 확보와 긴급 지원 연결",
                requires_professional_help=True,
            ),
            RecommendedAction(
                action_type="guardian_contact",
                title="가까운 보호자에게 알리기",
                description="혼자 견디지 않도록 가까운 사람에게 현재 상태를 알립니다.",
                reason="보호자나 신뢰할 수 있는 사람과 함께 있는 것이 우선입니다.",
                priority=2,
                safety_level=safety_level,
                expected_benefit="혼자 있는 시간을 줄이고 도움을 받을 가능성을 높임",
                requires_professional_help=True,
            ),
            RecommendedAction(
                action_type="professional_support",
                title="전문기관 상담 연결",
                description="전문 상담 기관 또는 의료기관에 도움을 요청합니다.",
                reason="위기 상황은 전문적인 지원과 연결되는 것이 안전합니다.",
                priority=3,
                safety_level=safety_level,
                expected_benefit="전문가 도움을 통한 안전 계획 수립",
                requires_professional_help=True,
            ),
        ]
    if safety_level == "professional_support":
        return [
            RecommendedAction(
                action_type="self_care_tracking",
                title="수면과 식사 기록하기",
                description="수면, 식사, 기분 변화를 짧게 기록합니다.",
                reason="최근 상태를 정리하면 상담이나 자기관리에 참고할 수 있습니다.",
                priority=1,
                safety_level=safety_level,
                expected_benefit="상태 변화 파악과 자기관리 기준 마련",
            ),
            RecommendedAction(
                action_type="low_burden_goal",
                title="가벼운 자기관리 목표 세우기",
                description="부담이 낮은 목표를 하나만 정합니다.",
                reason="무기력하거나 번아웃이 있을 때는 낮은 난이도의 실천이 적절합니다.",
                priority=2,
                safety_level=safety_level,
                expected_benefit="일상 회복을 위한 작은 실행 경험",
            ),
            RecommendedAction(
                action_type="professional_support",
                title="전문 상담 일정 확인",
                description="상담센터, 의료기관, 학교/직장 상담 창구를 확인합니다.",
                reason="우울, 무기력, 번아웃 표현은 자기관리와 함께 전문 상담 권고가 필요합니다.",
                priority=3,
                safety_level=safety_level,
                expected_benefit="전문가와 상태를 점검하고 도움 계획을 세움",
                requires_professional_help=True,
            ),
        ]
    if safety_level == "self_care":
        return [
            RecommendedAction(
                action_type="sleep_tracking",
                title="수면 시간 기록하기",
                description="오늘 잠든 시간과 깬 시간을 기록합니다.",
                reason="스트레스, 불안, 수면 문제는 생활 리듬 확인이 도움이 될 수 있습니다.",
                priority=1,
                safety_level=safety_level,
                expected_benefit="수면 패턴 확인과 자기관리 시작",
            ),
            RecommendedAction(
                action_type="self_care_challenge",
                title="짧은 호흡 챌린지",
                description="1~3분 정도의 짧은 호흡 루틴을 시도합니다.",
                reason="일반 스트레스/불안 표현에서는 낮은 부담의 자기관리 챌린지를 추천할 수 있습니다.",
                priority=2,
                safety_level=safety_level,
                expected_benefit="긴장 완화와 즉시 실천 가능한 자기관리",
            ),
            RecommendedAction(
                action_type="light_activity",
                title="가벼운 산책 목표 세우기",
                description="몸 상태에 맞춰 짧게 걷는 목표를 세웁니다.",
                reason="무리하지 않는 신체활동은 기분 전환과 생활 리듬 관리에 도움이 될 수 있습니다.",
                priority=3,
                safety_level=safety_level,
                expected_benefit="생활 리듬과 활동량 회복",
            ),
        ]

    normalized_message = message.lower()
    normalized_context = str(context_type or "").upper()
    if normalized_context == "DIET" or any(keyword in normalized_message for keyword in ["식단", "음식", "칼로리"]):
        return [
            RecommendedAction(
                action_type="diet_log",
                title="오늘 식단 기록하기",
                description="오늘 먹은 음식을 간단히 기록합니다.",
                reason="최근 식단 기록은 질환군별 식단 관리 참고자료가 됩니다.",
                priority=1,
                safety_level=safety_level,
                related_risk_factor=related_risk_factor,
                expected_benefit="식습관 패턴 확인",
            ),
            RecommendedAction(
                action_type="diet_analysis_review",
                title="식단 분석 확인하기",
                description="AI 추정 결과를 확인하고 필요한 경우 수정합니다.",
                reason="식단 분석은 사용자 확인 후 관리 참고 정보로 활용하는 것이 안전합니다.",
                priority=2,
                safety_level=safety_level,
                related_risk_factor=related_risk_factor,
                expected_benefit="식단 관리 우선순위 확인",
            ),
            RecommendedAction(
                action_type="low_sugar_swap",
                title="단 음료 대신 물 선택하기",
                description="오늘 한 번은 단 음료 대신 물을 선택합니다.",
                reason="부담이 낮고 반복하기 쉬운 식습관 개선 행동입니다.",
                priority=3,
                safety_level=safety_level,
                related_risk_factor=related_risk_factor,
                expected_benefit="당류 섭취 관리에 도움",
            ),
        ]
    if normalized_context == "CHALLENGE" or any(
        keyword in normalized_message for keyword in ["운동", "걷기", "챌린지", "습관"]
    ):
        return [
            RecommendedAction(
                action_type="challenge_browse",
                title="챌린지 목록 보기",
                description="현재 상태에 맞는 챌린지 후보를 확인합니다.",
                reason="실행 부담과 개선 가능성을 함께 보고 선택하는 것이 좋습니다.",
                priority=1,
                safety_level=safety_level,
                related_risk_factor=related_risk_factor,
                expected_benefit="실행 가능한 생활습관 목표 선택",
            ),
            RecommendedAction(
                action_type="challenge_checkin",
                title="오늘 완료 체크하기",
                description="이미 시작한 챌린지가 있다면 오늘 실천 여부를 기록합니다.",
                reason="수행률은 다음 추천 난이도를 조정하는 참고 기준입니다.",
                priority=2,
                safety_level=safety_level,
                expected_benefit="수행률 기반 자기관리 지속",
            ),
            RecommendedAction(
                action_type="walking_goal",
                title="주 3회 걷기 목표 세우기",
                description="무리하지 않는 빈도로 걷기 목표를 설정합니다.",
                reason="일반 건강관리에서 부담이 낮은 활동 목표로 시작할 수 있습니다.",
                priority=3,
                safety_level=safety_level,
                related_risk_factor=related_risk_factor,
                expected_benefit="활동량 증가와 생활 리듬 개선",
            ),
        ]
    if any(keyword in normalized_message for keyword in ["혈당", "당뇨", "hba1c"]):
        return [
            RecommendedAction(
                action_type="health_data_input",
                title="건강정보 입력",
                description="최근 건강정보나 검진 수치를 입력합니다.",
                reason="입력 데이터가 있어야 위험요인 참고와 관리 우선순위를 더 잘 정리할 수 있습니다.",
                priority=1,
                safety_level=safety_level,
                related_risk_factor=related_risk_factor or "혈당",
                expected_benefit="분석 입력 데이터 신뢰도 개선",
            ),
            RecommendedAction(
                action_type="analysis_run",
                title="건강 분석 실행하기",
                description="입력된 건강정보를 바탕으로 분석을 실행합니다.",
                reason="분석 결과는 진단이 아니라 위험 관리 참고 정보로 활용됩니다.",
                priority=2,
                safety_level=safety_level,
                related_risk_factor=related_risk_factor or "혈당",
                expected_benefit="관리 우선순위 확인",
            ),
            RecommendedAction(
                action_type="self_care_challenge",
                title="식후 10분 산책 챌린지 참여",
                description="식후 무리 없는 짧은 걷기 목표를 시도합니다.",
                reason="혈당 관리 참고 행동으로 부담이 낮은 활동을 제안합니다.",
                priority=3,
                safety_level=safety_level,
                related_risk_factor=related_risk_factor or "혈당",
                expected_benefit="식후 활동량 관리에 도움",
            ),
        ]
    if any(keyword in normalized_message for keyword in ["혈압", "고혈압"]):
        return [
            RecommendedAction(
                action_type="blood_pressure_log",
                title="혈압 기록하기",
                description="최근 혈압 기록을 추가합니다.",
                reason="혈압 변화는 생활습관 관리 참고 지표입니다.",
                priority=1,
                safety_level=safety_level,
                related_risk_factor=related_risk_factor or "혈압",
                expected_benefit="혈압 변화 추이 확인",
            ),
            RecommendedAction(
                action_type="diet_habit",
                title="나트륨 줄이기",
                description="오늘 한 끼에서 짠 음식이나 국물 섭취를 줄여봅니다.",
                reason="고혈압 위험 관리 참고 행동으로 나트륨 섭취 점검이 중요할 수 있습니다.",
                priority=2,
                safety_level=safety_level,
                related_risk_factor=related_risk_factor or "혈압",
                expected_benefit="식습관 개선에 도움",
            ),
            RecommendedAction(
                action_type="sleep_check",
                title="수면 시간 점검",
                description="최근 수면 시간과 피로도를 확인합니다.",
                reason="수면과 스트레스는 혈압 관리 참고 요소입니다.",
                priority=3,
                safety_level=safety_level,
                related_risk_factor=related_risk_factor or "혈압",
                expected_benefit="생활 리듬 점검",
            ),
        ]
    if any(keyword in normalized_message for keyword in ["약", "복약", "영양제"]):
        return [
            RecommendedAction(
                action_type="medication_log",
                title="복약 기록하기",
                description="복용 중인 약이나 영양제를 기록합니다.",
                reason="복약 기록은 건강관리 참고 정보로 활용됩니다.",
                priority=1,
                safety_level=safety_level,
                expected_benefit="복약 이력 정리",
            ),
            RecommendedAction(
                action_type="medication_reminder",
                title="복약 알림 설정",
                description="복용 시간을 잊지 않도록 알림을 설정합니다.",
                reason="복약 누락을 줄이는 데 도움이 될 수 있습니다.",
                priority=2,
                safety_level=safety_level,
                expected_benefit="복약 루틴 유지",
            ),
            RecommendedAction(
                action_type="professional_note",
                title="의료진 상담 메모 남기기",
                description="궁금한 점을 진료 시 확인할 메모로 남깁니다.",
                reason="약 변경이나 중단은 전문가 상담이 필요합니다.",
                priority=3,
                safety_level=safety_level,
                expected_benefit="상담 준비와 안전한 의사결정 지원",
                requires_professional_help=True,
            ),
        ]
    return [
        RecommendedAction(
            action_type="health_data_input",
            title="건강정보 입력",
            description="기본 건강정보를 입력합니다.",
            reason="사용자 입력 데이터가 있어야 맞춤 추천의 신뢰도를 높일 수 있습니다.",
            priority=1,
            safety_level=safety_level,
            related_risk_factor=related_risk_factor,
            expected_benefit="추천 입력 데이터 보강",
        ),
        RecommendedAction(
            action_type="analysis_readiness",
            title="분석 준비 상태 확인",
            description="분석에 필요한 정보가 충분한지 확인합니다.",
            reason="분석 결과는 챌린지 추천과 설명의 입력으로 활용됩니다.",
            priority=2,
            safety_level=safety_level,
            related_risk_factor=related_risk_factor,
            expected_benefit="분석 기반 추천 준비",
        ),
        RecommendedAction(
            action_type="dashboard_review",
            title="대시보드 보기",
            description="최근 기록과 추천 행동을 한눈에 확인합니다.",
            reason="최근 식단, 운동, 복약, 챌린지 상태를 함께 보는 것이 좋습니다.",
            priority=3,
            safety_level=safety_level,
            related_risk_factor=related_risk_factor,
            expected_benefit="건강관리 현황 점검",
        ),
    ]


def _recommendation_trace_metadata(
    action_specs: list[RecommendedAction],
    safety_level: str | None,
) -> dict[str, Any]:
    first_action = action_specs[0] if action_specs else None
    return {
        "recommended_action_count": len(action_specs),
        "action_types": [action.action_type for action in action_specs],
        "safety_level": safety_level,
        "top_recommendation_reason": first_action.reason if first_action is not None else None,
        "actions": [action.to_trace_metadata() for action in action_specs],
    }


def _primary_related_risk_factor(
    *,
    user_context: dict[str, Any],
    risk_factors: list[dict[str, Any]],
) -> str | None:
    candidates: list[Any] = []
    candidates.extend(risk_factors)
    context_risk_factors = user_context.get("risk_factors") or user_context.get("riskFactors")
    if isinstance(context_risk_factors, list):
        candidates.extend(context_risk_factors)
    analysis_result = user_context.get("analysis_result") or user_context.get("analysisResult")
    if isinstance(analysis_result, dict):
        analysis_risk_factors = analysis_result.get("risk_factors") or analysis_result.get("riskFactors")
        if isinstance(analysis_risk_factors, list):
            candidates.extend(analysis_risk_factors)

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return sanitize_for_trace(candidate.strip(), limit=40)
        if isinstance(candidate, dict):
            for key in ("name", "factor", "risk_factor", "riskFactor", "disease_type", "diseaseType"):
                value = candidate.get(key)
                if isinstance(value, str) and value.strip():
                    return sanitize_for_trace(value.strip(), limit=40)
    return None
