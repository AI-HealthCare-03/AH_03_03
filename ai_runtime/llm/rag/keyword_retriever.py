from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ai_runtime.llm.rag.source_loader import (
    DEFAULT_RAG_SOURCE_DIR,
    RagSourceDocument,
    load_all_rag_source_documents,
)
from ai_runtime.llm.rag.source_trust import source_trust_level_for_type
from ai_runtime.llm.schemas import RetrievedContext

DISEASE_SOURCE_MAP = {
    "HTN": "hypertension",
    "DM": "diabetes",
    "DL": "dyslipidemia",
    "OBE": "obesity",
    "DIABETES": "diabetes",
    "HYPERTENSION": "hypertension",
    "DYSLIPIDEMIA": "dyslipidemia",
    "OBESITY": "obesity",
}

SOURCE_KEYWORDS = {
    "hypertension": ("혈압", "수축기", "이완기", "고혈압", "저염", "나트륨"),
    "diabetes": ("혈당", "공복혈당", "당화혈색소", "hba1c", "당뇨", "당류"),
    "dyslipidemia": ("ldl", "hdl", "중성지방", "콜레스테롤", "이상지질혈증", "포화지방"),
    "obesity": ("bmi", "허리둘레", "비만", "체중", "열량", "칼로리"),
    "diet_caution": ("주의", "제한", "진단", "처방", "의료진", "상담", "제한식", "개인별"),
    "diet_nutrition": ("식단", "영양", "나트륨", "당류", "포화지방", "열량", "식사"),
}

SAFETY_SOURCE_ID = "diet_caution"


@dataclass(frozen=True)
class KeywordRagMatch:
    document: RagSourceDocument
    score: int
    matched_keywords: tuple[str, ...]
    match_reason: str

    @property
    def source_id(self) -> str:
        return self.document.id

    def to_metadata(self) -> dict[str, object]:
        metadata = self.document.metadata
        return {
            "id": metadata.id,
            "title": metadata.title,
            "source_org": metadata.source_org,
            "source_url": metadata.source_url,
            "year": metadata.year,
            "status": metadata.status,
            "source_type": metadata.source_type,
            "source_trust_level": source_trust_level_for_type(metadata.source_type),
            "disease_type": metadata.disease_type,
            "disease_code": metadata.disease_code,
            "topic_tags": list(metadata.topic_tags),
            "issue_keys": list(metadata.issue_keys),
            "score": self.score,
            "matched_keywords": list(self.matched_keywords),
            "match_reason": self.match_reason,
        }


def retrieve_keyword_rag_matches(
    user_message: str = "",
    disease_type: str | None = None,
    *,
    disease_code: str | None = None,
    topic_tags: Iterable[str] | None = None,
    issue_keys: Iterable[str] | None = None,
    top_k: int = 3,
    include_safety_disclaimer: bool = False,
    source_dir: Path = DEFAULT_RAG_SOURCE_DIR,
) -> list[KeywordRagMatch]:
    """질환 코드와 짧은 키워드만 사용해 시연용 후보 문서를 고른다."""
    documents = {document.id: document for document in load_all_rag_source_documents(source_dir)}
    scored_matches: dict[str, KeywordRagMatch] = {}

    for source_id in _source_ids_for_disease_code(documents, disease_code or disease_type):
        scored_matches[source_id] = KeywordRagMatch(
            document=documents[source_id],
            score=100,
            matched_keywords=(),
            match_reason="disease_code",
        )

    normalized_message = _normalize_text(user_message)
    filter_keywords = tuple(topic_tags or ()) + tuple(issue_keys or ())
    for source_id, document in documents.items():
        keywords = _keywords_for_document(source_id, document, extra_keywords=filter_keywords)
        matched_keywords = tuple(keyword for keyword in keywords if _normalize_text(keyword) in normalized_message)
        matched_filters = tuple(keyword for keyword in filter_keywords if keyword in document.metadata.issue_keys)
        if not matched_keywords and not matched_filters:
            continue
        keyword_score = len(matched_keywords) * 10 + len(matched_filters) * 25
        existing = scored_matches.get(source_id)
        if existing is not None:
            scored_matches[source_id] = KeywordRagMatch(
                document=existing.document,
                score=existing.score + keyword_score,
                matched_keywords=matched_keywords,
                match_reason="disease_code+keyword",
            )
        else:
            scored_matches[source_id] = KeywordRagMatch(
                document=document,
                score=keyword_score,
                matched_keywords=matched_keywords,
                match_reason="issue_key" if matched_filters and not matched_keywords else "keyword",
            )

    matches = sorted(scored_matches.values(), key=lambda match: (-match.score, match.source_id))
    selected = matches[: max(top_k, 0)]

    if include_safety_disclaimer and SAFETY_SOURCE_ID in documents:
        selected = _append_safety_disclaimer(selected, documents[SAFETY_SOURCE_ID])

    return selected


def retrieve_keyword_rag_contexts(
    user_message: str = "",
    disease_type: str | None = None,
    *,
    disease_code: str | None = None,
    topic_tags: Iterable[str] | None = None,
    issue_keys: Iterable[str] | None = None,
    top_k: int = 3,
    include_safety_disclaimer: bool = False,
    source_dir: Path = DEFAULT_RAG_SOURCE_DIR,
) -> list[RetrievedContext]:
    from ai_runtime.llm.rag.rag_context_builder import build_retrieved_contexts

    matches = retrieve_keyword_rag_matches(
        user_message=user_message,
        disease_type=disease_type,
        disease_code=disease_code,
        topic_tags=topic_tags,
        issue_keys=issue_keys,
        top_k=top_k,
        include_safety_disclaimer=include_safety_disclaimer,
        source_dir=source_dir,
    )
    return build_retrieved_contexts(matches)


def _source_ids_for_disease_code(documents: dict[str, RagSourceDocument], disease_code: str | None) -> list[str]:
    if disease_code is None:
        return []
    normalized = str(disease_code).strip().upper()
    by_direct_code = [
        source_id for source_id, document in documents.items() if document.metadata.disease_code.upper() == normalized
    ]
    if by_direct_code:
        return by_direct_code
    mapped = DISEASE_SOURCE_MAP.get(normalized)
    return [mapped] if mapped and mapped in documents else []


def _keywords_for_document(
    source_id: str,
    document: RagSourceDocument,
    *,
    extra_keywords: Iterable[str],
) -> tuple[str, ...]:
    return (
        *SOURCE_KEYWORDS.get(source_id, ()),
        *document.metadata.topic_tags,
        *document.metadata.issue_keys,
        *extra_keywords,
    )


def _normalize_text(value: str) -> str:
    return value.casefold().replace(" ", "")


def _append_safety_disclaimer(
    selected: list[KeywordRagMatch],
    safety_document: RagSourceDocument,
) -> list[KeywordRagMatch]:
    if any(match.source_id == SAFETY_SOURCE_ID for match in selected):
        return selected
    return [
        *selected,
        KeywordRagMatch(
            document=safety_document,
            score=1,
            matched_keywords=(),
            match_reason="safety_disclaimer",
        ),
    ]
