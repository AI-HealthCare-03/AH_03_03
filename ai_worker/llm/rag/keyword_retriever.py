from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_worker.llm.rag.source_loader import (
    DEFAULT_RAG_SOURCE_DIR,
    RagSourceDocument,
    load_all_rag_source_documents,
)

DISEASE_SOURCE_MAP = {
    "DM": "diabetes",
    "DIABETES": "diabetes",
    "HTN": "hypertension",
    "HYPERTENSION": "hypertension",
    "DL": "dyslipidemia",
    "DYSLIPIDEMIA": "dyslipidemia",
    "OBE": "obesity",
    "OBESITY": "obesity",
}

SOURCE_KEYWORDS = {
    "hypertension": ("혈압", "수축기", "이완기", "고혈압", "저염", "나트륨"),
    "diabetes": ("혈당", "공복혈당", "당화혈색소", "hba1c", "당뇨", "당류"),
    "dyslipidemia": ("ldl", "hdl", "중성지방", "콜레스테롤", "이상지질혈증", "포화지방"),
    "obesity": ("bmi", "허리둘레", "비만", "체중", "열량", "칼로리"),
    "diet_nutrition": ("식단", "영양", "나트륨", "당류", "포화지방", "열량", "식사"),
    "food_nutrition_api": ("식품영양성분", "영양성분 api", "공공데이터", "data.go.kr", "api"),
}

SAFETY_SOURCE_ID = "safety_disclaimer"


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
            "disease_type": metadata.disease_type,
            "score": self.score,
            "matched_keywords": list(self.matched_keywords),
            "match_reason": self.match_reason,
        }


def retrieve_keyword_rag_matches(
    user_message: str = "",
    disease_type: str | None = None,
    *,
    top_k: int = 3,
    include_safety_disclaimer: bool = False,
    source_dir: Path = DEFAULT_RAG_SOURCE_DIR,
) -> list[KeywordRagMatch]:
    documents = {document.id: document for document in load_all_rag_source_documents(source_dir)}
    scored_matches: dict[str, KeywordRagMatch] = {}

    disease_source_id = _source_id_for_disease_type(disease_type)
    if disease_source_id and disease_source_id in documents:
        scored_matches[disease_source_id] = KeywordRagMatch(
            document=documents[disease_source_id],
            score=100,
            matched_keywords=(),
            match_reason="disease_type",
        )

    normalized_message = _normalize_text(user_message)
    for source_id, keywords in SOURCE_KEYWORDS.items():
        if source_id not in documents:
            continue
        matched_keywords = tuple(keyword for keyword in keywords if _normalize_text(keyword) in normalized_message)
        if not matched_keywords:
            continue
        keyword_score = len(matched_keywords) * 10
        existing = scored_matches.get(source_id)
        if existing is not None:
            scored_matches[source_id] = KeywordRagMatch(
                document=existing.document,
                score=existing.score + keyword_score,
                matched_keywords=matched_keywords,
                match_reason="disease_type+keyword",
            )
        else:
            scored_matches[source_id] = KeywordRagMatch(
                document=documents[source_id],
                score=keyword_score,
                matched_keywords=matched_keywords,
                match_reason="keyword",
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
    top_k: int = 3,
    include_safety_disclaimer: bool = False,
    source_dir: Path = DEFAULT_RAG_SOURCE_DIR,
):
    from ai_worker.llm.rag.rag_context_builder import build_retrieved_contexts

    matches = retrieve_keyword_rag_matches(
        user_message=user_message,
        disease_type=disease_type,
        top_k=top_k,
        include_safety_disclaimer=include_safety_disclaimer,
        source_dir=source_dir,
    )
    return build_retrieved_contexts(matches)


def _source_id_for_disease_type(disease_type: str | None) -> str | None:
    if disease_type is None:
        return None
    return DISEASE_SOURCE_MAP.get(str(disease_type).strip().upper())


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
