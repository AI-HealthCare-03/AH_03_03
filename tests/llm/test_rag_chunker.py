from ai_runtime.llm.rag.chunker import build_rag_chunk_drafts, build_rag_chunks_from_index, summarize_rag_chunks


def test_rag_chunker_uses_enabled_sources_only() -> None:
    chunks = build_rag_chunk_drafts()
    source_ids = {chunk.source_id for chunk in chunks}

    assert {"hypertension", "diabetes", "dyslipidemia", "obesity", "diet_nutrition", "diet_caution"} <= source_ids
    assert "ckd" not in source_ids
    assert "anemia" not in source_ids
    assert "fatty_liver" not in source_ids
    assert "diet_faq" not in source_ids
    assert all(chunk.enabled for chunk in chunks)


def test_rag_chunker_generates_stable_metadata_and_keys() -> None:
    chunks = build_rag_chunk_drafts()
    hypertension_chunks = [chunk for chunk in chunks if chunk.source_id == "hypertension"]

    assert hypertension_chunks
    first = hypertension_chunks[0]
    assert first.chunk_key == "rag:hypertension:section:000:chunk:0000"
    assert len(first.content_hash) == 64
    assert first.source_id == "hypertension"
    assert first.document_id == "hypertension"
    assert first.source_key == "hypertension"
    assert first.disease_code == "HTN"
    assert first.title
    assert first.filename == "hypertension.md"
    assert first.source_org == "대한고혈압학회"
    assert first.source_url
    assert first.source_type == "official_society"
    assert "나트륨" in first.topic_tags
    assert "sodium_high" in first.issue_keys
    assert first.review_status == "candidate_unreviewed"
    assert first.usage_scope
    assert first.safety_level == "normal"
    assert first.notes
    assert first.content_length == len(first.content)
    assert first.token_estimate > 0
    assert first.heading_path


def test_rag_chunker_splits_by_markdown_heading() -> None:
    chunks = build_rag_chunk_drafts()
    hypertension_titles = {chunk.section_title for chunk in chunks if chunk.source_id == "hypertension"}

    assert "고혈압, 주의혈압, 혈압 관리 후보 지식" in hypertension_titles
    assert "사용 목적" in hypertension_titles
    assert "요약" in hypertension_titles
    assert "keyword 후보" in hypertension_titles


def test_rag_chunk_dry_run_summary_contains_schema_fields() -> None:
    chunks = build_rag_chunk_drafts()
    summary = summarize_rag_chunks(chunks)
    payload = summary.to_dict()

    assert payload["enabled_documents"] == 6
    assert payload["disabled_documents"] == 4
    assert payload["total_chunks"] == len(chunks)
    assert payload["chunks_by_disease_code"]["HTN"] > 0
    assert "chunk_key" in payload["chunk_fields"]
    assert "content_hash" in payload["chunk_fields"]
    assert "source_key" in payload["chunk_fields"]
    assert "document_id" in payload["chunk_fields"]
    assert "disease_code" in payload["chunk_fields"]
    assert "review_status" in payload["chunk_fields"]
    assert "safety_level" in payload["chunk_fields"]
    assert "heading_path" in payload["chunk_fields"]
    assert all(source["chunk_count"] > 0 for source in payload["sources"])


def test_rag_chunker_can_include_disabled_sources_for_inspection() -> None:
    chunks = build_rag_chunk_drafts(enabled_only=False)
    source_ids = {chunk.source_id for chunk in chunks}

    assert "ckd" not in source_ids
    assert "anemia" not in source_ids
    assert "fatty_liver" not in source_ids
    assert "diet_faq" not in source_ids


def test_build_rag_chunks_from_index_matches_default_chunker() -> None:
    chunks = build_rag_chunks_from_index()
    default_chunks = build_rag_chunk_drafts()

    assert [chunk.chunk_key for chunk in chunks] == [chunk.chunk_key for chunk in default_chunks]


def test_chunk_index_is_sequential_per_document() -> None:
    chunks = build_rag_chunk_drafts()
    by_source: dict[str, list[int]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source_id, []).append(chunk.chunk_index)

    for indexes in by_source.values():
        assert indexes == list(range(len(indexes)))


def test_chunks_have_no_empty_content_and_hash_is_content_based() -> None:
    chunks = build_rag_chunk_drafts()

    assert all(chunk.content.strip() for chunk in chunks)
    for chunk in chunks:
        same_content = [candidate for candidate in chunks if candidate.content == chunk.content]
        assert all(candidate.content_hash == chunk.content_hash for candidate in same_content)
