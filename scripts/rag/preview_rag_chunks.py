from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_runtime.llm.rag.chunker import build_rag_chunks_from_index, summarize_rag_chunks
from ai_runtime.llm.rag.source_loader import DEFAULT_RAG_SOURCE_DIR, INDEX_FILE_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview RAG markdown chunks without DB writes or embeddings.")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    parser.add_argument(
        "--source-dir", type=Path, default=DEFAULT_RAG_SOURCE_DIR, help="RAG markdown source directory."
    )
    parser.add_argument("--index-path", type=Path, default=None, help="RAG source index path.")
    parser.add_argument("--max-chars", type=int, default=1000, help="Maximum characters per chunk.")
    parser.add_argument("--overlap-chars", type=int, default=150, help="Overlap characters for long section splits.")
    args = parser.parse_args()

    index_path = args.index_path or args.source_dir / INDEX_FILE_NAME
    chunks = build_rag_chunks_from_index(
        index_path=index_path,
        source_dir=args.source_dir,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )
    summary = summarize_rag_chunks(chunks, index_path=index_path, source_dir=args.source_dir).to_dict()
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print("RAG chunk preview")
    print(f"- enabled_documents: {summary['enabled_documents']}")
    print(f"- disabled_documents: {summary['disabled_documents']}")
    print(f"- total_chunks: {summary['total_chunks']}")
    print(f"- chunks_by_disease_code: {summary['chunks_by_disease_code']}")
    print("- sources:")
    for source in summary["sources"]:
        print(
            "  - "
            f"{source['source_id']} ({source['disease_code']}): "
            f"{source['chunk_count']} chunks, "
            f"length {source['min_content_length']}~{source['max_content_length']}, "
            f"status={source['review_status']}"
        )


if __name__ == "__main__":
    main()
