from ai_runtime.llm import rag_generator


def test_rag_response_extracts_answer_from_json_code_fence() -> None:
    raw_response = """
근거 수준이 제한적이므로 참고용입니다.
```json
{
  "answer": "혈당 관리는 식사와 활동 습관을 함께 살펴보면 좋습니다.",
  "intent": "internal",
  "source": "rag_llm",
  "is_safe": true
}
```
""".strip()

    answer = rag_generator.extract_answer_from_rag_response(raw_response)

    assert answer == "혈당 관리는 식사와 활동 습관을 함께 살펴보면 좋습니다."
    assert '"answer":' not in answer
    assert '"intent":' not in answer
    assert '"source":' not in answer
    assert '"is_safe":' not in answer
    assert "```" not in answer


def test_rag_response_extracts_answer_from_prefixed_unclosed_json_code_fence() -> None:
    raw_response = """
근거 수준이 제한적이므로 단정하지 않고 참고용으로 안내드립니다. ```json
{
  "answer": "고혈압이 있을 때는 국물과 짠 소스를 줄이는 식습관부터 참고해 보세요.",
  "intent": "main_health_rag_guidance",
  "source": "rag_llm",
  "caution_message": "이 정보는 진단이 아니며, 정확한 진단과 치료는 의료진 상담이 필요합니다.",
  "is_safe": true
}
""".strip()

    answer = rag_generator.extract_answer_from_rag_response(raw_response)

    assert answer == "고혈압이 있을 때는 국물과 짠 소스를 줄이는 식습관부터 참고해 보세요."
    for term in ['"answer":', '"intent":', '"source":', '"is_safe":', '"caution_message":', "```"]:
        assert term not in answer


def test_rag_response_extracts_answer_from_embedded_json_object() -> None:
    raw_response = (
        '참고용 안내입니다. {"answer": "탄수화물은 끼니마다 양을 일정하게 살펴보면 좋습니다.", '
        '"intent": "internal", "source": "rag_llm", "is_safe": true} 추가 설명'
    )

    answer = rag_generator.extract_answer_from_rag_response(raw_response)

    assert answer == "탄수화물은 끼니마다 양을 일정하게 살펴보면 좋습니다."
    assert '"answer":' not in answer
    assert '"source":' not in answer


def test_rag_response_sanitizes_nested_answer_json() -> None:
    json_response = """
{
  "answer": "```json\\n{\\"answer\\": \\"신장 관련 수치가 걱정될 때는 식사일지를 남기고 의료진과 상담해 보세요.\\", \\"source\\": \\"rag_llm\\"}\\n```"
}
""".strip()

    answer = rag_generator.extract_answer_from_rag_response(json_response)

    assert answer == "신장 관련 수치가 걱정될 때는 식사일지를 남기고 의료진과 상담해 보세요."
    assert rag_generator.is_public_rag_answer(answer) is True
    assert '"answer":' not in answer
    assert "```" not in answer


def test_rag_generator_uses_json_llm_and_returns_public_answer(monkeypatch) -> None:
    captured = {}

    def fake_call_llm_json(prompt, schema=None, schema_name=None, metadata=None):
        captured.update(
            {
                "prompt": prompt,
                "schema": schema,
                "schema_name": schema_name,
                "metadata": metadata,
            }
        )
        return """
```json
{"answer": "혈압 관리가 필요한 경우 국물과 짠 소스를 줄이는 방식부터 참고해 보세요."}
```
""".strip()

    monkeypatch.setattr(rag_generator, "call_llm_json", fake_call_llm_json)

    output = rag_generator.generate_main_health_rag_response(
        user_message="혈압 관리는 어떻게 하나요?",
        retrieved_context="고혈압 식생활 관리는 저염 식습관을 참고할 수 있습니다.",
        context_sources=[
            {
                "title": "고혈압 식생활",
                "source_org": "질병관리청 국가건강정보포털",
                "source_url": "https://health.kdca.go.kr/hypertension",
            }
        ],
        use_real_llm=True,
    )

    assert output.source == "rag_llm"
    assert output.answer.startswith("혈압 관리가 필요한 경우")
    assert "진단이 아니" in output.answer
    assert "의료진 상담" in output.answer
    assert '"answer":' not in output.answer
    assert '"intent":' not in output.answer
    assert '"source":' not in output.answer
    assert '"is_safe":' not in output.answer
    assert "```" not in output.answer
    assert captured["schema_name"] == "main_health_rag_answer"
    assert captured["schema"]["required"] == ["answer"]
