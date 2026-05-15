import os
from typing import Any

from dotenv import load_dotenv

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
REWRITE_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {
            "type": "string",
            "description": "룰엔진 답변의 의미를 유지해 사용자 친화적으로 다시 쓴 답변",
        },
    },
    "required": ["answer"],
}


load_dotenv()


def call_llm(prompt: str) -> str:
    client = build_openai_client()
    model = get_openai_model()
    response = client.responses.create(
        model=model,
        input=prompt,
    )

    return extract_response_text(response)


def call_llm_json(
    prompt: str,
    schema: dict[str, Any] | None = None,
    schema_name: str = "health_chatbot_response",
) -> str:
    client = build_openai_client()
    model = get_openai_model()
    response = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema or REWRITE_RESPONSE_SCHEMA,
                "strict": True,
            }
        },
    )

    return extract_response_text(response)


def build_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Set the environment variable before using real LLM mode.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is not installed. Install it before using real LLM mode.") from exc

    return OpenAI(api_key=api_key)


def get_openai_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def extract_response_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    raise RuntimeError("OpenAI response did not include output_text.")
