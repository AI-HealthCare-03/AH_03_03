import os
import sys
from typing import Any

from dotenv import load_dotenv

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
LANGFUSE_ENV_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
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


def call_llm(prompt: str, metadata: dict | None = None) -> str:
    client = build_openai_client()
    model = get_openai_model()
    response = create_openai_response(
        client=client,
        model=model,
        prompt=prompt,
        metadata=metadata,
    )

    return extract_response_text(response)


def call_llm_json(
    prompt: str,
    schema: dict[str, Any] | None = None,
    schema_name: str = "health_chatbot_response",
    metadata: dict | None = None,
) -> str:
    client = build_openai_client()
    model = get_openai_model()
    response = create_openai_response(
        client=client,
        model=model,
        prompt=prompt,
        metadata=metadata,
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


def create_openai_response(
    client,
    model: str,
    prompt: str,
    metadata: dict | None = None,
    **kwargs,
):
    langfuse = build_langfuse_client()
    if langfuse is None:
        return client.responses.create(
            model=model,
            input=prompt,
            **kwargs,
        )

    try:
        observation_context = langfuse.start_as_current_observation(
            name=build_langfuse_observation_name(metadata),
            as_type="generation",
            input=prompt,
            metadata=metadata,
            model=model,
        )
    except Exception:
        return client.responses.create(
            model=model,
            input=prompt,
            **kwargs,
        )

    try:
        generation = observation_context.__enter__()
    except Exception:
        return client.responses.create(
            model=model,
            input=prompt,
            **kwargs,
        )

    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            **kwargs,
        )
        output_text = extract_response_text(response)
        try:
            generation.update(output=output_text)
        except Exception:
            pass
        return response
    except Exception:
        exc_info = sys.exc_info()
        try:
            observation_context.__exit__(*exc_info)
        except Exception:
            pass
        raise
    finally:
        if sys.exc_info()[0] is None:
            try:
                observation_context.__exit__(None, None, None)
            except Exception:
                pass


def build_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Set the environment variable before using real LLM mode.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is not installed. Install it before using real LLM mode.") from exc

    return OpenAI(api_key=api_key)


def build_langfuse_client():
    if not has_langfuse_config():
        return None

    try:
        from langfuse import Langfuse
    except ImportError:
        return None

    try:
        return Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_BASE_URL"),
        )
    except Exception:
        return None


def has_langfuse_config() -> bool:
    return all(os.getenv(key) for key in LANGFUSE_ENV_KEYS)


def build_langfuse_observation_name(metadata: dict | None) -> str:
    if not metadata:
        return "llm_call"

    chatbot_type = metadata.get("chatbot_type")
    source = metadata.get("source")
    if chatbot_type and source:
        return f"{chatbot_type}.{source}"
    if source:
        return str(source)
    return "llm_call"


def get_openai_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def extract_response_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    raise RuntimeError("OpenAI response did not include output_text.")
