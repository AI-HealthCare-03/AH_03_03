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


def record_langfuse_event(
    *,
    name: str,
    input_payload: dict[str, Any] | str | None = None,
    output_payload: dict[str, Any] | str | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Record a lightweight non-generation event when Langfuse is configured.

    This helper is intentionally best-effort. It never raises into service code
    and it does not expose secret values in metadata.
    """
    langfuse = build_langfuse_client()
    if langfuse is None:
        return False

    try:
        observation_context = langfuse.start_as_current_observation(
            name=name,
            as_type="span",
            input=input_payload,
            metadata=metadata,
        )
        observation = observation_context.__enter__()
    except Exception:
        return False

    try:
        try:
            observation.update(output=output_payload)
        except Exception:
            pass
        return True
    finally:
        try:
            observation_context.__exit__(None, None, None)
        except Exception:
            pass


def record_langfuse_generation(
    *,
    name: str,
    model: str,
    input_payload: dict[str, Any] | str | None = None,
    output_payload: dict[str, Any] | str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """GPT Vision 등 실제 모델 호출을 generation 타입으로 기록합니다.

    generation 타입으로 기록해야 Langfuse가 토큰 수를 인식하고
    모델 단가 설정에 따라 비용을 자동 계산합니다.
    """
    langfuse = build_langfuse_client()
    if langfuse is None:
        return False

    # Langfuse SDK v3: usage_details (Dict[str, int]) 로 토큰 수 전달
    usage_details: dict[str, int] = {}
    if input_tokens is not None:
        usage_details["input"] = input_tokens
    if output_tokens is not None:
        usage_details["output"] = output_tokens
    if input_tokens and output_tokens:
        usage_details["total"] = input_tokens + output_tokens

    try:
        observation_context = langfuse.start_as_current_observation(
            name=name,
            as_type="generation",
            model=model,
            input=input_payload,
            metadata=metadata,
            usage_details=usage_details if usage_details else None,
        )
        observation = observation_context.__enter__()
    except Exception:
        return False

    try:
        try:
            observation.update(output=output_payload)
        except Exception:
            pass
        return True
    finally:
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
    if not is_langfuse_enabled():
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
            timeout=10,        # 기본값보다 짧게: 10초 안에 안 되면 포기
            flush_interval=2,  # 2초마다 전송 시도
        )
    except Exception:
        return None


def has_langfuse_config() -> bool:
    return all(os.getenv(key) for key in LANGFUSE_ENV_KEYS)


def is_langfuse_enabled() -> bool:
    enabled_value = os.getenv("LANGFUSE_ENABLED")
    if enabled_value is not None and enabled_value.strip().lower() in {"0", "false", "no", "off"}:
        return False
    return has_langfuse_config()


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
