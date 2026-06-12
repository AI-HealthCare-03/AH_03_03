import sys
from typing import Any

from dotenv import load_dotenv

from app.core import config
from app.core.providers import has_langfuse_config, has_openai_config

DEFAULT_OPENAI_MODEL = "gpt-4o"
OPENAI_CONNECT_TIMEOUT_SECONDS = 3.0
OPENAI_READ_TIMEOUT_SECONDS = 15.0
OPENAI_WRITE_TIMEOUT_SECONDS = 15.0
OPENAI_POOL_TIMEOUT_SECONDS = 3.0
OPENAI_TOTAL_TIMEOUT_SECONDS = 20.0
OPENAI_MAX_RETRIES = 2
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


def build_openai_client():
    if not has_openai_config(config):
        raise RuntimeError("OPENAI_API_KEY is not set. Set the environment variable before using real LLM mode.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is not installed. Install it before using real LLM mode.") from exc

    try:
        import httpx

        timeout = httpx.Timeout(
            OPENAI_TOTAL_TIMEOUT_SECONDS,
            connect=OPENAI_CONNECT_TIMEOUT_SECONDS,
            read=OPENAI_READ_TIMEOUT_SECONDS,
            write=OPENAI_WRITE_TIMEOUT_SECONDS,
            pool=OPENAI_POOL_TIMEOUT_SECONDS,
        )
    except ImportError:
        timeout = OPENAI_TOTAL_TIMEOUT_SECONDS

    return OpenAI(api_key=config.OPENAI_API_KEY, timeout=timeout, max_retries=OPENAI_MAX_RETRIES)


def build_langfuse_client():
    if not is_langfuse_enabled():
        return None

    try:
        from langfuse import Langfuse
    except ImportError:
        return None

    try:
        return Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_BASE_URL,
        )
    except Exception:
        return None


def is_langfuse_enabled() -> bool:
    return config.LANGFUSE_ENABLED and has_langfuse_config(config)


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
    return config.OPENAI_MODEL or DEFAULT_OPENAI_MODEL


def extract_response_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    raise RuntimeError("OpenAI response did not include output_text.")
