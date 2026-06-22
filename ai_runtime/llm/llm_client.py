import logging
import re
import sys
from time import perf_counter
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
logger = logging.getLogger(__name__)


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
        return _create_openai_response_with_observability(
            client=client,
            model=model,
            prompt=prompt,
            metadata=metadata,
            **kwargs,
        )

    try:
        observation_context = langfuse.start_as_current_observation(
            name=build_langfuse_observation_name(metadata),
            as_type="generation",
            input=redact_llm_prompt(prompt),
            metadata=metadata,
            model=model,
        )
    except Exception:
        return _create_openai_response_with_observability(
            client=client,
            model=model,
            prompt=prompt,
            metadata=metadata,
            **kwargs,
        )

    try:
        generation = observation_context.__enter__()
    except Exception:
        return _create_openai_response_with_observability(
            client=client,
            model=model,
            prompt=prompt,
            metadata=metadata,
            **kwargs,
        )

    try:
        response = _create_openai_response_with_observability(
            client=client,
            model=model,
            prompt=prompt,
            metadata=metadata,
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


def redact_llm_prompt(prompt: str | None) -> str:
    text = str(prompt or "")
    if not text:
        return text

    text = re.sub(r"[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}", "[email]", text)
    text = re.sub(r"\b(?:\+?82[-.\s]?)?0?1[016789][-\s.]?\d{3,4}[-\s.]?\d{4}\b", "[phone]", text)
    text = re.sub(r"\b\d{6}[-\s]?[1-4]\d{6}\b", "[rrn]", text)
    text = re.sub(
        r"((?:이름|성명|name)\s*[:=]\s*)([가-힣A-Za-z][가-힣A-Za-z\s]{1,30})",
        r"\1[name]",
        text,
        flags=re.IGNORECASE,
    )

    health_terms = (
        "혈압",
        "수축기",
        "이완기",
        "SBP",
        "DBP",
        "공복혈당",
        "혈당",
        "glucose",
        "HbA1c",
        "당화혈색소",
        "총콜레스테롤",
        "콜레스테롤",
        "LDL",
        "HDL",
        "TG",
        "중성지방",
        "triglyceride",
        "BMI",
        "체질량지수",
        "키",
        "신장",
        "몸무게",
        "체중",
        "허리둘레",
        "waist",
    )
    health_pattern = (
        r"((?:"
        + "|".join(re.escape(term) for term in health_terms)
        + r")\s*[:=]?\s*)(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?\s*(?:mg/dl|mmhg|%|kg|cm)?)"
    )
    text = re.sub(health_pattern, r"\1[health_value]", text, flags=re.IGNORECASE)

    medication_pattern = (
        r"((?:복용약|복약|처방약|약물|약 이름|medication|medicine)\s*[:=]\s*)"
        r"([^\n,.;]{1,80})"
    )
    text = re.sub(medication_pattern, r"\1[medication]", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b([A-Za-z가-힣]{2,30})\s+\d+(?:\.\d+)?\s*(?:mg|㎎|정|tablet|tab)\b",
        r"[medication] [dose]",
        text,
        flags=re.IGNORECASE,
    )
    return text


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


def _create_openai_response_with_observability(
    *,
    client,
    model: str,
    prompt: str,
    metadata: dict | None,
    **kwargs,
):
    started_at = perf_counter()
    request_kwargs = _with_default_temperature(kwargs)
    try:
        response = _create_response_with_temperature_fallback(
            client=client,
            model=model,
            prompt=prompt,
            request_kwargs=request_kwargs,
        )
    except Exception as exc:
        _log_openai_runtime_call(
            metadata=metadata,
            model=model,
            latency_ms=round((perf_counter() - started_at) * 1000, 2),
            success=False,
            error_type=type(exc).__name__,
            usage=None,
        )
        raise

    _log_openai_runtime_call(
        metadata=metadata,
        model=model,
        latency_ms=round((perf_counter() - started_at) * 1000, 2),
        success=True,
        error_type=None,
        usage=_extract_token_usage(response),
    )
    return response


def _with_default_temperature(kwargs: dict[str, Any]) -> dict[str, Any]:
    request_kwargs = dict(kwargs)
    if "temperature" not in request_kwargs:
        request_kwargs["temperature"] = config.OPENAI_TEMPERATURE
    return request_kwargs


def _create_response_with_temperature_fallback(
    *,
    client,
    model: str,
    prompt: str,
    request_kwargs: dict[str, Any],
):
    try:
        return client.responses.create(
            model=model,
            input=prompt,
            **request_kwargs,
        )
    except Exception as exc:
        if "temperature" not in request_kwargs or not _is_temperature_unsupported_error(exc):
            raise

        fallback_kwargs = dict(request_kwargs)
        fallback_kwargs.pop("temperature", None)
        logger.info(
            "OpenAI Responses API temperature unsupported; retrying without temperature",
            extra={
                "llm_runtime": {
                    "llm_provider": "openai",
                    "llm_model": model,
                    "fallback_reason": type(exc).__name__,
                    "temperature_retry_without_parameter": True,
                }
            },
        )
        return client.responses.create(
            model=model,
            input=prompt,
            **fallback_kwargs,
        )


def _is_temperature_unsupported_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "temperature" not in message:
        return isinstance(exc, TypeError)
    if isinstance(exc, TypeError):
        return True
    return any(
        marker in message
        for marker in (
            "unsupported",
            "not supported",
            "unknown parameter",
            "unrecognized",
            "unexpected keyword",
            "invalid parameter",
        )
    )


def _log_openai_runtime_call(
    *,
    metadata: dict | None,
    model: str,
    latency_ms: float,
    success: bool,
    error_type: str | None,
    usage: dict[str, int | None] | None,
) -> bool:
    if not _should_log_llm_runtime():
        return False

    metadata = metadata or {}
    usage = usage or {}
    llm_call_path = _llm_call_path(metadata)
    payload = {
        "llm_provider": "openai",
        "llm_model": model,
        "llm_call_path": llm_call_path,
        "latency_ms": latency_ms,
        "success": success,
        "error_type": error_type,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    logger.info(
        "llm_runtime llm_provider=%s llm_model=%s llm_call_path=%s latency_ms=%s "
        "success=%s error_type=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
        payload["llm_provider"],
        payload["llm_model"],
        payload["llm_call_path"],
        payload["latency_ms"],
        payload["success"],
        payload["error_type"],
        payload["prompt_tokens"],
        payload["completion_tokens"],
        payload["total_tokens"],
        extra={"llm_runtime": payload},
    )
    return True


def _should_log_llm_runtime() -> bool:
    env = getattr(config, "ENV", "")
    env_value = getattr(env, "value", str(env))
    return str(env_value).lower() in {"local", "dev"}


def _llm_call_path(metadata: dict[str, Any]) -> str:
    source = str(metadata.get("source") or "llm_call")
    chatbot_type = metadata.get("chatbot_type")
    if chatbot_type:
        return f"{chatbot_type}.{source}"
    return source


def _extract_token_usage(response) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    if isinstance(usage, dict):
        prompt_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
        completion_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    else:
        prompt_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


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
