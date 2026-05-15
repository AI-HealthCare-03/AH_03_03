import os


def call_llm(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Set the environment variable before using real LLM mode.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is not installed. Install it before using real LLM mode.") from exc

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=prompt,
    )

    return response.output_text
