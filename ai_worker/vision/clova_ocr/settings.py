import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"


try:
    from dotenv import load_dotenv
except ImportError:
    # python-dotenv is optional for this PoC. Without it, load values by running:
    # set -a && source .env && set +a
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(ENV_PATH, override=False)


@dataclass(frozen=True)
class ClovaOCRSettings:
    api_url: str
    secret_key: str


def get_clova_ocr_settings() -> ClovaOCRSettings:
    api_url = os.getenv("CLOVA_OCR_API_URL")
    secret_key = os.getenv("CLOVA_OCR_SECRET_KEY")

    missing = []
    if not api_url:
        missing.append("CLOVA_OCR_API_URL")
    if not secret_key:
        missing.append("CLOVA_OCR_SECRET_KEY")

    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Missing CLOVA OCR environment variable(s): {names}")

    return ClovaOCRSettings(api_url=api_url, secret_key=secret_key)
