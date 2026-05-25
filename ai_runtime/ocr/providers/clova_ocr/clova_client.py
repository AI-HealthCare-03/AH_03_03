import json
import time
import uuid
from pathlib import Path
from typing import Any

import requests

try:
    from .settings import get_clova_ocr_settings
except ImportError:  # pragma: no cover - direct script execution fallback
    from settings import get_clova_ocr_settings

SUPPORTED_OCR_EXTENSIONS = {"jpg", "jpeg", "png", "pdf"}


class ClovaOCRClient:
    """Minimal CLOVA OCR V2 multipart client.

    Note: this PoC uses `requests`. If the runtime environment does not include
    it, install/add `requests` in a separate dependency-management PR.
    """

    def __init__(self, api_url: str | None = None, secret_key: str | None = None) -> None:
        settings = get_clova_ocr_settings()
        self.api_url = api_url or settings.api_url
        self.secret_key = secret_key or settings.secret_key

    def request_ocr(self, image_path: str) -> dict[str, Any]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"CLOVA OCR image file not found: {path}")

        file_format = path.suffix.lstrip(".").lower()
        if file_format not in SUPPORTED_OCR_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_OCR_EXTENSIONS))
            raise ValueError(f"Unsupported CLOVA OCR file extension: {file_format}. Supported: {supported}")

        payload = {
            "version": "V2",
            "requestId": str(uuid.uuid4()),
            "timestamp": int(time.time() * 1000),
            "images": [
                {
                    "format": file_format,
                    "name": path.stem,
                }
            ],
        }
        headers = {"X-OCR-SECRET": self.secret_key}

        with path.open("rb") as image_file:
            files = {
                "message": (None, json.dumps(payload), "application/json"),
                "file": (path.name, image_file, "application/octet-stream"),
            }
            response = requests.post(self.api_url, headers=headers, files=files, timeout=30)

        response.raise_for_status()
        return response.json()
