import json
import os
from typing import Any

import firebase_admin
from firebase_admin import auth, credentials

FIREBASE_APP_NAME = "ai-health-firebase"


def _get_firebase_app() -> firebase_admin.App:
    try:
        return firebase_admin.get_app(FIREBASE_APP_NAME)
    except ValueError:
        pass

    project_id = os.getenv("FIREBASE_PROJECT_ID")
    credentials_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
    credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH")

    if credentials_json:
        try:
            cert_payload = json.loads(credentials_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError("FIREBASE_CREDENTIALS_JSON must be a valid JSON string.") from exc
        cred = credentials.Certificate(cert_payload)
    elif credentials_path:
        cred = credentials.Certificate(credentials_path)
    else:
        raise RuntimeError(
            "Firebase credentials are not configured. Set FIREBASE_CREDENTIALS_JSON or FIREBASE_CREDENTIALS_PATH."
        )

    options = {"projectId": project_id} if project_id else None
    return firebase_admin.initialize_app(cred, options=options, name=FIREBASE_APP_NAME)


def verify_firebase_id_token(id_token: str) -> dict[str, Any]:
    app = _get_firebase_app()
    return auth.verify_id_token(id_token, app=app)
