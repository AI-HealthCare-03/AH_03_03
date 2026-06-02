from pathlib import Path, PurePosixPath
from typing import Protocol

from app.core import config


class StorageService(Protocol):
    def save_bytes(self, content: bytes, key: str, content_type: str | None = None) -> str: ...

    def read_bytes(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...

    def exists(self, key: str) -> bool: ...

    def get_presigned_url(self, key: str, expires_in: int = 3600) -> str | None: ...


def normalize_storage_key(key: str, prefix: str | None = None) -> str:
    normalized_key = _clean_key_part(key)
    normalized_prefix = _clean_key_part(prefix or "", allow_empty=True)
    if not normalized_prefix:
        return normalized_key
    if normalized_key == normalized_prefix or normalized_key.startswith(f"{normalized_prefix}/"):
        return normalized_key
    return f"{normalized_prefix}/{normalized_key}"


class LocalStorageService:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def save_bytes(self, content: bytes, key: str, content_type: str | None = None) -> str:
        _ = content_type
        normalized_key = normalize_storage_key(key)
        path = self._path_for_key(normalized_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return normalized_key

    def read_bytes(self, key: str) -> bytes:
        return self._path_for_key(key).read_bytes()

    def delete(self, key: str) -> None:
        path = self._path_for_key(key)
        if path.exists():
            path.unlink()

    def exists(self, key: str) -> bool:
        return self._path_for_key(key).exists()

    def get_presigned_url(self, key: str, expires_in: int = 3600) -> str | None:
        _ = key, expires_in
        return None

    def _path_for_key(self, key: str) -> Path:
        normalized_key = normalize_storage_key(key)
        root = self.root.resolve()
        path = (root / normalized_key).resolve()
        if not path.is_relative_to(root):
            raise ValueError("storage key escapes local storage root")
        return path


class S3StorageService:
    def __init__(
        self,
        *,
        bucket_name: str,
        region: str,
        prefix: str = "",
        client: object | None = None,
    ):
        if not bucket_name:
            raise ValueError("S3 bucket name is required")
        self.bucket_name = bucket_name
        self.region = region
        self.prefix = _clean_key_part(prefix, allow_empty=True)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def save_bytes(self, content: bytes, key: str, content_type: str | None = None) -> str:
        object_key = normalize_storage_key(key, self.prefix)
        kwargs: dict[str, object] = {
            "Bucket": self.bucket_name,
            "Key": object_key,
            "Body": content,
        }
        if content_type:
            kwargs["ContentType"] = content_type
        self.client.put_object(**kwargs)
        return object_key

    def read_bytes(self, key: str) -> bytes:
        object_key = normalize_storage_key(key, self.prefix)
        response = self.client.get_object(Bucket=self.bucket_name, Key=object_key)
        return response["Body"].read()

    def delete(self, key: str) -> None:
        object_key = normalize_storage_key(key, self.prefix)
        self.client.delete_object(Bucket=self.bucket_name, Key=object_key)

    def exists(self, key: str) -> bool:
        object_key = normalize_storage_key(key, self.prefix)
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=object_key)
        except Exception as exc:
            error_code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
        return True

    def get_presigned_url(self, key: str, expires_in: int = 3600) -> str | None:
        object_key = normalize_storage_key(key, self.prefix)
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": object_key},
            ExpiresIn=expires_in,
        )


def get_storage_service() -> StorageService:
    backend = config.STORAGE_BACKEND.strip().lower()
    if backend == "local":
        return LocalStorageService(config.LOCAL_STORAGE_ROOT)
    if backend == "s3":
        return S3StorageService(
            bucket_name=config.S3_BUCKET_NAME or "",
            region=config.S3_REGION,
            prefix=config.S3_PREFIX,
        )
    raise ValueError(f"Unsupported storage backend: {config.STORAGE_BACKEND}")


def _clean_key_part(value: str, *, allow_empty: bool = False) -> str:
    stripped = str(value or "").strip().replace("\\", "/").strip("/")
    if not stripped:
        if allow_empty:
            return ""
        raise ValueError("storage key must not be empty")

    path = PurePosixPath(stripped)
    parts = path.parts
    if path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("storage key contains unsafe path segments")
    return "/".join(parts)
