from __future__ import annotations

import pytest

from app.services import storage


class FakeBody:
    def __init__(self, content: bytes):
        self.content = content

    def read(self) -> bytes:
        return self.content


class FakeS3Client:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_calls: list[dict[str, object]] = []
        self.deleted: list[tuple[str, str]] = []

    def put_object(self, **kwargs):
        self.put_calls.append(kwargs)
        self.objects[(str(kwargs["Bucket"]), str(kwargs["Key"]))] = kwargs["Body"]

    def get_object(self, *, Bucket: str, Key: str):
        return {"Body": FakeBody(self.objects[(Bucket, Key)])}

    def delete_object(self, *, Bucket: str, Key: str):
        self.deleted.append((Bucket, Key))
        self.objects.pop((Bucket, Key), None)

    def head_object(self, *, Bucket: str, Key: str):
        if (Bucket, Key) not in self.objects:
            exc = RuntimeError("not found")
            exc.response = {"Error": {"Code": "404"}}
            raise exc
        return {}

    def generate_presigned_url(self, operation_name: str, *, Params: dict[str, str], ExpiresIn: int):
        return f"https://example.test/{Params['Bucket']}/{Params['Key']}?op={operation_name}&expires={ExpiresIn}"


def test_local_storage_save_read_exists_delete(tmp_path) -> None:
    service = storage.LocalStorageService(tmp_path)

    key = service.save_bytes(b"hello", "exams/1/source.pdf", "application/pdf")

    assert key == "exams/1/source.pdf"
    assert service.exists(key) is True
    assert service.read_bytes(key) == b"hello"
    assert service.get_presigned_url(key) is None

    service.delete(key)

    assert service.exists(key) is False


def test_storage_key_prefix_normalization() -> None:
    assert storage.normalize_storage_key("/exams/1/source.pdf", " private/uploads/ ") == (
        "private/uploads/exams/1/source.pdf"
    )
    assert storage.normalize_storage_key("private/uploads/exams/1/source.pdf", "private/uploads") == (
        "private/uploads/exams/1/source.pdf"
    )

    with pytest.raises(ValueError):
        storage.normalize_storage_key("../secret.txt")


def test_get_storage_service_uses_local_backend_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(storage.config, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(storage.config, "LOCAL_STORAGE_ROOT", str(tmp_path))

    service = storage.get_storage_service()
    key = service.save_bytes(b"local", "uploads/a.txt")

    assert service.read_bytes(key) == b"local"


def test_s3_storage_save_read_exists_delete_and_presigned_url() -> None:
    client = FakeS3Client()
    service = storage.S3StorageService(
        bucket_name="private-bucket",
        region="ap-northeast-2",
        prefix="ai-health/dev",
        client=client,
    )

    key = service.save_bytes(b"image", "diet/1/source.jpg", "image/jpeg")

    assert key == "ai-health/dev/diet/1/source.jpg"
    assert client.put_calls[0]["ContentType"] == "image/jpeg"
    assert service.exists("diet/1/source.jpg") is True
    assert service.read_bytes("diet/1/source.jpg") == b"image"
    assert service.get_presigned_url("diet/1/source.jpg", expires_in=60) == (
        "https://example.test/private-bucket/ai-health/dev/diet/1/source.jpg?op=get_object&expires=60"
    )

    service.delete("diet/1/source.jpg")

    assert service.exists("diet/1/source.jpg") is False


def test_s3_storage_requires_bucket_name() -> None:
    with pytest.raises(ValueError):
        storage.S3StorageService(bucket_name="", region="ap-northeast-2")
