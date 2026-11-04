"""Cloudflare R2 storage primitives for future document workflows."""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any, BinaryIO

import boto3
import structlog
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError

from app.config.settings import Settings, get_settings


LOGGER = structlog.get_logger("vyaparsathi.ai.r2_storage")


class R2StorageError(Exception):
    """Base exception for R2 storage failures safe to surface to API layers."""


class R2ConfigurationError(R2StorageError):
    """Raised when required R2 configuration is absent or invalid."""


class R2ObjectNotFoundError(R2StorageError):
    """Raised when the requested object does not exist."""


class R2BucketNotFoundError(R2StorageError):
    """Raised when the configured R2 bucket does not exist."""


class R2AccessDeniedError(R2StorageError):
    """Raised for credentials and permission failures."""


class R2ConnectionError(R2StorageError):
    """Raised when R2 cannot be reached."""


class R2OperationError(R2StorageError):
    """Raised for an R2 operation that failed for an unclassified reason."""


def generate_file_id() -> str:
    return f"file_{uuid.uuid4().hex}"


def _validate_segment(value: str, name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError(f"{name} must be a single, non-empty path segment")
    if any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{name} contains an invalid control character")
    return normalized


def build_object_key(
    *, store_id: str, user_id: str, filename: str, file_id: str | None = None
) -> tuple[str, str]:
    """Return ``(file_id, object_key)`` using a backend-owned key convention."""
    safe_store_id = _validate_segment(store_id, "store_id")
    safe_user_id = _validate_segment(user_id, "user_id")
    safe_file_id = _validate_segment(file_id or generate_file_id(), "file_id")
    if not filename or "/" in filename or "\\" in filename:
        raise ValueError("filename must be a single, non-empty file name")
    safe_filename = Path(filename).name
    if safe_filename != filename or safe_filename in {".", ".."}:
        raise ValueError("filename must be a single, non-empty file name")
    return safe_file_id, (
        f"stores/{safe_store_id}/users/{safe_user_id}/files/{safe_file_id}/{safe_filename}"
    )


build_document_object_key = build_object_key


class R2StorageService:
    """S3-compatible Cloudflare R2 client with async-friendly operations."""

    def __init__(self, settings: Settings | None = None, client: BaseClient | None = None) -> None:
        self._settings = settings or get_settings()
        self._bucket_name = self._require_configuration()
        self._client = client or self._create_client()

    def _require_configuration(self) -> str:
        r2_account_id = (
            self._settings.effective_cloudflare_r2_account_id
            or os.environ.get("CLOUDFLARE_R2_ACCOUNT_ID")
            or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        )
        required = {
            "CLOUDFLARE_R2_ACCOUNT_ID": r2_account_id,
            "CLOUDFLARE_R2_ENDPOINT": self._settings.cloudflare_r2_endpoint,
            "CLOUDFLARE_R2_ACCESS_KEY_ID": self._settings.cloudflare_r2_access_key_id,
            "CLOUDFLARE_R2_SECRET_ACCESS_KEY": self._settings.cloudflare_r2_secret_access_key,
            "CLOUDFLARE_R2_BUCKET_NAME": self._settings.cloudflare_r2_bucket_name,
        }
        missing = [name for name, value in required.items() if not value or not value.strip()]
        if missing:
            raise R2ConfigurationError("Cloudflare R2 is not configured. Missing: " + ", ".join(missing))
        return required["CLOUDFLARE_R2_BUCKET_NAME"].strip()

    def _create_client(self) -> BaseClient:
        return boto3.client(
            "s3",
            endpoint_url=self._settings.cloudflare_r2_endpoint.rstrip("/"),
            region_name="auto",
            aws_access_key_id=self._settings.cloudflare_r2_access_key_id,
            aws_secret_access_key=self._settings.cloudflare_r2_secret_access_key,
            config=Config(signature_version="s3v4"),
        )

    @property
    def bucket_name(self) -> str:
        return self._bucket_name

    @staticmethod
    def build_object_key(**kwargs: Any) -> tuple[str, str]:
        return build_object_key(**kwargs)

    async def upload_file(self, source_path: str | Path, object_key: str, content_type: str | None = None) -> None:
        kwargs = {"ExtraArgs": {"ContentType": content_type}} if content_type else {}
        await self._call("upload_file", self._client.upload_file, str(source_path), self._bucket_name, object_key, **kwargs)

    async def upload_bytes(
        self,
        data: bytes | BinaryIO,
        object_key: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        parameters: dict[str, Any] = {"Bucket": self._bucket_name, "Key": object_key, "Body": data}
        if content_type:
            parameters["ContentType"] = content_type
        if metadata:
            parameters["Metadata"] = metadata
        return await self._call("upload_bytes", self._client.put_object, **parameters)

    async def download_file(self, object_key: str, destination_path: str | Path) -> None:
        await self._call("download_file", self._client.download_file, self._bucket_name, object_key, str(destination_path))

    async def download_bytes(self, object_key: str) -> bytes:
        def fetch() -> bytes:
            body = self._client.get_object(Bucket=self._bucket_name, Key=object_key)["Body"]
            try:
                return body.read()
            finally:
                body.close()

        return await self._call("download_bytes", fetch)

    async def delete_file(self, object_key: str) -> None:
        await self._call("delete_file", self._client.delete_object, Bucket=self._bucket_name, Key=object_key)

    async def file_exists(self, object_key: str) -> bool:
        try:
            await self._call("file_exists", self._client.head_object, Bucket=self._bucket_name, Key=object_key)
        except R2ObjectNotFoundError:
            return False
        return True

    async def get_file_metadata(self, object_key: str) -> dict[str, Any]:
        return await self._call("get_file_metadata", self._client.head_object, Bucket=self._bucket_name, Key=object_key)

    async def generate_presigned_download_url(self, object_key: str, expires_in: int = 3600) -> str:
        if expires_in <= 0:
            raise ValueError("expires_in must be greater than zero")
        return await self._call(
            "generate_presigned_download_url",
            self._client.generate_presigned_url,
            "get_object",
            Params={"Bucket": self._bucket_name, "Key": object_key},
            ExpiresIn=expires_in,
        )

    async def health_check(self) -> bool:
        await self._call("health_check", self._client.head_bucket, Bucket=self._bucket_name)
        return True

    async def check_connection(self) -> bool:
        return await self.health_check()

    async def _call(self, operation: str, function: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await asyncio.to_thread(function, *args, **kwargs)
        except ClientError as exc:
            self._raise_client_error(operation, exc)
        except EndpointConnectionError as exc:
            LOGGER.warning("r2_connection_failed", operation=operation)
            raise R2ConnectionError("Unable to connect to Cloudflare R2") from exc
        except BotoCoreError as exc:
            LOGGER.warning("r2_operation_failed", operation=operation, error_type=type(exc).__name__)
            raise R2OperationError("Cloudflare R2 operation failed") from exc

    @staticmethod
    def _raise_client_error(operation: str, exc: ClientError) -> None:
        code = str(exc.response.get("Error", {}).get("Code", "Unknown"))
        LOGGER.warning("r2_client_error", operation=operation, error_code=code)
        if code in {"404", "NoSuchKey", "NotFound"}:
            raise R2ObjectNotFoundError("R2 object was not found") from exc
        if code == "NoSuchBucket":
            raise R2BucketNotFoundError("Configured R2 bucket was not found") from exc
        if code in {"401", "403", "AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"}:
            raise R2AccessDeniedError("R2 credentials are invalid or access is denied") from exc
        raise R2OperationError("Cloudflare R2 operation failed") from exc


__all__ = [
    "R2StorageService", "R2StorageError", "R2ConfigurationError", "R2ObjectNotFoundError",
    "R2BucketNotFoundError", "R2AccessDeniedError", "R2ConnectionError", "R2OperationError",
    "generate_file_id", "build_object_key", "build_document_object_key",
]