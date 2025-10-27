from functools import lru_cache
from io import BytesIO
from typing import BinaryIO

from django.conf import settings
from minio import Minio
from minio.error import S3Error


@lru_cache(maxsize=1)
def get_minio_client() -> Minio:
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_USE_SSL,
    )


def ensure_bucket_exists() -> None:
    client = get_minio_client()
    bucket_name = settings.MINIO_BUCKET_NAME
    
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
    except S3Error as e:
        raise RuntimeError(f'Failed to create bucket {bucket_name}: {e}') from e


def upload_file(
    bucket: str,
    object_name: str,
    file_data: BinaryIO | bytes,
    content_type: str = 'application/octet-stream',
) -> None:
    client = get_minio_client()
    
    if isinstance(file_data, bytes):
        file_data = BytesIO(file_data)
    
    file_data.seek(0, 2)
    file_size = file_data.tell()
    file_data.seek(0)
    
    try:
        client.put_object(
            bucket,
            object_name,
            file_data,
            length=file_size,
            content_type=content_type,
        )
    except S3Error as e:
        raise RuntimeError(f'Failed to upload {object_name} to {bucket}: {e}') from e


def download_file(bucket: str, object_name: str) -> bytes:
    client = get_minio_client()
    
    response = None
    try:
        response = client.get_object(bucket, object_name)
        data = response.read()
        return data
    except S3Error as e:
        raise RuntimeError(f'Failed to download {object_name} from {bucket}: {e}') from e
    finally:
        if response:
            response.close()
            response.release_conn()


def delete_file(bucket: str, object_name: str) -> None:
    client = get_minio_client()
    
    try:
        client.remove_object(bucket, object_name)
    except S3Error as e:
        raise RuntimeError(f'Failed to delete {object_name} from {bucket}: {e}') from e


def list_objects(bucket: str, prefix: str = '') -> list[str]:
    client = get_minio_client()
    
    try:
        objects = client.list_objects(bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects]
    except S3Error as e:
        raise RuntimeError(f'Failed to list objects in {bucket}: {e}') from e


def delete_folder(bucket: str, prefix: str) -> None:
    client = get_minio_client()
    
    try:
        objects = client.list_objects(bucket, prefix=prefix, recursive=True)
        object_names = [obj.object_name for obj in objects]
        
        for obj_name in object_names:
            client.remove_object(bucket, obj_name)
    except S3Error as e:
        raise RuntimeError(f'Failed to delete folder {prefix} from {bucket}: {e}') from e
