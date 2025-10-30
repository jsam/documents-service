from datetime import datetime
from typing import Any
from uuid import UUID

from ninja import Schema


class UploadResponse(Schema):
    job_id: UUID
    status: str
    message: str


class ProcessingStepStatus(Schema):
    step_name: str
    step_order: int
    status: str
    started_at: datetime | None
    completed_at: datetime | None
    progress_current: int
    progress_total: int | None
    progress_percentage: float
    error_message: str | None
    retry_count: int


class JobStatusResponse(Schema):
    job_id: UUID
    status: str
    original_filename: str
    file_size: int
    created_at: datetime
    updated_at: datetime
    current_step: str | None
    total_pages: int | None
    processing_started_at: datetime | None
    processing_completed_at: datetime | None
    error_message: str | None
    error_step: str | None
    retry_count: int
    steps: list[ProcessingStepStatus]


class DocumentElement(Schema):
    id: int
    page_number: int
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    element_type: str
    confidence: float
    extracted_text: str | None
    table_html: str | None
    table_data: list[list[str]] | None
    sequence: int
    minio_image_key: str | None


class JobResultsResponse(Schema):
    job_id: UUID
    status: str
    original_filename: str
    total_pages: int | None
    processing_completed_at: datetime | None
    document_graph: dict[str, Any] | None
    elements: list[DocumentElement]
    error_message: str | None


class ErrorResponse(Schema):
    detail: str
