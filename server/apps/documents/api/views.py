from uuid import UUID

from django.conf import settings
from django.shortcuts import get_object_or_404
from ninja import File, Router
from ninja.files import UploadedFile

from server.apps.documents.api.schemas import (
    ErrorResponse,
    JobResultsResponse,
    JobStatusResponse,
    UploadResponse,
)
from server.apps.documents.models import DocumentJob, ProcessingStep
from server.apps.documents.utils.minio_client import upload_file
from server.apps.documents.utils.storage import get_original_pdf_path

router = Router()


@router.post('/upload', response={200: UploadResponse, 400: ErrorResponse})
def upload_document(request, file: UploadedFile = File(...)):
    if not file.name.lower().endswith('.pdf'):
        return 400, {'detail': 'Only PDF files are supported'}
    
    file_size = 0
    file_content = b''
    for chunk in file.chunks():
        file_content += chunk
        file_size += len(chunk)
    
    max_size_bytes = settings.MAX_PDF_SIZE_MB * 1024 * 1024
    if file_size > max_size_bytes:
        return 400, {'detail': f'File size exceeds maximum allowed size of {settings.MAX_PDF_SIZE_MB}MB'}
    
    job = DocumentJob.objects.create(
        original_filename=file.name,
        file_size=file_size,
        minio_bucket=settings.MINIO_BUCKET_NAME,
        minio_key=get_original_pdf_path(None, file.name),
        status=DocumentJob.STATUS_PENDING,
    )
    
    job.minio_key = get_original_pdf_path(job.id, file.name)
    job.save(update_fields=['minio_key'])
    
    try:
        upload_file(
            bucket=job.minio_bucket,
            object_name=job.minio_key,
            file_data=file_content,
            content_type='application/pdf',
        )
    except RuntimeError as e:
        job.mark_failed(str(e), 'UPLOAD')
        return 400, {'detail': f'Failed to upload file to storage: {e}'}
    
    for idx, step_name in enumerate(['UPLOAD', 'PDF_TO_IMAGES', 'ML_INFERENCE', 'DRAW_BOUNDING_BOXES', 'TEXT_EXTRACTION', 'OCR_PROCESSING', 'ASSEMBLE_GRAPH']):
        ProcessingStep.objects.create(
            job=job,
            step_name=step_name,
            step_order=idx,
            status=ProcessingStep.STATUS_COMPLETED if step_name == 'UPLOAD' else ProcessingStep.STATUS_PENDING,
        )
    
    from server.apps.documents.tasks import process_document_pipeline
    process_document_pipeline.delay(str(job.id))
    
    return 200, {
        'job_id': job.id,
        'status': job.status,
        'message': 'Document uploaded successfully and queued for processing',
    }


@router.get('/{job_id}/status', response={200: JobStatusResponse, 404: ErrorResponse})
def get_job_status(request, job_id: UUID):
    job = get_object_or_404(DocumentJob, id=job_id)
    
    steps = job.steps.all().order_by('step_order')
    
    step_statuses = [
        {
            'step_name': step.step_name,
            'step_order': step.step_order,
            'status': step.status,
            'started_at': step.started_at,
            'completed_at': step.completed_at,
            'progress_current': step.progress_current,
            'progress_total': step.progress_total,
            'progress_percentage': step.progress_percentage,
            'error_message': step.error_message,
            'retry_count': step.retry_count,
        }
        for step in steps
    ]
    
    return 200, {
        'job_id': job.id,
        'status': job.status,
        'original_filename': job.original_filename,
        'file_size': job.file_size,
        'created_at': job.created_at,
        'updated_at': job.updated_at,
        'current_step': job.current_step,
        'total_pages': job.total_pages,
        'processing_started_at': job.processing_started_at,
        'processing_completed_at': job.processing_completed_at,
        'error_message': job.error_message,
        'error_step': job.error_step,
        'retry_count': job.retry_count,
        'steps': step_statuses,
    }


@router.get('/{job_id}/results', response={200: JobResultsResponse, 404: ErrorResponse, 400: ErrorResponse})
def get_job_results(request, job_id: UUID):
    job = get_object_or_404(DocumentJob, id=job_id)
    
    if job.status not in [DocumentJob.STATUS_COMPLETED, DocumentJob.STATUS_FAILED]:
        return 400, {'detail': 'Job is still processing. Check status endpoint for progress.'}
    
    elements = job.elements.all().order_by('page_number', 'sequence')
    
    element_list = [
        {
            'id': elem.id,
            'page_number': elem.page_number,
            'bbox_x1': elem.bbox_x1,
            'bbox_y1': elem.bbox_y1,
            'bbox_x2': elem.bbox_x2,
            'bbox_y2': elem.bbox_y2,
            'element_type': elem.element_type,
            'confidence': elem.confidence,
            'extracted_text': elem.extracted_text,
            'sequence': elem.sequence,
            'minio_image_key': elem.minio_image_key,
        }
        for elem in elements
    ]
    
    return 200, {
        'job_id': job.id,
        'status': job.status,
        'original_filename': job.original_filename,
        'total_pages': job.total_pages,
        'processing_completed_at': job.processing_completed_at,
        'document_graph': job.document_graph,
        'elements': element_list,
        'error_message': job.error_message,
    }
