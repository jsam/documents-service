import io
from uuid import UUID

import fitz
from django.conf import settings
from django.utils import timezone
from PIL import Image

from server.apps.documents.models import DocumentJob, ProcessingStep
from server.apps.documents.utils.minio_client import download_file, upload_file
from server.apps.documents.utils.storage import get_page_image_path


def execute_step1(job_id: str, celery_task_id: str) -> dict:
    job = DocumentJob.objects.get(id=UUID(job_id))
    step = ProcessingStep.objects.get(job=job, step_name='PDF_TO_IMAGES')
    
    step.mark_in_progress(celery_task_id)
    job.current_step = 'PDF_TO_IMAGES'
    job.save(update_fields=['current_step'])
    
    try:
        pdf_bytes = download_file(job.minio_bucket, job.minio_key)
        
        pdf_document = fitz.open(stream=pdf_bytes, filetype='pdf')
        total_pages = len(pdf_document)
        
        job.total_pages = total_pages
        job.save(update_fields=['total_pages'])
        
        step.update_progress(0, total_pages)
        
        dpi = settings.PDF_TO_IMAGE_DPI
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        
        image_paths = []
        
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=matrix)
            
            img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
            
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            image_path = get_page_image_path(job.id, page_num + 1)
            
            upload_file(
                bucket=job.minio_bucket,
                object_name=image_path,
                file_data=img_bytes,
                content_type='image/png',
            )
            
            image_paths.append(image_path)
            
            step.update_progress(page_num + 1, total_pages)
        
        pdf_document.close()
        
        step.status = ProcessingStep.STATUS_COMPLETED
        step.completed_at = timezone.now()
        step.result_data = {'image_paths': image_paths, 'total_pages': total_pages}
        step.save(update_fields=['status', 'completed_at', 'result_data'])
        
        return {'success': True, 'total_pages': total_pages, 'image_paths': image_paths}
    
    except Exception as e:
        step.status = ProcessingStep.STATUS_FAILED
        step.error_message = str(e)
        step.retry_count += 1
        step.save(update_fields=['status', 'error_message', 'retry_count'])
        
        job.mark_failed(str(e), 'PDF_TO_IMAGES')
        
        raise
