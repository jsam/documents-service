import logging
from uuid import UUID

from celery import chain, chord, group, shared_task
from django.utils import timezone

from server.apps.documents.models import DocumentJob, ProcessingStep

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def process_document_pipeline(self, job_id: str):
    job = DocumentJob.objects.get(id=UUID(job_id))
    
    job.status = DocumentJob.STATUS_PROCESSING
    job.processing_started_at = timezone.now()
    job.celery_task_id = self.request.id
    job.save(update_fields=['status', 'processing_started_at', 'celery_task_id'])
    
    ml_chain = chain(
        ml_inference.s(job_id),
        draw_bounding_boxes.s(job_id),
        text_extraction.s(job_id),
    )
    
    ocr_task = ocr_processing.si(job_id)
    
    logger.info(f'[PIPELINE] Building pipeline for job {job_id}')
    logger.info(f'[PIPELINE] ML chain: {ml_chain}')
    logger.info(f'[PIPELINE] OCR task: {ocr_task}')
    
    task_pipeline = chain(
        pdf_to_images.s(job_id),
        chord(
            group(ml_chain, ocr_task),
            assemble_graph.s(job_id),
        ),
    )
    
    logger.info(f'[PIPELINE] Final pipeline: {task_pipeline}')
    
    return task_pipeline.apply_async()


@shared_task(bind=True, max_retries=3)
def pdf_to_images(self, job_id: str):
    from server.apps.documents.tasks.pdf_to_images import execute_pdf_to_images
    return execute_pdf_to_images(job_id, self.request.id)


@shared_task(bind=True, max_retries=3, queue='ml_inference')
def ml_inference(self, previous_result, job_id: str):
    from server.apps.documents.tasks.ml_inference import execute_ml_inference
    return execute_ml_inference(job_id, self.request.id)


@shared_task(bind=True, max_retries=3)
def draw_bounding_boxes(self, previous_result, job_id: str):
    from server.apps.documents.tasks.draw_bounding_boxes import execute_draw_bounding_boxes
    return execute_draw_bounding_boxes(job_id, self.request.id)


@shared_task(bind=True, max_retries=3)
def text_extraction(self, previous_result, job_id: str):
    from server.apps.documents.tasks.text_extraction import execute_text_extraction
    return execute_text_extraction(job_id, self.request.id)


@shared_task(bind=True, max_retries=3)
def ocr_processing(self, job_id: str):
    logger.info(f'[PIPELINE] ocr_processing STARTED for job {job_id}')
    from server.apps.documents.tasks.ocr_processing import execute_ocr_processing
    result = execute_ocr_processing(job_id, self.request.id)
    logger.info(f'[PIPELINE] ocr_processing COMPLETED for job {job_id}')
    return result


@shared_task(bind=True, max_retries=3)
def assemble_graph(self, previous_result, job_id: str):
    from server.apps.documents.tasks.assemble_graph import execute_assemble_graph
    return execute_assemble_graph(job_id, self.request.id)
