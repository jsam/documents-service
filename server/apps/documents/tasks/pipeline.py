from uuid import UUID

from celery import chain, shared_task
from django.utils import timezone

from server.apps.documents.models import DocumentJob, ProcessingStep


@shared_task(bind=True, max_retries=3)
def process_document_pipeline(self, job_id: str):
    job = DocumentJob.objects.get(id=UUID(job_id))
    
    job.status = DocumentJob.STATUS_PROCESSING
    job.processing_started_at = timezone.now()
    job.celery_task_id = self.request.id
    job.save(update_fields=['status', 'processing_started_at', 'celery_task_id'])
    
    task_chain = chain(
        step1_pdf_to_images.s(job_id),
        step2_ml_inference.s(job_id),
        step3_draw_bounding_boxes.s(job_id),
        step4_text_extraction.s(job_id),
        step6_points_ocr.s(job_id),
        step5_assemble_graph.s(job_id),
    )
    
    return task_chain.apply_async()


@shared_task(bind=True, max_retries=3)
def step1_pdf_to_images(self, job_id: str):
    from server.apps.documents.tasks.step1_pdf_to_images import execute_step1
    return execute_step1(job_id, self.request.id)


@shared_task(bind=True, max_retries=3, queue='ml_inference')
def step2_ml_inference(self, previous_result, job_id: str):
    from server.apps.documents.tasks.step2_ml_inference import execute_step2
    return execute_step2(job_id, self.request.id)


@shared_task(bind=True, max_retries=3)
def step3_draw_bounding_boxes(self, previous_result, job_id: str):
    from server.apps.documents.tasks.step3_draw_bounding_boxes import execute_step3
    return execute_step3(job_id, self.request.id)


@shared_task(bind=True, max_retries=3)
def step4_text_extraction(self, previous_result, job_id: str):
    from server.apps.documents.tasks.step4_text_extraction import execute_step4
    return execute_step4(job_id, self.request.id)


@shared_task(bind=True, max_retries=3, queue='points_ocr')
def step6_points_ocr(self, previous_result, job_id: str):
    from server.apps.documents.tasks.step6_points_ocr import execute_step6
    return execute_step6(job_id, self.request.id)


@shared_task(bind=True, max_retries=3)
def step5_assemble_graph(self, previous_result, job_id: str):
    from server.apps.documents.tasks.step5_assemble_graph import execute_step5
    return execute_step5(job_id, self.request.id)
