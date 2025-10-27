import base64
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from uuid import UUID

from django.utils import timezone
from openai import OpenAI
from PIL import Image

from server.apps.documents.models import DocumentJob, ProcessingStep
from server.apps.documents.utils.minio_client import download_file, upload_file
from server.apps.documents.utils.storage import (
    get_page_image_path,
    get_ocr_path,
)

logger = logging.getLogger(__name__)

DEEPSEEK_OCR_SERVER_URL = os.environ.get(
    'DEEPSEEK_OCR_SERVER_URL',
    'http://host.docker.internal:8001',
)


def process_single_page(
    job_id: UUID,
    bucket: str,
    page_num: int,
    total_pages: int,
    client: OpenAI,
) -> tuple[int, str]:
    logger.info(
        f'[OCR] Processing page {page_num}/{total_pages} for job {job_id}',
    )

    page_image_path = get_page_image_path(job_id, page_num)
    image_bytes = download_file(bucket, page_image_path)

    image = Image.open(BytesIO(image_bytes)).convert('RGB')

    buffered = BytesIO()
    image.save(buffered, format='PNG')
    image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    logger.info(f'[OCR] Sending page {page_num} to DeepSeek OCR API')

    response = client.chat.completions.create(
        model='deepseek-ocr',
        messages=[
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{image_b64}',
                        },
                    },
                    {
                        'type': 'text',
                        'text': '<|grounding|>Convert the document to markdown.',
                    },
                ],
            },
        ],
        max_tokens=4096,
    )

    markdown_content = response.choices[0].message.content
    logger.info(
        f'[OCR] Page {page_num} completed: {len(markdown_content)} chars',
    )

    return page_num, markdown_content


def execute_ocr_processing(job_id: str, celery_task_id: str) -> dict:
    job = DocumentJob.objects.get(id=UUID(job_id))
    step = ProcessingStep.objects.get(job=job, step_name='OCR_PROCESSING')

    step.mark_in_progress(celery_task_id)
    job.current_step = 'OCR_PROCESSING'
    job.save(update_fields=['current_step'])

    try:
        total_pages = job.total_pages
        step.update_progress(0, total_pages)

        client = OpenAI(
            base_url=f'{DEEPSEEK_OCR_SERVER_URL}/v1',
            api_key='dummy-key',
        )

        page_results = {}

        max_workers = min(total_pages, 4)
        logger.info(
            f'[OCR] Processing {total_pages} pages in parallel '
            f'with {max_workers} workers',
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_page,
                    job.id,
                    job.minio_bucket,
                    page_num,
                    total_pages,
                    client,
                ): page_num
                for page_num in range(1, total_pages + 1)
            }

            completed = 0
            for future in as_completed(futures):
                page_num, markdown_content = future.result()
                page_results[page_num] = markdown_content
                completed += 1
                step.update_progress(completed, total_pages)

        sorted_pages = sorted(page_results.items())
        full_markdown = '\n\n---\n\n'.join(
            f'# Page {page_num}\n\n{content}'
            for page_num, content in sorted_pages
        )

        markdown_path = f'{get_ocr_path(job.id)}/full_document.md'
        upload_file(
            bucket=job.minio_bucket,
            object_name=markdown_path,
            file_data=full_markdown.encode('utf-8'),
            content_type='text/markdown',
        )

        logger.info(
            f'[OCR] Assembled markdown document: {len(full_markdown)} chars',
        )

        step.status = ProcessingStep.STATUS_COMPLETED
        step.completed_at = timezone.now()
        step.result_data = {
            'markdown_path': markdown_path,
            'total_pages': total_pages,
            'total_chars': len(full_markdown),
        }
        step.save(update_fields=['status', 'completed_at', 'result_data'])

        return {
            'success': True,
            'total_pages': total_pages,
            'markdown_path': markdown_path,
        }

    except Exception as e:
        logger.error(f'[OCR] Error: {e}', exc_info=True)
        step.status = ProcessingStep.STATUS_FAILED
        step.error_message = str(e)
        step.retry_count += 1
        step.save(update_fields=['status', 'error_message', 'retry_count'])

        job.mark_failed(str(e), 'OCR_PROCESSING')

        raise
