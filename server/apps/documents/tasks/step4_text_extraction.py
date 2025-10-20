import json
import logging
from io import BytesIO
from uuid import UUID

import easyocr
import numpy as np
from django.utils import timezone
from PIL import Image

from server.apps.documents.models import DocumentElement, DocumentJob, ProcessingStep
from server.apps.documents.utils.minio_client import download_file, upload_file
from server.apps.documents.utils.storage import (
    get_element_image_path,
    get_page_detections_path,
    get_page_image_path,
)

logger = logging.getLogger(__name__)

reader = easyocr.Reader(['en'], gpu=False)


def extract_text_from_region(image: Image.Image) -> str:
    try:
        logger.info(f'[OCR] Processing image size: {image.size}')
        result = reader.readtext(np.array(image))
        
        logger.info(f'[OCR] Found {len(result)} text regions')
        
        lines = []
        for detection in result:
            bbox, text, conf = detection
            lines.append((text, conf))
            logger.info(f'[OCR] Extracted: "{text}" (conf: {conf:.2f})')
        
        full_text = ' '.join(t for t, _ in lines) if lines else ''
        logger.info(f'[OCR] Final text: "{full_text}" ({len(lines)} lines)')
        return full_text.strip()
    except Exception as e:
        logger.error(f'[OCR] Error: {e}', exc_info=True)
        return ''


def execute_step4(job_id: str, celery_task_id: str) -> dict:
    job = DocumentJob.objects.get(id=UUID(job_id))
    step = ProcessingStep.objects.get(job=job, step_name='TEXT_EXTRACTION')
    
    step.mark_in_progress(celery_task_id)
    job.current_step = 'TEXT_EXTRACTION'
    job.save(update_fields=['current_step'])
    
    try:
        total_pages = job.total_pages
        step.update_progress(0, total_pages)
        
        total_elements = 0
        
        for page_num in range(1, total_pages + 1):
            image_path = get_page_image_path(job.id, page_num)
            image_bytes = download_file(job.minio_bucket, image_path)
            
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            img_width, img_height = image.size
            
            detections_path = get_page_detections_path(job.id, page_num)
            detections_bytes = download_file(job.minio_bucket, detections_path)
            detections_data = json.loads(detections_bytes.decode('utf-8'))
            
            detections = detections_data['detections']
            
            logger.info(f'[TEXT] Page {page_num}: {len(detections)} detections')
            
            detections.sort(key=lambda d: (d['bbox']['y1'], d['bbox']['x1']))
            
            for idx, detection in enumerate(detections):
                bbox = detection['bbox']
                element_type = detection['element_type']
                confidence = detection['confidence']
                
                x1 = int(bbox['x1'] * img_width)
                y1 = int(bbox['y1'] * img_height)
                x2 = int(bbox['x2'] * img_width)
                y2 = int(bbox['y2'] * img_height)
                
                logger.info(f'[TEXT] Element {idx}: type={element_type}, bbox=({x1},{y1},{x2},{y2})')
                
                cropped = image.crop((x1, y1, x2, y2))
                
                element_img_path = get_element_image_path(job.id, page_num, idx)
                
                cropped_bytes = BytesIO()
                cropped.save(cropped_bytes, format='PNG')
                cropped_bytes.seek(0)
                
                upload_file(
                    bucket=job.minio_bucket,
                    object_name=element_img_path,
                    file_data=cropped_bytes,
                    content_type='image/png',
                )
                
                extracted_text = None
                if element_type in ['title', 'plain_text', 'figure_caption', 'table_caption', 'table_footnote', 'formula_caption']:
                    logger.info(f'[TEXT] Running OCR on element {idx}')
                    extracted_text = extract_text_from_region(cropped)
                    logger.info(f'[TEXT] OCR result: "{extracted_text}"')
                else:
                    logger.info(f'[TEXT] Skipping OCR for element type: {element_type}')
                
                DocumentElement.objects.create(
                    job=job,
                    page_number=page_num,
                    bbox_x1=bbox['x1'],
                    bbox_y1=bbox['y1'],
                    bbox_x2=bbox['x2'],
                    bbox_y2=bbox['y2'],
                    element_type=element_type,
                    confidence=confidence,
                    extracted_text=extracted_text,
                    minio_image_key=element_img_path,
                    sequence=idx,
                )
                
                total_elements += 1
            
            step.update_progress(page_num, total_pages)
        
        step.status = ProcessingStep.STATUS_COMPLETED
        step.completed_at = timezone.now()
        step.result_data = {'total_elements': total_elements, 'total_pages': total_pages}
        step.save(update_fields=['status', 'completed_at', 'result_data'])
        
        return {'success': True, 'total_pages': total_pages, 'total_elements': total_elements}
    
    except Exception as e:
        step.status = ProcessingStep.STATUS_FAILED
        step.error_message = str(e)
        step.retry_count += 1
        step.save(update_fields=['status', 'error_message', 'retry_count'])
        
        job.mark_failed(str(e), 'TEXT_EXTRACTION')
        
        raise
