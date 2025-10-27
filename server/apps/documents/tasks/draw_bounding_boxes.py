import json
from io import BytesIO
from uuid import UUID, uuid4

from django.utils import timezone
from PIL import Image, ImageDraw

from server.apps.documents.models import DocumentJob, ProcessingStep
from server.apps.documents.utils.minio_client import download_file, upload_file
from server.apps.documents.utils.storage import (
    get_annotated_image_path,
    get_page_detections_path,
    get_page_image_path,
)

ELEMENT_COLORS = {
    'title': '#FF0000',
    'plain_text': '#00FF00',
    'abandon': '#808080',
    'figure': '#0000FF',
    'figure_caption': '#00FFFF',
    'table': '#FF00FF',
    'table_caption': '#FFFF00',
    'table_footnote': '#FFA500',
    'isolate_formula': '#800080',
    'formula_caption': '#FFC0CB',
}


def execute_draw_bounding_boxes(job_id: str, celery_task_id: str) -> dict:
    job = DocumentJob.objects.get(id=UUID(job_id))
    step = ProcessingStep.objects.get(job=job, step_name='DRAW_BOUNDING_BOXES')
    
    step.mark_in_progress(celery_task_id)
    job.current_step = 'DRAW_BOUNDING_BOXES'
    job.save(update_fields=['current_step'])
    
    try:
        total_pages = job.total_pages
        step.update_progress(0, total_pages)
        
        annotated_paths = []
        
        for page_num in range(1, total_pages + 1):
            image_path = get_page_image_path(job.id, page_num)
            image_bytes = download_file(job.minio_bucket, image_path)
            
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            img_width, img_height = image.size
            
            detections_path = get_page_detections_path(job.id, page_num)
            detections_bytes = download_file(job.minio_bucket, detections_path)
            detections_data = json.loads(detections_bytes.decode('utf-8'))
            
            draw = ImageDraw.Draw(image)
            
            for detection in detections_data['detections']:
                bbox = detection['bbox']
                element_type = detection['element_type']
                confidence = detection['confidence']
                
                element_id = str(uuid4())[:8]
                
                x1 = max(0, min(bbox['x1'] * img_width, img_width))
                y1 = max(0, min(bbox['y1'] * img_height, img_height))
                x2 = max(0, min(bbox['x2'] * img_width, img_width))
                y2 = max(0, min(bbox['y2'] * img_height, img_height))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                color = ELEMENT_COLORS.get(element_type, '#FFFFFF')
                
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                label = f'e={element_type},c={confidence:.2f},id={element_id}'
                draw.text((x1, max(0, y1 - 13)), label, fill=color)
            
            output_bytes = BytesIO()
            image.save(output_bytes, format='PNG')
            output_bytes.seek(0)
            
            annotated_path = get_annotated_image_path(job.id, page_num)
            
            upload_file(
                bucket=job.minio_bucket,
                object_name=annotated_path,
                file_data=output_bytes,
                content_type='image/png',
            )
            
            annotated_paths.append(annotated_path)
            
            step.update_progress(page_num, total_pages)
        
        step.status = ProcessingStep.STATUS_COMPLETED
        step.completed_at = timezone.now()
        step.result_data = {'annotated_paths': annotated_paths, 'total_pages': total_pages}
        step.save(update_fields=['status', 'completed_at', 'result_data'])
        
        return {'success': True, 'total_pages': total_pages, 'annotated_paths': annotated_paths}
    
    except Exception as e:
        step.status = ProcessingStep.STATUS_FAILED
        step.error_message = str(e)
        step.retry_count += 1
        step.save(update_fields=['status', 'error_message', 'retry_count'])
        
        job.mark_failed(str(e), 'DRAW_BOUNDING_BOXES')
        
        raise
