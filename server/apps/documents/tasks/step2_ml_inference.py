import json
import logging
from io import BytesIO
from uuid import UUID

import torch
from django.utils import timezone
from PIL import Image
from torchvision import transforms

from server.apps.documents.models import DocumentJob, ProcessingStep
from server.apps.documents.utils.minio_client import download_file, upload_file
from server.apps.documents.utils.model_loader import (
    get_confidence_threshold,
    get_iou_threshold,
    load_doclayout_model,
)
from server.apps.documents.utils.storage import (
    get_page_detections_path,
    get_page_image_path,
)

logger = logging.getLogger(__name__)

ELEMENT_CLASSES = [
    'title',
    'plain_text',
    'abandon',
    'figure',
    'figure_caption',
    'table',
    'table_caption',
    'table_footnote',
    'isolate_formula',
    'formula_caption',
]


def execute_step2(job_id: str, celery_task_id: str) -> dict:
    job = DocumentJob.objects.get(id=UUID(job_id))
    step = ProcessingStep.objects.get(job=job, step_name='ML_INFERENCE')
    
    step.mark_in_progress(celery_task_id)
    job.current_step = 'ML_INFERENCE'
    job.save(update_fields=['current_step'])
    
    try:
        logger.info(f'[ML] Loading model for job {job_id}')
        model = load_doclayout_model()
        conf_threshold = get_confidence_threshold()
        iou_threshold = get_iou_threshold()
        
        logger.info(f'[ML] Model loaded. Conf threshold: {conf_threshold}, IOU: {iou_threshold}')
        
        total_pages = job.total_pages
        step.update_progress(0, total_pages)
        
        all_detections = []
        
        for page_num in range(1, total_pages + 1):
            logger.info(f'[ML] Processing page {page_num}/{total_pages} for job {job_id}')
            
            image_path = get_page_image_path(job.id, page_num)
            image_bytes = download_file(job.minio_bucket, image_path)
            
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            original_width, original_height = image.size
            logger.info(f'[ML] Original image size: {original_width}x{original_height}')
            
            det_res = model.predict(
                image,
                imgsz=1280,
                conf=conf_threshold,
                device='cpu',
            )[0]
            
            boxes = det_res.__dict__['boxes'].xyxy
            classes = det_res.__dict__['boxes'].cls
            scores = det_res.__dict__['boxes'].conf
            
            logger.info(f'[ML] Raw detections: {len(boxes)} boxes')
            
            import torchvision
            if len(boxes) > 0:
                indices = torchvision.ops.nms(boxes=torch.Tensor(boxes), scores=torch.Tensor(scores), iou_threshold=iou_threshold)
                boxes = boxes[indices]
                scores = scores[indices]
                classes = classes[indices]
                
                logger.info(f'[ML] After NMS: {len(boxes)} boxes')
            
            page_detections = []
            for i in range(len(boxes)):
                box = boxes[i]
                score = float(scores[i])
                label_idx = int(classes[i])
                
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f'[ML] Invalid box: x1={x1}, y1={y1}, x2={x2}, y2={y2}')
                    continue
                
                x1 = max(0, min(x1, original_width))
                y1 = max(0, min(y1, original_height))
                x2 = max(0, min(x2, original_width))
                y2 = max(0, min(y2, original_height))
                
                norm_x1 = x1 / original_width
                norm_y1 = y1 / original_height
                norm_x2 = x2 / original_width
                norm_y2 = y2 / original_height
                
                element_type = ELEMENT_CLASSES[label_idx] if label_idx < len(ELEMENT_CLASSES) else 'unknown'
                
                detection = {
                    'bbox': {
                        'x1': norm_x1,
                        'y1': norm_y1,
                        'x2': norm_x2,
                        'y2': norm_y2,
                    },
                    'element_type': element_type,
                    'confidence': score,
                }
                page_detections.append(detection)
            
            logger.info(f'[ML] Page {page_num}: {len(page_detections)} final detections')
            
            detection_data = {
                'page_number': page_num,
                'image_width': original_width,
                'image_height': original_height,
                'detections': page_detections,
            }
            
            detections_path = get_page_detections_path(job.id, page_num)
            detections_json = json.dumps(detection_data, indent=2).encode('utf-8')
            
            upload_file(
                bucket=job.minio_bucket,
                object_name=detections_path,
                file_data=detections_json,
                content_type='application/json',
            )
            
            all_detections.append(detections_path)
            
            step.update_progress(page_num, total_pages)
        
        step.status = ProcessingStep.STATUS_COMPLETED
        step.completed_at = timezone.now()
        step.result_data = {'detection_paths': all_detections, 'total_pages': total_pages}
        step.save(update_fields=['status', 'completed_at', 'result_data'])
        
        return {'success': True, 'total_pages': total_pages, 'detection_paths': all_detections}
    
    except Exception as e:
        step.status = ProcessingStep.STATUS_FAILED
        step.error_message = str(e)
        step.retry_count += 1
        step.save(update_fields=['status', 'error_message', 'retry_count'])
        
        job.mark_failed(str(e), 'ML_INFERENCE')
        
        raise
