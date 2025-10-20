import json
import logging
import os
import tempfile
from io import BytesIO
from pathlib import Path
from uuid import UUID

import torch
from django.utils import timezone
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLImageProcessor

from server.apps.documents.models import DocumentJob, ProcessingStep
from server.apps.documents.utils.minio_client import download_file, upload_file
from server.apps.documents.utils.storage import (
    get_annotated_image_path,
    get_points_result_path,
)

logger = logging.getLogger(__name__)

USE_GPU = os.environ.get('POINTS_USE_GPU', 'false').lower() == 'true'

model = None
tokenizer = None
image_processor = None


def load_points_model():
    global model, tokenizer, image_processor
    
    if model is not None:
        return
    
    MODEL_PATH = 'tencent/POINTS-Reader'
    logger.info(f'[POINTS] Loading model: {MODEL_PATH}')
    logger.info(f'[POINTS] Use GPU: {USE_GPU}')
    logger.info('[POINTS] Note: Model files should be pre-patched at worker startup')
    
    if not USE_GPU:
        logger.error('[POINTS] CRITICAL: CPU-only inference is not fully supported')
        logger.error('[POINTS] The Qwen2VL vision encoder has known issues on CPU')
        logger.error('[POINTS] Please set POINTS_USE_GPU=true and ensure CUDA is available')
        raise RuntimeError(
            'POINTS OCR requires GPU. Set POINTS_USE_GPU=true in production with CUDA support.'
        )
    
    if not torch.cuda.is_available():
        error_msg = 'GPU requested but CUDA is not available'
        logger.error(f'[POINTS] {error_msg}')
        raise RuntimeError(error_msg)
    
    logger.info('[POINTS] Loading model on GPU with device_map=auto')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    image_processor = Qwen2VLImageProcessor.from_pretrained(MODEL_PATH)
    
    logger.info('[POINTS] Model loaded successfully')


def execute_step6(job_id: str, celery_task_id: str) -> dict:
    job = DocumentJob.objects.get(id=UUID(job_id))
    step = ProcessingStep.objects.get(job=job, step_name='POINTS_OCR')
    
    step.mark_in_progress(celery_task_id)
    job.current_step = 'POINTS_OCR'
    job.save(update_fields=['current_step'])
    
    try:
        load_points_model()
        
        total_pages = job.total_pages
        step.update_progress(0, total_pages)
        
        result_paths = []
        prompt = "Perform OCR on the image precisely."
        
        for page_num in range(1, total_pages + 1):
            logger.info(f'[POINTS] Processing page {page_num}/{total_pages} for job {job_id}')
            
            annotated_path = get_annotated_image_path(job.id, page_num)
            image_bytes = download_file(job.minio_bucket, annotated_path)
            
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            logger.info(f'[POINTS] Image size: {image.size}')
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name
                image.save(temp_path, format='PNG')
            
            try:
                content = [
                    dict(type='image', image=temp_path),
                    dict(type='text', text=prompt)
                ]
                messages = [
                    {
                        'role': 'user',
                        'content': content
                    }
                ]
                
                generation_config = {
                    'max_new_tokens': 2048,
                    'repetition_penalty': 1.05,
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 20,
                    'do_sample': True
                }
                
                logger.info('[POINTS] Running inference')
                response = model.chat(
                    messages,
                    tokenizer,
                    image_processor,
                    generation_config
                )
                
                logger.info(f'[POINTS] Response length: {len(response)} chars')
                
                result_data = {
                    'page_number': page_num,
                    'prompt': prompt,
                    'response': response
                }
                
                result_path = get_points_result_path(job.id, page_num)
                result_json = json.dumps(result_data, indent=2).encode('utf-8')
                
                upload_file(
                    bucket=job.minio_bucket,
                    object_name=result_path,
                    file_data=result_json,
                    content_type='application/json',
                )
                
                result_paths.append(result_path)
            finally:
                Path(temp_path).unlink(missing_ok=True)
            
            step.update_progress(page_num, total_pages)
        
        step.status = ProcessingStep.STATUS_COMPLETED
        step.completed_at = timezone.now()
        step.result_data = {'result_paths': result_paths, 'total_pages': total_pages}
        step.save(update_fields=['status', 'completed_at', 'result_data'])
        
        return {'success': True, 'total_pages': total_pages, 'result_paths': result_paths}
    
    except Exception as e:
        logger.error(f'[POINTS] Error: {e}', exc_info=True)
        step.status = ProcessingStep.STATUS_FAILED
        step.error_message = str(e)
        step.retry_count += 1
        step.save(update_fields=['status', 'error_message', 'retry_count'])
        
        job.mark_failed(str(e), 'POINTS_OCR')
        
        raise
