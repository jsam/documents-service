import json
import logging
import tempfile
from io import BytesIO
from pathlib import Path
from uuid import UUID

import torch
from django.utils import timezone
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from server.apps.documents.models import DocumentJob, ProcessingStep
from server.apps.documents.utils.minio_client import download_file, upload_file
from server.apps.documents.utils.storage import (
    get_ocr_result_path,
    get_page_image_path,
)

logger = logging.getLogger(__name__)

import os

USE_GPU = os.environ.get('DEEPSEEK_USE_GPU', 'true').lower() == 'true'

model = None
tokenizer = None


def load_deepseek_model():
    global model, tokenizer
    
    if model is not None:
        return
    
    MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'
    logger.info(f'[DEEPSEEK] Loading model: {MODEL_PATH}')
    logger.info(f'[DEEPSEEK] Use GPU: {USE_GPU}')
    
    if not USE_GPU:
        logger.error('[DEEPSEEK] CRITICAL: CPU-only inference is not supported')
        logger.error('[DEEPSEEK] Please set DEEPSEEK_USE_GPU=true and ensure CUDA is available')
        raise RuntimeError(
            'DeepSeek OCR requires GPU. Set DEEPSEEK_USE_GPU=true in production with CUDA support.'
        )
    
    if not torch.cuda.is_available():
        logger.error('[DEEPSEEK] GPU requested but CUDA is not available')
        logger.error('[DEEPSEEK] Please ensure the container has GPU access configured in docker-compose')
        raise RuntimeError(
            'DeepSeek OCR requires CUDA GPU. Container must have nvidia GPU access configured.'
        )
    
    logger.info('[DEEPSEEK] Loading model on GPU with flash_attention_2')
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        device_map='cuda'
    )
    model = model.eval()
    
    logger.info('[DEEPSEEK] Model loaded successfully')


def execute_step6(job_id: str, celery_task_id: str) -> dict:
    job = DocumentJob.objects.get(id=UUID(job_id))
    step = ProcessingStep.objects.get(job=job, step_name='OCR_PROCESSING')
    
    step.mark_in_progress(celery_task_id)
    job.current_step = 'OCR_PROCESSING'
    job.save(update_fields=['current_step'])
    
    try:
        load_deepseek_model()
        
        total_pages = job.total_pages
        step.update_progress(0, total_pages)
        
        result_paths = []
        
        for page_num in range(1, total_pages + 1):
            logger.info(f'[DEEPSEEK] Processing page {page_num}/{total_pages} for job {job_id}')
            
            page_image_path = get_page_image_path(job.id, page_num)
            image_bytes = download_file(job.minio_bucket, page_image_path)
            
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            logger.info(f'[DEEPSEEK] Image size: {image.size}')
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name
                image.save(temp_path, format='PNG')
            
            try:
                logger.info('[DEEPSEEK] Running OCR inference')
                
                prompt = '<image>\n<|grounding|>Convert the document to markdown. '
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    response = model.infer(
                        tokenizer,
                        prompt=prompt,
                        image_file=temp_path,
                        output_path=tmpdir,
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        save_results=False,
                        test_compress=False,
                        eval_mode=True
                    )
                
                logger.info(f'[DEEPSEEK] Response length: {len(str(response))} chars')
                
                result_data = {
                    'page_number': page_num,
                    'prompt': prompt,
                    'response': str(response)
                }
                
                result_path = get_ocr_result_path(job.id, page_num)
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
        logger.error(f'[DEEPSEEK] Error: {e}', exc_info=True)
        step.status = ProcessingStep.STATUS_FAILED
        step.error_message = str(e)
        step.retry_count += 1
        step.save(update_fields=['status', 'error_message', 'retry_count'])
        
        job.mark_failed(str(e), 'DEEPSEEK_OCR')
        
        raise
