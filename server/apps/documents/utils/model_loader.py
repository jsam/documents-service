from functools import lru_cache
from pathlib import Path

import torch
from django.conf import settings


@lru_cache(maxsize=1)
def load_doclayout_model():
    from doclayout_yolo import YOLOv10
    
    model_path = Path(settings.ML_MODEL_PATH)
    
    if not model_path.exists():
        raise FileNotFoundError(f'Model file not found at {model_path}')
    
    try:
        model = YOLOv10(str(model_path))
        return model
    except Exception as e:
        raise RuntimeError(f'Failed to load model from {model_path}: {e}') from e


@lru_cache(maxsize=1)
def load_olmocr_model():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    
    try:
        print(f'[OLMOCR] Loading model: {settings.OLMOCR_MODEL_NAME}')
        print(f'[OLMOCR] Device: {settings.OLMOCR_DEVICE}')
        
        processor = AutoProcessor.from_pretrained(
            'Qwen/Qwen2.5-VL-7B-Instruct',
            trust_remote_code=True,
        )
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            settings.OLMOCR_MODEL_NAME,
            torch_dtype=torch.bfloat16 if settings.OLMOCR_DEVICE == 'cuda' else torch.float32,
            trust_remote_code=True,
        ).eval()
        
        model.to(settings.OLMOCR_DEVICE)
        
        print(f'[OLMOCR] Model loaded successfully')
        return model, processor
    except Exception as e:
        raise RuntimeError(f'Failed to load olmOCR model {settings.OLMOCR_MODEL_NAME}: {e}') from e


def get_model_device() -> str:
    return settings.ML_DEVICE


def get_confidence_threshold() -> float:
    return settings.ML_CONFIDENCE_THRESHOLD


def get_iou_threshold() -> float:
    return settings.ML_IOU_THRESHOLD
