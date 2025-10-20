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


def get_model_device() -> str:
    return settings.ML_DEVICE


def get_confidence_threshold() -> float:
    return settings.ML_CONFIDENCE_THRESHOLD


def get_iou_threshold() -> float:
    return settings.ML_IOU_THRESHOLD
