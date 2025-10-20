from celery.signals import worker_ready
import os


@worker_ready.connect
def preload_ml_model(sender, **kwargs):
    queues = [q.name for q in sender.task_consumer.queues]
    
    if 'ml_inference' in queues:
        try:
            from server.apps.documents.utils.model_loader import load_doclayout_model
            from django.conf import settings
            
            print(f'[ML Worker] Preloading model from {settings.ML_MODEL_PATH}...')
            print(f'[ML Worker] Working directory: {os.getcwd()}')
            print(f'[ML Worker] Model exists: {os.path.exists(settings.ML_MODEL_PATH)}')
            
            model = load_doclayout_model()
            print(f'[ML Worker] ✓ Model loaded successfully on {settings.ML_DEVICE}')
        except Exception as e:
            print(f'[ML Worker] ✗ Failed to preload model: {e}')
            import traceback
            traceback.print_exc()
            raise
