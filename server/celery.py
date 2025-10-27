import os
from celery import Celery, shared_task
from celery.signals import worker_process_init
from datetime import datetime
from kombu import Queue

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

# Replace 'myproject' with your project name.
app = Celery('server')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Define task queues
app.conf.task_queues = (
    Queue('celery', routing_key='celery'),
    Queue('ml_inference', routing_key='ml_inference'),
    Queue('ocr_processing', routing_key='ocr_processing'),
)

app.conf.task_default_queue = 'celery'
app.conf.task_default_exchange = 'tasks'
app.conf.task_default_routing_key = 'celery'

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')

@shared_task(name="heartbeat_task")
def heartbeat():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Celery:heartbeat_task - Current time: {current_time}")
    return 1


@worker_process_init.connect
def init_worker_process(**kwargs):
    import logging
    import sys
    logger = logging.getLogger(__name__)
    
    worker_id = kwargs.get('sender')
    logger.info(f'[CELERY] Worker process init signal received for {worker_id}')
    
    cmd_line = ' '.join(sys.argv)
    logger.info(f'[CELERY] Command line: {cmd_line}')
    
    if 'ocr_processing' in cmd_line or '-Q ocr_processing' in cmd_line:
        logger.info('[CELERY] Detected ocr_processing queue worker')
        logger.info('[CELERY] DeepSeek-OCR model will load on first task execution')
    else:
        logger.info('[CELERY] This worker does not handle ocr_processing queue')
