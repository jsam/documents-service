bind = "0.0.0.0:8000"
workers = 4
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
preload_app = True

import logging
logger = logging.getLogger("gunicorn.error")

def worker_abort(worker):
    logger.critical("Gunicorn worker aborted due to timeout", extra={"pid": worker.pid})

