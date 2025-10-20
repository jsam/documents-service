# Gunicorn configuration file
# https://docs.gunicorn.org/en/stable/configure.html#configuration-file
# https://docs.gunicorn.org/en/stable/settings.html

import multiprocessing
import logging

bind = '0.0.0.0:8000'
# Use a fixed number of workers to make timeouts more predictable.
workers = 4
worker_class = 'sync'
worker_connections = 1000
timeout = 120
keepalive = 2
# Maintain reasonable request limits to prevent memory leaks.
max_requests = 1000
max_requests_jitter = 100

accesslog = '-'
chdir = '/code'
worker_tmp_dir = '/dev/shm'  # noqa: S108

logger = logging.getLogger('gunicorn.error')


def worker_abort(worker):
    """Log when a worker is aborted due to timeout."""
    logger.critical('Gunicorn worker aborted due to timeout', extra={'pid': worker.pid})

