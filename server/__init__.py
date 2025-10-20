# This will make sure the app is always imported when
# Django starts so that shared_task will use this app.
from .celery import app as celery_app
from . import celery_signals

__all__ = ('celery_app',)
