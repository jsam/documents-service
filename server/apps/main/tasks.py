from django.core.management import call_command
from celery import shared_task


@shared_task(name='clear_expired_sessions')
def clear_expired_sessions() -> None:
    """Delete expired sessions to prevent table bloat."""
    call_command('clearsessions')
