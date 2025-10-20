Session maintenance
===================

Gunicorn timeouts can occur if fetching sessions from the database
blocks for too long. Typical reasons include a bloated ``django_session``
table or long running locks. To keep the table small and avoid
excessive queries we schedule the built‑in ``clearsessions`` command
once per day and store sessions in the cached DB backend.

The periodic task is defined via ``celery`` and runs every day at
03:00 UTC::

    'clear-sessions-daily': {
        'task': 'clear_expired_sessions',
        'schedule': crontab(hour=3, minute=0),
    }

The task itself calls ``django-admin clearsessions`` and removes expired
rows. Ensure ``CELERY_BEAT_SCHEDULE`` is loaded and that Redis is
configured for the ``default`` cache alias used by
``SESSION_ENGINE = 'django.contrib.sessions.backends.cached_db'``.
