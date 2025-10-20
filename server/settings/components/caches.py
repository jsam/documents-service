# Caching
# https://docs.djangoproject.com/en/4.2/topics/cache/
from server.settings.components import config

CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': config('REDIS_URL'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            # Redis connections can block indefinitely which results in
            # Gunicorn worker timeouts.  We use small socket timeouts so
            # failed cache calls fail fast instead of hanging.
            'SOCKET_CONNECT_TIMEOUT': 5,
            'SOCKET_TIMEOUT': 5,
            # Swallow connection errors to keep the application working even
            # when Redis is unavailable.
            'IGNORE_EXCEPTIONS': True,
        }
    }
}

# django-axes
# https://django-axes.readthedocs.io/en/latest/4_configuration.html#configuring-caches

AXES_CACHE = 'default'
