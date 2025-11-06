"""
Django settings for server project.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/topics/settings/

For the full list of settings and their config, see
https://docs.djangoproject.com/en/4.2/ref/settings/
"""

from django.utils.translation import gettext_lazy as _

from server.settings.components import BASE_DIR, config

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

SECRET_KEY = config('DJANGO_SECRET_KEY')

# Application definition:

INSTALLED_APPS: tuple[str, ...] = (
    # Your apps go here:
    'server.apps.main',
    'server.apps.documents',
    # Default django apps:
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # django-admin:
    'django.contrib.admin',
    'django.contrib.admindocs',
    # Security:
    'axes',
    # A lightweight healthcheck view is implemented in ``server.apps.main``.
    # The ``django-health-check`` package is not used anymore to avoid
    # unnecessary database queries during periodic container health checks.
    # Celery:
    'django_celery_results',
    'django_celery_beat',
    # django-ninja:
    'ninja'
)

MIDDLEWARE: tuple[str, ...] = (
    # Logging:
    'server.settings.components.logging.LoggingContextVarsMiddleware',
    # Content Security Policy:
    'csp.middleware.CSPMiddleware',
    # Django:
    'django.middleware.security.SecurityMiddleware',
    # django-permissions-policy
    'django_permissions_policy.PermissionsPolicyMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.locale.LocaleMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    # Axes:
    'axes.middleware.AxesMiddleware',
)

ROOT_URLCONF = 'server.urls'

WSGI_APPLICATION = 'server.wsgi.application'


# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': config('POSTGRES_DB'),
        'USER': config('POSTGRES_USER'),
        'PASSWORD': config('POSTGRES_PASSWORD'),
        'HOST': config('DJANGO_DATABASE_HOST'),
        'PORT': config('DJANGO_DATABASE_PORT', cast=int),
        'CONN_MAX_AGE': config('CONN_MAX_AGE', cast=int, default=60),
        'OPTIONS': {
            # Disable the per-process pool so every request gets a fresh socket
            # when CONN_MAX_AGE says so.
            'pool': False,

            # Fail fast if the server is unreachable
            'connect_timeout': 30,

            # Let Postgres kill any query that runs >15 s
            'options': '-c statement_timeout=15000ms',

            # Keep the TCP session alive so firewalls/NAT don’t prune it
            'keepalives': 1,
            'keepalives_idle':     30,   # first probe after 30 s idle
            'keepalives_interval': 10,   # resend every 10 s
            'keepalives_count':     5,   # give up after 5 failed probes
        },
    },
}

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field
DEFAULT_AUTO_FIELD = 'django.db.models.AutoField'


# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

USE_I18N = True

LANGUAGES = (
    ('en', _('English')),
    ('ru', _('Russian')),
)

LOCALE_PATHS = ('locale/',)

USE_TZ = True
TIME_ZONE = 'UTC'


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = '/static/'

STATICFILES_FINDERS = (
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
)


# Templates
# https://docs.djangoproject.com/en/4.2/ref/templates/api

TEMPLATES = [
    {
        'APP_DIRS': True,
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            # Contains plain text templates, like `robots.txt`:
            BASE_DIR.joinpath('server', 'common', 'django', 'templates'),
        ],
        'OPTIONS': {
            'context_processors': [
                # Default template context processors:
                'django.contrib.auth.context_processors.auth',
                'django.template.context_processors.debug',
                'django.template.context_processors.i18n',
                'django.template.context_processors.media',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.request',
            ],
        },
    }
]


# Media files
# Media root dir is commonly changed in production
# (see development.py and production.py).
# https://docs.djangoproject.com/en/4.2/topics/files/

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR.joinpath('media')


# Django authentication system
# https://docs.djangoproject.com/en/4.2/topics/auth/

AUTHENTICATION_BACKENDS = (
    'axes.backends.AxesBackend',
    'django.contrib.auth.backends.ModelBackend',
)

PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.Argon2PasswordHasher',
    'django.contrib.auth.hashers.PBKDF2PasswordHasher',
    'django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher',
    'django.contrib.auth.hashers.BCryptSHA256PasswordHasher',
]


# Security
# https://docs.djangoproject.com/en/4.2/topics/security/

SESSION_COOKIE_HTTPONLY = True
CSRF_COOKIE_HTTPONLY = True

SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'

# Load allowed origins from the environment. Falls back to the current
# domain if no explicit value is provided.
_csrf_origins = config(
    'CSRF_TRUSTED_ORIGINS',
    default=f"https://{config('DOMAIN_NAME')}",
)
CSRF_TRUSTED_ORIGINS = [origin.strip() for origin in _csrf_origins.split(',') if origin.strip()]

SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True

X_FRAME_OPTIONS = 'DENY'

# https://docs.djangoproject.com/en/3.0/ref/middleware/#referrer-policy
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Referrer-Policy
SECURE_REFERRER_POLICY = 'same-origin'

# https://github.com/adamchainz/django-permissions-policy#setting
PERMISSIONS_POLICY: dict[str, str | list[str]] = {}


# Timeouts
# https://docs.djangoproject.com/en/4.2/ref/settings/#std:setting-EMAIL_TIMEOUT

EMAIL_TIMEOUT = 5

# Django Ninja
NINJA_PAGINATION_CLASS = 'ninja.pagination.PageNumberPagination'
NINJA_PAGINATION_PER_PAGE = config('NINJA_PAGINATION_PER_PAGE', default=20, cast=int)

# API key required to access the HTTP API
API_KEY = config('API_KEY', default=None)

# MinIO Configuration
MINIO_ENDPOINT = config('MINIO_ENDPOINT', default='localhost:9000')
MINIO_ACCESS_KEY = config('MINIO_ACCESS_KEY', default='minioadmin')
MINIO_SECRET_KEY = config('MINIO_SECRET_KEY', default='minioadmin')
MINIO_BUCKET_NAME = config('MINIO_BUCKET_NAME', default='document-processing')
MINIO_USE_SSL = config('MINIO_USE_SSL', default=False, cast=bool)

# Document Processing Configuration
MAX_PDF_SIZE_MB = config('MAX_PDF_SIZE_MB', default=50, cast=int)
PDF_TO_IMAGE_DPI = config('PDF_TO_IMAGE_DPI', default=150, cast=int)
ML_MODEL_PATH = config(
    'ML_MODEL_PATH',
    default='/code/mlmodels/doclayout_yolo.mlpkg/doclayout_yolo_model.pt',
)
ML_CONFIDENCE_THRESHOLD = config('ML_CONFIDENCE_THRESHOLD', default=0.25, cast=float)
ML_IOU_THRESHOLD = config('ML_IOU_THRESHOLD', default=0.45, cast=float)
ML_DEVICE = config('ML_DEVICE', default='cpu')

OLMOCR_MODEL_NAME = config('OLMOCR_MODEL_NAME', default='allenai/olmOCR-2-7B-1025')
OLMOCR_DEVICE = config('OLMOCR_DEVICE', default='cuda' if config('ML_DEVICE', default='cpu') == 'cuda' else 'cpu')

