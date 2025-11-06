############################################################
# Runtime / bookkeeping
############################################################
ENVIRONMENT=production            # or staging / development

############################################################
# General
############################################################
DOMAIN_NAME=documents-service.local
API_KEY=supersecretkey123

############################################################
# Django core settings
############################################################
# ⚠️  Replace with a fresh key (50+ chars):
DJANGO_SECRET_KEY=GDnX1e5vN54k8A5KFWmvbtBheb634DLVLMuYFW8DF1S9dg126K
DJANGO_ALLOWED_HOSTS=documents-service.local,74.161.152.34,localhost,127.0.0.1
CSRF_TRUSTED_ORIGINS=https://documents-service.local

############################################################
# PostgreSQL  — matches .env.db exactly
############################################################
DJANGO_DATABASE_NAME=documents_service
DJANGO_DATABASE_USER=documents_service
DJANGO_DATABASE_PASSWORD=documents_service
DJANGO_DATABASE_HOST=db         # private IP if you have one
DJANGO_DATABASE_PORT=5432

# ─── Legacy keys (old settings.py may still look for these) ───
POSTGRES_DB=${DJANGO_DATABASE_NAME}
POSTGRES_USER=${DJANGO_DATABASE_USER}
POSTGRES_PASSWORD=${DJANGO_DATABASE_PASSWORD}
POSTGRES_HOST=${DJANGO_DATABASE_HOST}
PGUSER=${DJANGO_DATABASE_USER}              # used by pg_isready, etc.

############################################################
# RabbitMQ  — matches .env.db
############################################################
RABBITMQ_DEFAULT_USER=documents_service
RABBITMQ_DEFAULT_PASS=documents_service
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
CELERY_BROKER_URL=amqp://${RABBITMQ_DEFAULT_USER}:${RABBITMQ_DEFAULT_PASS}@${RABBITMQ_HOST}:${RABBITMQ_PORT}//        # Django & Celery

############################################################
# Redis  — no password set (adjust if you secure it later)
############################################################
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_URL=redis://${REDIS_HOST}:${REDIS_PORT}/0
CELERY_RESULT_BACKEND=redis://${REDIS_HOST}:${REDIS_PORT}/1    # optional

############################################################
# Caddy / TLS
############################################################
TLS_EMAIL=contact@documents-service.local


# MinIO settings
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_USE_SSL=False
MINIO_BUCKET_NAME=document-processing
ML_MODEL_PATH=/code/mlmodels/doclayout_yolo.mlpkg/doclayout_yolo_model.pt


# ML Device settings
ML_DEVICE=cuda
OLMOCR_DEVICE=cuda
