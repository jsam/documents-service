from django.core.management.base import BaseCommand

from server.apps.documents.utils.minio_client import ensure_bucket_exists


class Command(BaseCommand):
    help = 'Initialize MinIO bucket for document processing'

    def handle(self, *args, **options):
        self.stdout.write('Initializing MinIO bucket...')
        
        try:
            ensure_bucket_exists()
            self.stdout.write(self.style.SUCCESS('MinIO bucket initialized successfully'))
        except RuntimeError as e:
            self.stdout.write(self.style.ERROR(f'Failed to initialize MinIO bucket: {e}'))
            raise
