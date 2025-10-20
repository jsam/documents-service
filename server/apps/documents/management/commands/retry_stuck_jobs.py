from django.core.management.base import BaseCommand
from django.utils import timezone

from server.apps.documents.models import DocumentJob, ProcessingStep
from server.apps.documents.tasks import process_document_pipeline


class Command(BaseCommand):
    help = 'Retry jobs that are stuck in processing state'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be retried without actually retrying',
        )
        parser.add_argument(
            '--stale-minutes',
            type=int,
            default=30,
            help='Consider jobs stale after this many minutes without updates',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        stale_minutes = options['stale_minutes']
        
        cutoff_time = timezone.now() - timezone.timedelta(minutes=stale_minutes)
        
        stuck_jobs = DocumentJob.objects.filter(
            status=DocumentJob.STATUS_PROCESSING,
            updated_at__lt=cutoff_time,
        )
        
        if not stuck_jobs.exists():
            self.stdout.write(self.style.SUCCESS('No stuck jobs found'))
            return
        
        self.stdout.write(f'Found {stuck_jobs.count()} stuck jobs')
        
        for job in stuck_jobs:
            if dry_run:
                self.stdout.write(f'Would retry job {job.id} ({job.original_filename})')
            else:
                self.stdout.write(f'Retrying job {job.id} ({job.original_filename})')
                
                job.status = DocumentJob.STATUS_PENDING
                job.retry_count += 1
                job.save(update_fields=['status', 'retry_count'])
                
                failed_steps = job.steps.filter(status__in=[ProcessingStep.STATUS_IN_PROGRESS, ProcessingStep.STATUS_FAILED])
                for step in failed_steps:
                    step.status = ProcessingStep.STATUS_PENDING
                    step.save(update_fields=['status'])
                
                process_document_pipeline.delay(str(job.id))
                
                self.stdout.write(self.style.SUCCESS(f'Queued job {job.id} for retry'))
