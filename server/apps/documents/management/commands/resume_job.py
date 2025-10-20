from uuid import UUID

from django.core.management.base import BaseCommand, CommandError

from server.apps.documents.models import DocumentJob, ProcessingStep
from server.apps.documents.tasks import process_document_pipeline


class Command(BaseCommand):
    help = 'Resume a failed or stuck job from the last successful step'

    def add_arguments(self, parser):
        parser.add_argument('job_id', type=str, help='Job UUID to resume')

    def handle(self, *args, **options):
        job_id = options['job_id']
        
        try:
            job_uuid = UUID(job_id)
        except ValueError:
            raise CommandError(f'Invalid job ID: {job_id}')
        
        try:
            job = DocumentJob.objects.get(id=job_uuid)
        except DocumentJob.DoesNotExist:
            raise CommandError(f'Job {job_id} not found')
        
        if job.status == DocumentJob.STATUS_COMPLETED:
            self.stdout.write(self.style.WARNING('Job is already completed'))
            return
        
        failed_or_pending_steps = job.steps.filter(
            status__in=[ProcessingStep.STATUS_PENDING, ProcessingStep.STATUS_FAILED, ProcessingStep.STATUS_IN_PROGRESS]
        ).order_by('step_order')
        
        if not failed_or_pending_steps.exists():
            self.stdout.write(self.style.ERROR('No steps to resume'))
            return
        
        for step in failed_or_pending_steps:
            if step.status == ProcessingStep.STATUS_FAILED:
                if step.retry_count >= step.max_retries:
                    self.stdout.write(self.style.ERROR(f'Step {step.step_name} has reached max retries'))
                    return
            
            step.status = ProcessingStep.STATUS_PENDING
            step.error_message = None
            if step.status == ProcessingStep.STATUS_FAILED:
                step.retry_count += 1
            step.save(update_fields=['status', 'error_message', 'retry_count'])
        
        job.status = DocumentJob.STATUS_PENDING
        job.error_message = None
        job.save(update_fields=['status', 'error_message'])
        
        process_document_pipeline.delay(str(job.id))
        
        self.stdout.write(self.style.SUCCESS(f'Job {job_id} queued for resumption'))
