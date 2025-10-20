from uuid import UUID

from django.core.management.base import BaseCommand, CommandError

from server.apps.documents.models import ProcessingStep


class Command(BaseCommand):
    help = 'Retry a specific failed processing step'

    def add_arguments(self, parser):
        parser.add_argument('job_id', type=str, help='Job UUID')
        parser.add_argument('step_name', type=str, help='Step name to retry')

    def handle(self, *args, **options):
        job_id = options['job_id']
        step_name = options['step_name']
        
        try:
            job_uuid = UUID(job_id)
        except ValueError:
            raise CommandError(f'Invalid job ID: {job_id}')
        
        try:
            step = ProcessingStep.objects.get(job_id=job_uuid, step_name=step_name)
        except ProcessingStep.DoesNotExist:
            raise CommandError(f'Step {step_name} not found for job {job_id}')
        
        if step.status == ProcessingStep.STATUS_COMPLETED:
            self.stdout.write(self.style.WARNING(f'Step {step_name} is already completed'))
            return
        
        if step.retry_count >= step.max_retries:
            self.stdout.write(self.style.ERROR(f'Step {step_name} has reached max retries ({step.max_retries})'))
            return
        
        step.status = ProcessingStep.STATUS_PENDING
        step.retry_count += 1
        step.error_message = None
        step.save(update_fields=['status', 'retry_count', 'error_message'])
        
        self.stdout.write(self.style.SUCCESS(f'Step {step_name} for job {job_id} reset to pending (retry {step.retry_count}/{step.max_retries})'))
