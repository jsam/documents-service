import json
from uuid import UUID

from django.utils import timezone

from server.apps.documents.models import DocumentJob, ProcessingStep
from server.apps.documents.utils.minio_client import upload_file
from server.apps.documents.utils.storage import get_graph_path


def execute_assemble_graph(job_id: str, celery_task_id: str) -> dict:
    job = DocumentJob.objects.get(id=UUID(job_id))
    step = ProcessingStep.objects.get(job=job, step_name='ASSEMBLE_GRAPH')
    
    step.mark_in_progress(celery_task_id)
    job.current_step = 'ASSEMBLE_GRAPH'
    job.save(update_fields=['current_step'])
    
    try:
        elements = job.elements.all().order_by('page_number', 'sequence')
        
        pages = {}
        for element in elements:
            page_num = element.page_number
            
            if page_num not in pages:
                pages[page_num] = {
                    'page_number': page_num,
                    'elements': [],
                }
            
            pages[page_num]['elements'].append({
                'element_id': element.id,
                'sequence': element.sequence,
                'element_type': element.element_type,
                'bbox': {
                    'x1': element.bbox_x1,
                    'y1': element.bbox_y1,
                    'x2': element.bbox_x2,
                    'y2': element.bbox_y2,
                },
                'confidence': element.confidence,
                'extracted_text': element.extracted_text,
                'minio_image_key': element.minio_image_key,
            })
        
        document_graph = {
            'job_id': str(job.id),
            'original_filename': job.original_filename,
            'total_pages': job.total_pages,
            'processing_completed_at': timezone.now().isoformat(),
            'pages': [pages[page_num] for page_num in sorted(pages.keys())],
        }
        
        job.document_graph = document_graph
        job.save(update_fields=['document_graph'])
        
        graph_path = get_graph_path(job.id)
        graph_json = json.dumps(document_graph, indent=2).encode('utf-8')
        
        upload_file(
            bucket=job.minio_bucket,
            object_name=graph_path,
            file_data=graph_json,
            content_type='application/json',
        )
        
        step.status = ProcessingStep.STATUS_COMPLETED
        step.completed_at = timezone.now()
        step.result_data = {'graph_path': graph_path, 'total_pages': job.total_pages, 'total_elements': elements.count()}
        step.save(update_fields=['status', 'completed_at', 'result_data'])
        
        job.mark_completed()
        
        return {'success': True, 'graph_path': graph_path}
    
    except Exception as e:
        step.status = ProcessingStep.STATUS_FAILED
        step.error_message = str(e)
        step.retry_count += 1
        step.save(update_fields=['status', 'error_message', 'retry_count'])
        
        job.mark_failed(str(e), 'ASSEMBLE_GRAPH')
        
        raise
