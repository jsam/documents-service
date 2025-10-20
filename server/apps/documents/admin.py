from django.contrib import admin
from django.utils.html import format_html

from server.apps.documents.models import DocumentElement, DocumentJob, ProcessingStep


class ProcessingStepInline(admin.TabularInline):
    model = ProcessingStep
    extra = 0
    readonly_fields = ('step_name', 'step_order', 'status', 'started_at', 'completed_at', 'progress_display', 'error_message', 'retry_count', 'celery_task_id')
    can_delete = False
    
    def progress_display(self, obj):
        if obj.progress_total:
            percentage = (obj.progress_current / obj.progress_total) * 100
            return format_html(
                '<progress value="{}" max="100"></progress> {}%',
                percentage,
                f'{percentage:.1f}',
            )
        return f'{obj.progress_current}'
    progress_display.short_description = 'Progress'


class DocumentElementInline(admin.TabularInline):
    model = DocumentElement
    extra = 0
    readonly_fields = ('page_number', 'element_type', 'confidence', 'sequence', 'bbox_display', 'extracted_text')
    can_delete = False
    max_num = 50
    
    def bbox_display(self, obj):
        return f'({obj.bbox_x1:.3f}, {obj.bbox_y1:.3f}) → ({obj.bbox_x2:.3f}, {obj.bbox_y2:.3f})'
    bbox_display.short_description = 'Bounding Box'


@admin.register(DocumentJob)
class DocumentJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'original_filename', 'status', 'total_pages', 'created_at', 'processing_started_at', 'processing_completed_at')
    list_filter = ('status', 'created_at')
    search_fields = ('id', 'original_filename')
    readonly_fields = (
        'id', 'created_at', 'updated_at', 'original_filename', 'file_size', 
        'minio_bucket', 'minio_key', 'status', 'current_step', 'total_pages',
        'processing_started_at', 'processing_completed_at', 'error_message', 
        'error_step', 'retry_count', 'celery_task_id', 'document_graph'
    )
    inlines = [ProcessingStepInline, DocumentElementInline]
    
    fieldsets = (
        ('Job Information', {
            'fields': ('id', 'original_filename', 'file_size', 'status', 'current_step')
        }),
        ('Storage', {
            'fields': ('minio_bucket', 'minio_key')
        }),
        ('Processing Timeline', {
            'fields': ('created_at', 'updated_at', 'processing_started_at', 'processing_completed_at', 'total_pages')
        }),
        ('Error Details', {
            'fields': ('error_message', 'error_step', 'retry_count'),
            'classes': ('collapse',)
        }),
        ('Technical', {
            'fields': ('celery_task_id', 'document_graph'),
            'classes': ('collapse',)
        }),
    )
    
    def has_add_permission(self, request):
        return False
    
    def has_delete_permission(self, request, obj=None):
        return True


@admin.register(ProcessingStep)
class ProcessingStepAdmin(admin.ModelAdmin):
    list_display = ('id', 'job', 'step_name', 'step_order', 'status', 'progress_display', 'started_at', 'completed_at', 'retry_count')
    list_filter = ('status', 'step_name', 'started_at')
    search_fields = ('job__id', 'job__original_filename', 'step_name')
    readonly_fields = (
        'job', 'step_name', 'step_order', 'status', 'started_at', 'completed_at',
        'progress_current', 'progress_total', 'progress_percentage', 'result_data',
        'error_message', 'retry_count', 'max_retries', 'celery_task_id'
    )
    
    def progress_display(self, obj):
        if obj.progress_total:
            return f'{obj.progress_current}/{obj.progress_total} ({obj.progress_percentage:.1f}%)'
        return f'{obj.progress_current}'
    progress_display.short_description = 'Progress'
    
    def has_add_permission(self, request):
        return False


@admin.register(DocumentElement)
class DocumentElementAdmin(admin.ModelAdmin):
    list_display = ('id', 'job', 'page_number', 'sequence', 'element_type', 'confidence', 'has_text')
    list_filter = ('element_type', 'page_number')
    search_fields = ('job__id', 'job__original_filename', 'extracted_text')
    readonly_fields = (
        'job', 'page_number', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
        'element_type', 'confidence', 'extracted_text', 'minio_image_key', 'sequence'
    )
    
    def has_text(self, obj):
        return bool(obj.extracted_text)
    has_text.boolean = True
    has_text.short_description = 'Has Text'
    
    def has_add_permission(self, request):
        return False
