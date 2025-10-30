import uuid

from django.db import models
from django.utils import timezone


class DocumentJob(models.Model):
    """Tracks PDF processing jobs with full state persistence for fault tolerance."""

    STATUS_PENDING = 'PENDING'
    STATUS_PROCESSING = 'PROCESSING'
    STATUS_COMPLETED = 'COMPLETED'
    STATUS_FAILED = 'FAILED'

    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_PROCESSING, 'Processing'),
        (STATUS_COMPLETED, 'Completed'),
        (STATUS_FAILED, 'Failed'),
    ]

    STEP_UPLOAD = 'UPLOAD'
    STEP_PDF_TO_IMAGES = 'PDF_TO_IMAGES'
    STEP_PAGE_SEGMENTATION = 'PAGE_SEGMENTATION'
    STEP_DRAW_BOUNDING_BOXES = 'DRAW_BOUNDING_BOXES'
    STEP_TEXT_EXTRACTION = 'TEXT_EXTRACTION'
    STEP_ASSEMBLE_GRAPH = 'ASSEMBLE_GRAPH'

    STEP_CHOICES = [
        (STEP_UPLOAD, 'Upload'),
        (STEP_PDF_TO_IMAGES, 'PDF to Images'),
        (STEP_PAGE_SEGMENTATION, 'Page Segmentation'),
        (STEP_DRAW_BOUNDING_BOXES, 'Draw Bounding Boxes'),
        (STEP_TEXT_EXTRACTION, 'Text Extraction'),
        (STEP_ASSEMBLE_GRAPH, 'Assemble Graph'),
    ]

    # Identification
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    # File information
    original_filename = models.CharField(max_length=255)
    file_size = models.BigIntegerField(help_text='File size in bytes')
    minio_bucket = models.CharField(max_length=255, default='document-processing')
    minio_key = models.CharField(max_length=500, help_text='Path in MinIO bucket')

    # Status tracking
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
        db_index=True,
    )
    current_step = models.CharField(
        max_length=50,
        choices=STEP_CHOICES,
        null=True,
        blank=True,
    )

    # Processing metadata
    total_pages = models.IntegerField(null=True, blank=True)
    processing_started_at = models.DateTimeField(null=True, blank=True)
    processing_completed_at = models.DateTimeField(null=True, blank=True)

    # Error handling
    error_message = models.TextField(null=True, blank=True)
    error_step = models.CharField(max_length=50, null=True, blank=True)
    retry_count = models.IntegerField(default=0)

    # Results
    document_graph = models.JSONField(
        null=True,
        blank=True,
        help_text='Complete document structure graph',
    )

    # Celery task tracking
    celery_task_id = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text='Main pipeline Celery task ID',
    )

    class Meta:
        db_table = 'documents_job'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['celery_task_id']),
        ]

    def __str__(self):
        return f'Job {self.id} - {self.original_filename} ({self.status})'

    @property
    def processing_duration(self):
        """Calculate processing duration if applicable."""
        if self.processing_started_at and self.processing_completed_at:
            return (
                self.processing_completed_at - self.processing_started_at
            ).total_seconds()
        return None

    def mark_processing(self):
        """Mark job as processing and set start time."""
        self.status = self.STATUS_PROCESSING
        if not self.processing_started_at:
            self.processing_started_at = timezone.now()
        self.save(update_fields=['status', 'processing_started_at', 'updated_at'])

    def mark_completed(self):
        """Mark job as completed."""
        self.status = self.STATUS_COMPLETED
        self.processing_completed_at = timezone.now()
        self.save(
            update_fields=['status', 'processing_completed_at', 'updated_at'],
        )

    def mark_failed(self, error_message: str, error_step: str | None = None):
        """Mark job as failed with error details."""
        self.status = self.STATUS_FAILED
        self.error_message = error_message
        if error_step:
            self.error_step = error_step
        self.save(
            update_fields=['status', 'error_message', 'error_step', 'updated_at'],
        )


class ProcessingStep(models.Model):
    """Tracks individual processing steps for fault tolerance and recovery."""

    STATUS_PENDING = 'PENDING'
    STATUS_IN_PROGRESS = 'IN_PROGRESS'
    STATUS_COMPLETED = 'COMPLETED'
    STATUS_FAILED = 'FAILED'
    STATUS_SKIPPED = 'SKIPPED'

    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_IN_PROGRESS, 'In Progress'),
        (STATUS_COMPLETED, 'Completed'),
        (STATUS_FAILED, 'Failed'),
        (STATUS_SKIPPED, 'Skipped'),
    ]

    # Step names match DocumentJob.STEP_* constants
    STEP_UPLOAD = 'UPLOAD'
    STEP_PDF_TO_IMAGES = 'PDF_TO_IMAGES'
    STEP_PAGE_SEGMENTATION = 'PAGE_SEGMENTATION'
    STEP_DRAW_BOUNDING_BOXES = 'DRAW_BOUNDING_BOXES'
    STEP_TEXT_EXTRACTION = 'TEXT_EXTRACTION'
    STEP_ASSEMBLE_GRAPH = 'ASSEMBLE_GRAPH'

    STEP_CHOICES = [
        (STEP_UPLOAD, 'Upload'),
        (STEP_PDF_TO_IMAGES, 'PDF to Images'),
        (STEP_PAGE_SEGMENTATION, 'Page Segmentation'),
        (STEP_DRAW_BOUNDING_BOXES, 'Draw Bounding Boxes'),
        (STEP_TEXT_EXTRACTION, 'Text Extraction'),
        (STEP_ASSEMBLE_GRAPH, 'Assemble Graph'),
    ]

    # Relationships
    job = models.ForeignKey(
        DocumentJob,
        on_delete=models.CASCADE,
        related_name='steps',
    )

    # Step identification
    step_name = models.CharField(max_length=50, choices=STEP_CHOICES)
    step_order = models.IntegerField(help_text='Sequential order of step execution')

    # Status tracking
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
        db_index=True,
    )

    # Timestamps
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    # Progress tracking
    progress_current = models.IntegerField(
        default=0,
        help_text='Current progress (e.g., pages processed)',
    )
    progress_total = models.IntegerField(
        null=True,
        blank=True,
        help_text='Total items to process (e.g., total pages)',
    )
    progress_percentage = models.FloatField(
        default=0.0,
        help_text='Progress as percentage (0-100)',
    )

    # Results and metadata
    result_data = models.JSONField(
        null=True,
        blank=True,
        help_text='Step-specific output data for recovery',
    )

    # Error handling
    error_message = models.TextField(null=True, blank=True)
    retry_count = models.IntegerField(default=0)
    max_retries = models.IntegerField(default=3)

    # Celery task tracking
    celery_task_id = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text='Individual step Celery task ID',
    )

    class Meta:
        db_table = 'documents_processing_step'
        ordering = ['step_order']
        unique_together = [['job', 'step_name']]
        indexes = [
            models.Index(fields=['job', 'status']),
            models.Index(fields=['step_name', 'status']),
            models.Index(fields=['status', 'started_at']),
        ]

    def __str__(self):
        return f'{self.job.id} - {self.get_step_name_display()} ({self.status})'

    @property
    def duration(self):
        """Calculate step duration if applicable."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_stale(self):
        """Check if step is stale (in progress but no update > 30 minutes)."""
        if self.status == self.STATUS_IN_PROGRESS and self.started_at:
            time_since_start = timezone.now() - self.started_at
            return time_since_start.total_seconds() > 1800  # 30 minutes
        return False

    def mark_in_progress(self, celery_task_id: str | None = None):
        """Mark step as in progress."""
        self.status = self.STATUS_IN_PROGRESS
        if not self.started_at:
            self.started_at = timezone.now()
        if celery_task_id:
            self.celery_task_id = celery_task_id
        self.save(
            update_fields=['status', 'started_at', 'celery_task_id'],
        )

    def update_progress(self, current: int, total: int | None = None):
        """Update progress tracking."""
        self.progress_current = current
        if total is not None:
            self.progress_total = total
        if self.progress_total and self.progress_total > 0:
            self.progress_percentage = (current / self.progress_total) * 100
        self.save(
            update_fields=[
                'progress_current',
                'progress_total',
                'progress_percentage',
            ],
        )

    def mark_completed(self, result_data: dict | None = None):
        """Mark step as completed."""
        self.status = self.STATUS_COMPLETED
        self.completed_at = timezone.now()
        if result_data:
            self.result_data = result_data
        self.progress_percentage = 100.0
        self.save(
            update_fields=[
                'status',
                'completed_at',
                'result_data',
                'progress_percentage',
            ],
        )

    def mark_failed(self, error_message: str):
        """Mark step as failed."""
        self.status = self.STATUS_FAILED
        self.error_message = error_message
        self.save(update_fields=['status', 'error_message'])

    def can_retry(self):
        """Check if step can be retried."""
        return self.retry_count < self.max_retries


class DocumentElement(models.Model):
    """Individual elements detected in document pages."""

    # Element type choices (matching DocLayout-YOLO model classes)
    TYPE_TITLE = 'title'
    TYPE_PLAIN_TEXT = 'plain_text'
    TYPE_ABANDON = 'abandon'
    TYPE_FIGURE = 'figure'
    TYPE_FIGURE_CAPTION = 'figure_caption'
    TYPE_TABLE = 'table'
    TYPE_TABLE_CAPTION = 'table_caption'
    TYPE_TABLE_FOOTNOTE = 'table_footnote'
    TYPE_ISOLATE_FORMULA = 'isolate_formula'
    TYPE_FORMULA_CAPTION = 'formula_caption'

    TYPE_CHOICES = [
        (TYPE_TITLE, 'Title'),
        (TYPE_PLAIN_TEXT, 'Plain Text'),
        (TYPE_ABANDON, 'Abandon'),
        (TYPE_FIGURE, 'Figure'),
        (TYPE_FIGURE_CAPTION, 'Figure Caption'),
        (TYPE_TABLE, 'Table'),
        (TYPE_TABLE_CAPTION, 'Table Caption'),
        (TYPE_TABLE_FOOTNOTE, 'Table Footnote'),
        (TYPE_ISOLATE_FORMULA, 'Isolated Formula'),
        (TYPE_FORMULA_CAPTION, 'Formula Caption'),
    ]

    # Relationships
    job = models.ForeignKey(
        DocumentJob,
        on_delete=models.CASCADE,
        related_name='elements',
    )

    # Location
    page_number = models.IntegerField(help_text='Page number (1-indexed)')

    # Bounding box (normalized coordinates 0-1)
    bbox_x1 = models.FloatField(help_text='Left x coordinate (normalized)')
    bbox_y1 = models.FloatField(help_text='Top y coordinate (normalized)')
    bbox_x2 = models.FloatField(help_text='Right x coordinate (normalized)')
    bbox_y2 = models.FloatField(help_text='Bottom y coordinate (normalized)')

    # Classification
    element_type = models.CharField(max_length=50, choices=TYPE_CHOICES)
    confidence = models.FloatField(help_text='Detection confidence (0-1)')

    # Content
    extracted_text = models.TextField(null=True, blank=True)
    
    # Structured table data
    table_html = models.TextField(
        null=True,
        blank=True,
        help_text='HTML representation of table structure (for table elements)',
    )
    table_data = models.JSONField(
        null=True,
        blank=True,
        help_text='Structured table data as 2D array (for table elements)',
    )

    # MinIO references
    minio_image_key = models.CharField(
        max_length=500,
        null=True,
        blank=True,
        help_text='Reference to cropped region image in text_extraction task',
    )

    # Ordering
    sequence = models.IntegerField(
        help_text='Reading order within page',
    )

    class Meta:
        db_table = 'documents_element'
        ordering = ['job', 'page_number', 'sequence']
        indexes = [
            models.Index(fields=['job', 'page_number']),
            models.Index(fields=['element_type']),
        ]

    def __str__(self):
        return (
            f'Page {self.page_number} - '
            f'{self.get_element_type_display()} '
            f'(seq {self.sequence})'
        )

    @property
    def bbox_absolute(self):
        """Get absolute bounding box coordinates for 1280x1280 image."""
        return [
            int(self.bbox_x1 * 1280),
            int(self.bbox_y1 * 1280),
            int(self.bbox_x2 * 1280),
            int(self.bbox_y2 * 1280),
        ]

    @property
    def bbox_normalized(self):
        """Get normalized bounding box as list."""
        return [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2]
