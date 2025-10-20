# PDF Document Processing Service - Technical Specification

## Overview

A Django-based service for processing PDF documents through a multi-stage pipeline that extracts document layout and content using machine learning models. The service provides asynchronous processing with status tracking and result retrieval.

## Architecture

### Components

1. **Django REST API** (django-ninja)
   - Document upload endpoint
   - Status/result retrieval endpoint
   - API key authentication

2. **Celery Task Pipeline**
   - Chained tasks for sequential processing
   - Background job execution
   - Result storage in Django models

3. **MinIO Object Storage**
   - PDF file storage
   - Intermediate processing artifacts
   - Final output storage

4. **PostgreSQL Database**
   - Job metadata and status tracking
   - Processing results and document graphs

5. **Redis**
   - Celery broker and result backend
   - Caching layer

## Data Models

### DocumentJob

```python
class DocumentJob(models.Model):
    """Tracks PDF processing jobs with full state persistence"""
    
    # Identification
    id = UUIDField(primary_key=True, default=uuid4)
    
    # Timestamps
    created_at = DateTimeField(auto_now_add=True)
    updated_at = DateTimeField(auto_now=True)
    
    # File information
    original_filename = CharField(max_length=255)
    file_size = BigIntegerField()  # bytes
    minio_bucket = CharField(max_length=255)
    minio_key = CharField(max_length=500)  # path in bucket
    
    # Status tracking
    status = CharField(max_length=20, choices=STATUS_CHOICES)
    # STATUS_CHOICES: PENDING, PROCESSING, COMPLETED, FAILED
    
    current_step = CharField(max_length=50, null=True, blank=True)
    # Steps: UPLOAD, PDF_TO_IMAGES, ML_INFERENCE, DRAW_BOUNDING_BOXES, 
    #        TEXT_EXTRACTION, ASSEMBLE_GRAPH
    
    # Processing metadata
    total_pages = IntegerField(null=True, blank=True)
    processing_started_at = DateTimeField(null=True, blank=True)
    processing_completed_at = DateTimeField(null=True, blank=True)
    
    # Error handling
    error_message = TextField(null=True, blank=True)
    error_step = CharField(max_length=50, null=True, blank=True)
    retry_count = IntegerField(default=0)
    
    # Results
    document_graph = JSONField(null=True, blank=True)
    # Structure: see "Document Graph Schema" section
    
    # Celery task tracking
    celery_task_id = CharField(max_length=255, null=True, blank=True)
    # Main pipeline task ID for monitoring
```

### ProcessingStep

```python
class ProcessingStep(models.Model):
    """Tracks individual processing steps for fault tolerance"""
    
    # Relationships
    job = ForeignKey(DocumentJob, related_name='steps', on_delete=CASCADE)
    
    # Step identification
    step_name = CharField(max_length=50)
    # UPLOAD, PDF_TO_IMAGES, ML_INFERENCE, DRAW_BOUNDING_BOXES,
    # TEXT_EXTRACTION, ASSEMBLE_GRAPH
    
    step_order = IntegerField()  # 1, 2, 3, 4, 5, 6
    
    # Status tracking
    status = CharField(max_length=20, choices=STEP_STATUS_CHOICES)
    # STEP_STATUS_CHOICES: PENDING, IN_PROGRESS, COMPLETED, FAILED, SKIPPED
    
    # Timestamps
    started_at = DateTimeField(null=True, blank=True)
    completed_at = DateTimeField(null=True, blank=True)
    
    # Progress tracking
    progress_current = IntegerField(default=0)  # e.g., pages processed
    progress_total = IntegerField(null=True, blank=True)  # e.g., total pages
    progress_percentage = FloatField(default=0.0)
    
    # Results and metadata
    result_data = JSONField(null=True, blank=True)
    # Stores step-specific output data for recovery
    
    # Error handling
    error_message = TextField(null=True, blank=True)
    retry_count = IntegerField(default=0)
    max_retries = IntegerField(default=3)
    
    # Celery task tracking
    celery_task_id = CharField(max_length=255, null=True, blank=True)
    
    class Meta:
        ordering = ['step_order']
        unique_together = [['job', 'step_name']]
        indexes = [
            models.Index(fields=['job', 'status']),
            models.Index(fields=['step_name', 'status']),
        ]
```

### DocumentElement

```python
class DocumentElement(models.Model):
    """Individual elements detected in document"""
    
    # Relationships
    job = ForeignKey(DocumentJob, related_name='elements', on_delete=CASCADE)
    
    # Location
    page_number = IntegerField()  # 1-indexed
    
    # Bounding box (normalized coordinates 0-1)
    bbox_x1 = FloatField()
    bbox_y1 = FloatField()
    bbox_x2 = FloatField()
    bbox_y2 = FloatField()
    
    # Classification
    element_type = CharField(max_length=50)
    # Types: title, plain_text, abandon, figure, figure_caption,
    #        table, table_caption, table_footnote, isolate_formula, formula_caption
    
    confidence = FloatField()  # 0.0 to 1.0
    
    # Content
    extracted_text = TextField(null=True, blank=True)
    
    # MinIO references
    minio_image_key = CharField(max_length=500, null=True, blank=True)
    # Reference to cropped region image in step4
    
    # Ordering
    sequence = IntegerField()  # Reading order within page
```

## API Endpoints

### 1. Upload Document

**POST** `/api/documents/upload`

**Authentication**: API Key required

**Request**:
- Content-Type: `multipart/form-data`
- Body:
  - `file`: PDF file (max 50MB)
  - `callback_url` (optional): URL to POST when processing completes

**Response** (202 Accepted):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PENDING",
  "created_at": "2025-10-13T19:00:00Z",
  "message": "Document uploaded successfully and queued for processing"
}
```

**Error Responses**:
- 400 Bad Request: Invalid file format, missing file
- 413 Payload Too Large: File exceeds size limit
- 401 Unauthorized: Missing or invalid API key

### 2. Get Job Status

**GET** `/api/documents/{job_id}/status`

**Authentication**: API Key required

**Response** (200 OK):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PROCESSING",
  "current_step": "ML_INFERENCE",
  "progress": {
    "total_pages": 10,
    "processed_pages": 5
  },
  "created_at": "2025-10-13T19:00:00Z",
  "updated_at": "2025-10-13T19:05:00Z",
  "processing_started_at": "2025-10-13T19:00:05Z"
}
```

**Error Responses**:
- 404 Not Found: Job ID does not exist
- 401 Unauthorized: Missing or invalid API key

### 3. Get Job Results

**GET** `/api/documents/{job_id}/results`

**Authentication**: API Key required

**Response** (200 OK - when completed):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "COMPLETED",
  "original_filename": "document.pdf",
  "total_pages": 10,
  "processing_time_seconds": 45.2,
  "document_graph": {
    "pages": [
      {
        "page_number": 1,
        "elements": [
          {
            "id": 123,
            "type": "title",
            "bbox": [0.1, 0.1, 0.9, 0.2],
            "confidence": 0.95,
            "text": "Introduction",
            "sequence": 1
          },
          {
            "id": 124,
            "type": "plain_text",
            "bbox": [0.1, 0.25, 0.9, 0.8],
            "confidence": 0.92,
            "text": "This is the main body text...",
            "sequence": 2
          }
        ]
      }
    ],
    "metadata": {
      "total_elements": 45,
      "element_type_counts": {
        "title": 5,
        "plain_text": 30,
        "figure": 5,
        "table": 5
      }
    }
  }
}
```

**Response** (202 Accepted - when still processing):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PROCESSING",
  "message": "Job is still processing. Please check back later."
}
```

**Response** (500 Internal Server Error - when failed):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "FAILED",
  "error_step": "ML_INFERENCE",
  "error_message": "Model inference failed: CUDA out of memory"
}
```

**Error Responses**:
- 404 Not Found: Job ID does not exist
- 401 Unauthorized: Missing or invalid API key

## Processing Pipeline

### Task Chain Overview

```
upload_pdf (API) 
  → process_document_pipeline (celery chain)
    → step1_pdf_to_images
    → step2_ml_inference
    → step3_draw_bounding_boxes
    → step4_text_extraction
    → step5_assemble_graph
```

### Step 1: PDF to Images

**Task**: `step1_pdf_to_images`

**Input**:
- `job_id`: UUID
- PDF file from MinIO at `documents/{job_id}/original.pdf`

**Processing**:
1. Download PDF from MinIO
2. Convert each page to PNG image (1280x1280 resolution)
3. Upload images to MinIO at `documents/{job_id}/step1/page_{page_num:04d}.png`
4. Update job record with `total_pages`

**Output**:
- Images stored in MinIO: `documents/{job_id}/step1/page_*.png`
- Returns: `{"job_id": str, "total_pages": int}`

**Libraries**:
- `pdf2image` or `PyMuPDF` (fitz)

**Error Handling**:
- Invalid PDF format
- Corrupted file
- MinIO upload failures

### Step 2: ML Inference

**Task**: `step2_ml_inference`

**Input**:
- `job_id`: UUID
- `total_pages`: int
- Images from MinIO at `documents/{job_id}/step1/page_*.png`

**Processing**:
1. Load DocLayout-YOLO model from `mlmodels/doclayout-yolo.mlpkg/doclayout_yolo_model.pt`
2. For each page image:
   - Preprocess image (resize, normalize)
   - Run inference
   - Parse outputs (bounding boxes, classes, confidences)
   - Apply NMS (Non-Maximum Suppression)
   - Save results as JSON
3. Upload inference results to MinIO at `documents/{job_id}/step2/page_{page_num:04d}.json`

**Output**:
- JSON files stored in MinIO: `documents/{job_id}/step2/page_*.json`
- JSON format:
```json
{
  "page_number": 1,
  "detections": [
    {
      "bbox": [100, 150, 900, 250],
      "class_id": 0,
      "class_name": "title",
      "confidence": 0.95
    }
  ]
}
```
- Returns: `{"job_id": str, "total_pages": int}`

**Model Details**:
- Model path: `mlmodels/doclayout-yolo.mlpkg/doclayout_yolo_model.pt`
- Input size: 1280x1280
- Classes: 10 document element types (see README.md)
- Confidence threshold: 0.25
- IOU threshold: 0.45

**Libraries**:
- `torch`
- `torchvision`
- Reference: `mlmodels/doclayout-yolo.mlpkg/infer.py`

**Error Handling**:
- Model loading failures
- CUDA/memory errors
- Invalid image format

### Step 3: Draw Bounding Boxes

**Task**: `step3_draw_bounding_boxes`

**Input**:
- `job_id`: UUID
- `total_pages`: int
- Images from MinIO at `documents/{job_id}/step1/page_*.png`
- Inference results from MinIO at `documents/{job_id}/step2/page_*.json`

**Processing**:
1. For each page:
   - Load original image from step1
   - Load inference results from step2
   - Draw bounding boxes on image with:
     - Different colors per class
     - Class label and confidence score
     - Line thickness: 2-3px
   - Save annotated image
2. Upload annotated images to MinIO at `documents/{job_id}/step3/page_{page_num:04d}.png`

**Output**:
- Annotated images stored in MinIO: `documents/{job_id}/step3/page_*.png`
- Returns: `{"job_id": str, "total_pages": int}`

**Libraries**:
- `PIL` (Pillow) or `OpenCV` (cv2)

**Error Handling**:
- Image loading failures
- Drawing errors

### Step 4: Text Extraction

**Task**: `step4_text_extraction`

**Input**:
- `job_id`: UUID
- `total_pages`: int
- Original images from MinIO at `documents/{job_id}/step1/page_*.png`
- Inference results from MinIO at `documents/{job_id}/step2/page_*.json`

**Processing**:
1. For each page:
   - Load original image from step1
   - Load inference results from step2
   - For each detected region:
     - Crop image to bounding box
     - Run MinerU text extraction on cropped region
     - Save extracted text and metadata
     - Save cropped image for reference
2. Upload results to MinIO at `documents/{job_id}/step4/page_{page_num:04d}/`
   - `element_{element_idx:03d}.json` (text + metadata)
   - `element_{element_idx:03d}.png` (cropped image)

**Output**:
- JSON files: `documents/{job_id}/step4/page_*/element_*.json`
```json
{
  "element_index": 0,
  "bbox": [100, 150, 900, 250],
  "class_name": "title",
  "confidence": 0.95,
  "extracted_text": "Introduction",
  "text_confidence": 0.88
}
```
- Cropped images: `documents/{job_id}/step4/page_*/element_*.png`
- Returns: `{"job_id": str, "total_pages": int}`

**Libraries**:
- `magic-pdf` (MinerU)
- `PIL` (Pillow)

**Error Handling**:
- Text extraction failures
- Image cropping errors

### Step 5: Assemble Document Graph

**Task**: `step5_assemble_graph`

**Input**:
- `job_id`: UUID
- `total_pages`: int
- Text extraction results from MinIO at `documents/{job_id}/step4/`

**Processing**:
1. For each page:
   - Load all element JSONs from step4
   - Determine reading order (top-to-bottom, left-to-right)
   - Create DocumentElement records in database
2. Assemble complete document graph:
   - Organize by pages
   - Order elements by reading sequence
   - Calculate metadata and statistics
3. Store graph in DocumentJob.document_graph field
4. Update job status to COMPLETED

**Output**:
- DocumentElement records in database
- DocumentJob.document_graph populated
- DocumentJob.status = COMPLETED
- Returns: `{"job_id": str, "status": "COMPLETED"}`

**Libraries**:
- Django ORM

**Error Handling**:
- Database transaction errors
- JSON parsing errors

## MinIO Storage Structure

```
bucket: document-processing
├── documents/
│   └── {job_id}/
│       ├── original.pdf
│       ├── step1/
│       │   ├── page_0001.png
│       │   ├── page_0002.png
│       │   └── ...
│       ├── step2/
│       │   ├── page_0001.json
│       │   ├── page_0002.json
│       │   └── ...
│       ├── step3/
│       │   ├── page_0001.png  (annotated)
│       │   ├── page_0002.png  (annotated)
│       │   └── ...
│       └── step4/
│           ├── page_0001/
│           │   ├── element_001.json
│           │   ├── element_001.png
│           │   ├── element_002.json
│           │   ├── element_002.png
│           │   └── ...
│           └── page_0002/
│               └── ...
```

## Configuration

### Environment Variables

```bash
# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=document-processing
MINIO_USE_SSL=false

# Processing Configuration
MAX_PDF_SIZE_MB=50
PDF_TO_IMAGE_DPI=150
ML_MODEL_PATH=mlmodels/doclayout-yolo.mlpkg/doclayout_yolo_model.pt
ML_CONFIDENCE_THRESHOLD=0.25
ML_IOU_THRESHOLD=0.45
ML_DEVICE=cuda  # or cpu

# Celery Configuration
CELERY_TASK_TIME_LIMIT=1800  # 30 minutes
CELERY_TASK_SOFT_TIME_LIMIT=1500  # 25 minutes
```

### Dependencies to Add

```toml
# Core processing
pdf2image = ">=1.16.0"
# OR
PyMuPDF = ">=1.23.0"

# ML inference
torch = ">=2.0.0"
torchvision = ">=0.15.0"

# Image processing
Pillow = ">=10.0.0"

# Text extraction
magic-pdf = ">=0.7.0"  # MinerU

# Object storage
minio = ">=7.2.0"

# Utilities
python-magic = ">=0.4.27"  # File type detection
```

## Error Handling Strategy

### Retry Logic

- Network errors (MinIO): 3 retries with exponential backoff
- Transient ML errors: 2 retries
- Database errors: 3 retries

### Failure Scenarios

1. **Upload Failures**
   - Return 4xx error to client
   - No cleanup needed

2. **Processing Failures**
   - Update job status to FAILED
   - Store error message and step
   - Keep artifacts for debugging
   - Send webhook if callback_url provided

3. **Partial Failures**
   - Individual page failures: Log and continue
   - Critical failures: Stop pipeline, mark as FAILED

### Monitoring

- Celery task events logged to structlog
- Job status transitions tracked in PostgreSQL
- Processing time metrics per step
- Error rate metrics per step
- Step-level progress tracking

### Fault Tolerance & Recovery

**PostgreSQL-Based State Management**:
- All job and step state stored in PostgreSQL (not just RabbitMQ)
- Each task updates database before and after execution
- Database transactions ensure consistency
- ProcessingStep records provide complete audit trail

**Recovery Scenarios**:

1. **Worker Crash During Task Execution**
   - Step status remains IN_PROGRESS in database
   - Recovery script detects stale IN_PROGRESS steps (no update > 30 min)
   - Can retry failed step without reprocessing completed steps
   - MinIO artifacts from completed steps are preserved

2. **Network Failure to MinIO**
   - Task retries with exponential backoff (3 attempts)
   - Step status updated to FAILED if all retries exhausted
   - Error message stored in ProcessingStep.error_message
   - Manual retry possible via management command

3. **Database Connection Lost**
   - Celery task retries with backoff
   - Read-only operations (status check) can use Redis cache fallback
   - Write operations wait for database recovery

4. **Partial Page Processing Failure**
   - ProcessingStep.progress_current tracks completed pages
   - Task can resume from last successful page
   - Failed pages logged separately for manual review

5. **Model Inference OOM (Out of Memory)**
   - Task fails gracefully with error logged
   - Step marked as FAILED with specific error
   - Admin can adjust batch size and retry

**Recovery Management Commands**:

```bash
# Detect and retry stuck jobs
python manage.py retry_stuck_jobs --older-than-minutes=30

# Retry specific failed step
python manage.py retry_step <job_id> <step_name>

# Resume job from last completed step
python manage.py resume_job <job_id>

# Clean up orphaned MinIO artifacts
python manage.py cleanup_orphaned_artifacts --dry-run
```

**Progress Tracking in Each Task**:
- Task begins: Create/update ProcessingStep with status=IN_PROGRESS
- During processing: Update progress_current for each page processed
- Task completes: Update ProcessingStep with status=COMPLETED, store result_data
- Task fails: Update ProcessingStep with status=FAILED, store error_message

**Example Flow for step2_ml_inference**:
```python
def step2_ml_inference(job_id, total_pages):
    # 1. Update step status
    step = ProcessingStep.objects.get(job=job_id, step_name='ML_INFERENCE')
    step.status = 'IN_PROGRESS'
    step.started_at = now()
    step.progress_total = total_pages
    step.save()
    
    # 2. Process each page
    for page_num in range(1, total_pages + 1):
        try:
            # Process page...
            
            # Update progress in DB
            step.progress_current = page_num
            step.progress_percentage = (page_num / total_pages) * 100
            step.save(update_fields=['progress_current', 'progress_percentage'])
            
        except Exception as e:
            # Log error but continue with next page
            logger.error(f"Page {page_num} failed: {e}")
    
    # 3. Mark complete
    step.status = 'COMPLETED'
    step.completed_at = now()
    step.result_data = {'processed_pages': total_pages}
    step.save()
```

**Idempotency Guarantees**:
- Each task checks if step is already COMPLETED before processing
- Tasks can be safely retried without duplicating work
- MinIO uploads use consistent object keys (overwrite on retry)
- Database updates use transactions to prevent partial updates

## Security Considerations

1. **File Validation**
   - Verify PDF magic bytes
   - Scan for malware (optional integration)
   - Size limits enforced

2. **MinIO Access**
   - Pre-signed URLs for temporary access
   - Bucket policies restrict access
   - No public access to buckets

3. **API Authentication**
   - API key required for all endpoints
   - Rate limiting (optional)

4. **Resource Limits**
   - Max concurrent jobs per API key
   - Memory limits per task
   - Timeout enforcement

## Performance Considerations

1. **Concurrency**
   - Multiple Celery workers for parallel processing
   - GPU queue for ML inference tasks
   - CPU queue for other tasks

2. **Caching**
   - Model loaded once per worker
   - Redis caching for status checks

3. **Optimization**
   - Batch processing for multiple pages
   - Lazy loading of large artifacts
   - Compression for stored JSONs

## Testing Strategy

1. **Unit Tests**
   - Each task in isolation
   - Mocked MinIO client
   - Mocked ML model

2. **Integration Tests**
   - Full pipeline with sample PDFs
   - Real MinIO (test bucket)
   - Real Celery (test queue)

3. **Load Tests**
   - Concurrent job submissions
   - Large PDF processing
   - Error recovery scenarios

## Future Enhancements

1. **Batch Processing**: Support multiple PDFs in one request
2. **Custom Models**: Allow users to upload their own models
3. **Webhooks**: POST results to callback URL when done
4. **Visualization**: Web UI for viewing annotated documents
5. **Export Formats**: PDF, DOCX, Markdown output
6. **Language Support**: Multi-language OCR
7. **Search**: Full-text search across processed documents
8. **Versioning**: Track processing pipeline versions

## Document Graph Schema

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "version": "1.0",
  "pages": [
    {
      "page_number": 1,
      "width": 1280,
      "height": 1280,
      "elements": [
        {
          "id": 123,
          "type": "title",
          "bbox": [0.1, 0.1, 0.9, 0.2],
          "bbox_absolute": [128, 128, 1152, 256],
          "confidence": 0.95,
          "text": "Introduction",
          "text_confidence": 0.88,
          "sequence": 1,
          "minio_image_key": "documents/{job_id}/step4/page_0001/element_001.png"
        }
      ]
    }
  ],
  "metadata": {
    "total_pages": 10,
    "total_elements": 45,
    "processing_time_seconds": 45.2,
    "model_version": "doclayout-yolo-v1",
    "element_type_counts": {
      "title": 5,
      "plain_text": 30,
      "figure": 5,
      "table": 5
    }
  }
}
```

## Implementation Order

1. Create Django app structure and models
2. Set up MinIO client utilities
3. Implement upload endpoint
4. Implement status/results endpoints
5. Implement Step 1: PDF to Images
6. Implement Step 2: ML Inference
7. Implement Step 3: Draw Bounding Boxes
8. Implement Step 4: Text Extraction
9. Implement Step 5: Assemble Graph
10. Add error handling and retries
11. Add tests
12. Documentation and deployment guide
