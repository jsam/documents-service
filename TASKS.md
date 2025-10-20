# PDF Document Processing Service - Implementation Tasks

## Progress Overview

- [ ] **Phase 1**: Setup & Infrastructure (Tasks 1-5)
- [ ] **Phase 2**: Models & Database (Tasks 6-8)
- [ ] **Phase 3**: MinIO Integration (Tasks 9-11)
- [ ] **Phase 4**: API Endpoints (Tasks 12-15)
- [ ] **Phase 5**: Celery Tasks - Core Processing (Tasks 16-20)
- [ ] **Phase 6**: Celery Tasks - ML & Text (Tasks 21-25)
- [ ] **Phase 7**: Integration & Testing (Tasks 26-30)

---

## Phase 1: Setup & Infrastructure

### Task 1: Create Django App Structure
- [ ] Create `server/apps/documents` directory
- [ ] Create `__init__.py`
- [ ] Create `apps.py` with DocumentsConfig
- [ ] Create `models.py` (placeholder)
- [ ] Create `api.py` (placeholder)
- [ ] Create `tasks.py` (placeholder)
- [ ] Create `schemas.py` (placeholder)
- [ ] Create `utils/` directory
- [ ] Create `tests/` directory

**Files to create:**
- `server/apps/documents/__init__.py`
- `server/apps/documents/apps.py`
- `server/apps/documents/models.py`
- `server/apps/documents/api.py`
- `server/apps/documents/tasks.py`
- `server/apps/documents/schemas.py`
- `server/apps/documents/admin.py`
- `server/apps/documents/utils/__init__.py`

---

### Task 2: Add Required Dependencies
- [ ] Add `PyMuPDF` (fitz) for PDF processing
- [ ] Add `Pillow` for image processing
- [ ] Add `minio` for object storage
- [ ] Add `python-magic` for file type detection
- [ ] Add `torch` and `torchvision` for ML inference
- [ ] Add `magic-pdf` (MinerU) for text extraction
- [ ] Run `uv sync` to install dependencies

**File to modify:**
- `pyproject.toml`

**Dependencies to add:**
```toml
dependencies = [
  # ... existing ...
  "PyMuPDF>=1.24.0",
  "Pillow>=10.0.0",
  "minio>=7.2.0",
  "python-magic>=0.4.27",
  "torch>=2.0.0",
  "torchvision>=0.15.0",
  "magic-pdf>=0.7.0",
]
```

---

### Task 3: Update Django Settings
- [ ] Add `server.apps.documents` to INSTALLED_APPS
- [ ] Add MinIO configuration settings
- [ ] Add processing configuration (thresholds, limits)
- [ ] Add ML model configuration

**File to modify:**
- `server/settings/components/common.py`

**Settings to add:**
```python
# MinIO Configuration
MINIO_ENDPOINT = config('MINIO_ENDPOINT', default='localhost:9000')
MINIO_ACCESS_KEY = config('MINIO_ACCESS_KEY', default='minioadmin')
MINIO_SECRET_KEY = config('MINIO_SECRET_KEY', default='minioadmin')
MINIO_BUCKET_NAME = config('MINIO_BUCKET_NAME', default='document-processing')
MINIO_USE_SSL = config('MINIO_USE_SSL', default=False, cast=bool)

# Document Processing Configuration
MAX_PDF_SIZE_MB = config('MAX_PDF_SIZE_MB', default=50, cast=int)
PDF_TO_IMAGE_DPI = config('PDF_TO_IMAGE_DPI', default=150, cast=int)
ML_MODEL_PATH = config('ML_MODEL_PATH', default='mlmodels/doclayout-yolo.mlpkg/doclayout_yolo_model.pt')
ML_CONFIDENCE_THRESHOLD = config('ML_CONFIDENCE_THRESHOLD', default=0.25, cast=float)
ML_IOU_THRESHOLD = config('ML_IOU_THRESHOLD', default=0.45, cast=float)
ML_DEVICE = config('ML_DEVICE', default='cpu')
```

---

### Task 4: Add Environment Variables
- [ ] Add MinIO variables to `.env.template`
- [ ] Add MinIO variables to `.env`
- [ ] Add processing configuration variables
- [ ] Add ML model configuration variables

**Files to modify:**
- `config/.env.template`
- `config/.env`

**Variables to add:**
```bash
# === MinIO ===
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=document-processing
MINIO_USE_SSL=false

# === Document Processing ===
MAX_PDF_SIZE_MB=50
PDF_TO_IMAGE_DPI=150
ML_MODEL_PATH=mlmodels/doclayout-yolo.mlpkg/doclayout_yolo_model.pt
ML_CONFIDENCE_THRESHOLD=0.25
ML_IOU_THRESHOLD=0.45
ML_DEVICE=cpu
```

---

### Task 5: Verify ML Model Files
- [ ] Check if `mlmodels/doclayout-yolo.mlpkg/` exists
- [ ] Verify `doclayout_yolo_model.pt` exists
- [ ] Verify `config.json` exists
- [ ] Review `infer.py` for inference logic reference
- [ ] Document model input/output format

**Files to check:**
- `mlmodels/doclayout-yolo.mlpkg/doclayout_yolo_model.pt`
- `mlmodels/doclayout-yolo.mlpkg/config.json`
- `mlmodels/doclayout-yolo.mlpkg/infer.py`

---

## Phase 2: Models & Database

### Task 6: Create DocumentJob Model
- [ ] Define DocumentJob model with all fields from spec
- [ ] Add UUID primary key
- [ ] Add status choices (PENDING, PROCESSING, COMPLETED, FAILED)
- [ ] Add step choices (UPLOAD, PDF_TO_IMAGES, ML_INFERENCE, etc.)
- [ ] Add JSONField for document_graph
- [ ] Add timestamps (created_at, updated_at)
- [ ] Add file metadata fields
- [ ] Add error tracking fields
- [ ] Add `__str__` method
- [ ] Add Meta class with ordering

**File to modify:**
- `server/apps/documents/models.py`

---

### Task 7: Create ProcessingStep Model
- [ ] Define ProcessingStep model with all fields from spec
- [ ] Add ForeignKey to DocumentJob with related_name='steps'
- [ ] Add step_name and step_order fields
- [ ] Add status choices (PENDING, IN_PROGRESS, COMPLETED, FAILED, SKIPPED)
- [ ] Add progress tracking fields (current, total, percentage)
- [ ] Add result_data JSONField for step outputs
- [ ] Add retry tracking fields
- [ ] Add celery_task_id field
- [ ] Add timestamps (started_at, completed_at)
- [ ] Add `__str__` method
- [ ] Add Meta class with unique_together constraint
- [ ] Add database indexes for performance

**File to modify:**
- `server/apps/documents/models.py`

---

### Task 8: Create DocumentElement Model
- [ ] Define DocumentElement model with all fields from spec
- [ ] Add ForeignKey to DocumentJob
- [ ] Add page_number field
- [ ] Add bounding box fields (x1, y1, x2, y2)
- [ ] Add element_type choices (title, plain_text, etc.)
- [ ] Add confidence field
- [ ] Add extracted_text field
- [ ] Add minio_image_key field
- [ ] Add sequence field for ordering
- [ ] Add `__str__` method
- [ ] Add Meta class with ordering and indexes

**File to modify:**
- `server/apps/documents/models.py`

---

### Task 9: Create and Run Migrations
- [ ] Run `python manage.py makemigrations documents`
- [ ] Review generated migration file
- [ ] Run `python manage.py migrate`
- [ ] Verify tables created in PostgreSQL
- [ ] Test model creation in Django shell

**Commands to run:**
```bash
python manage.py makemigrations documents
python manage.py migrate
python manage.py shell
# >>> from server.apps.documents.models import DocumentJob
# >>> DocumentJob.objects.create(...)
```

---

## Phase 3: MinIO Integration

### Task ### Task 31: Create MinIO Client Utility
- [ ] Create `server/apps/documents/utils/minio_client.py`
- [ ] Initialize MinIO client with settings
- [ ] Create bucket if not exists
- [ ] Implement `upload_file()` method
- [ ] Implement `download_file()` method
- [ ] Implement `upload_bytes()` method
- [ ] Implement `download_bytes()` method
- [ ] Implement `list_objects()` method
- [ ] Implement `delete_object()` method
- [ ] Add error handling and logging
- [ ] Create singleton instance

**File to create:**
- `server/apps/documents/utils/minio_client.py`

**Key functions:**
```python
def get_minio_client() -> Minio:
    """Get singleton MinIO client instance"""

def ensure_bucket_exists(bucket_name: str):
    """Create bucket if it doesn't exist"""

def upload_file(bucket: str, object_name: str, file_path: str):
    """Upload file to MinIO"""

def download_file(bucket: str, object_name: str, file_path: str):
    """Download file from MinIO"""
```

---

### Task ### Task 31: Create Storage Path Utilities
- [ ] Create `server/apps/documents/utils/storage.py`
- [ ] Implement `get_original_pdf_path(job_id)` function
- [ ] Implement `get_step1_image_path(job_id, page_num)` function
- [ ] Implement `get_step2_json_path(job_id, page_num)` function
- [ ] Implement `get_step3_image_path(job_id, page_num)` function
- [ ] Implement `get_step4_element_path(job_id, page_num, element_idx)` function
- [ ] Add path formatting utilities
- [ ] Add tests for path generation

**File to create:**
- `server/apps/documents/utils/storage.py`

**Key functions:**
```python
def get_original_pdf_path(job_id: str) -> str:
    """Returns: documents/{job_id}/original.pdf"""

def get_step1_image_path(job_id: str, page_num: int) -> str:
    """Returns: documents/{job_id}/step1/page_{page_num:04d}.png"""
```

---

### Task ### Task 31: Create MinIO Initialization Script
- [ ] Create `server/apps/documents/management/commands/init_minio.py`
- [ ] Implement command to create bucket
- [ ] Implement command to set bucket policy
- [ ] Implement command to verify connection
- [ ] Add verbose output
- [ ] Test command execution

**File to create:**
- `server/apps/documents/management/commands/init_minio.py`

**Command:**
```bash
python manage.py init_minio
```

---

## Phase 4: API Endpoints

### Task ### Task 31: Create API Schemas
- [ ] Create `UploadResponse` schema
- [ ] Create `StatusResponse` schema
- [ ] Create `ResultsResponse` schema
- [ ] Create `DocumentGraphSchema` schema
- [ ] Create `PageSchema` schema
- [ ] Create `ElementSchema` schema
- [ ] Create `ErrorResponse` schema
- [ ] Add schema validation

**File to modify:**
- `server/apps/documents/schemas.py`

**Schemas:**
```python
class UploadResponse(Schema):
    job_id: str
    status: str
    created_at: datetime
    message: str

class StatusResponse(Schema):
    job_id: str
    status: str
    current_step: str | None
    progress: dict | None
    created_at: datetime
    updated_at: datetime
```

---

### Task ### Task 31: Implement Upload Endpoint
- [ ] Create POST `/api/documents/upload` endpoint
- [ ] Add file validation (PDF, size limit)
- [ ] Create DocumentJob record
- [ ] Upload PDF to MinIO
- [ ] Trigger Celery task pipeline
- [ ] Return 202 Accepted with job_id
- [ ] Add error handling (400, 413, 500)
- [ ] Add request logging
- [ ] Test with sample PDFs

**File to modify:**
- `server/apps/documents/api.py`

**Endpoint:**
```python
@router.post('/upload', response={202: UploadResponse, 400: ErrorResponse})
def upload_document(request, file: UploadedFile):
    """Upload PDF document for processing"""
```

---

### Task ### Task 31: Implement Status Endpoint
- [ ] Create GET `/api/documents/{job_id}/status` endpoint
- [ ] Query DocumentJob by ID
- [ ] Return status and progress information
- [ ] Handle 404 for non-existent jobs
- [ ] Add caching for status checks
- [ ] Add logging
- [ ] Test with various job states

**File to modify:**
- `server/apps/documents/api.py`

**Endpoint:**
```python
@router.get('/{job_id}/status', response={200: StatusResponse, 404: ErrorResponse})
def get_job_status(request, job_id: str):
    """Get processing status of a document job"""
```

---

### Task ### Task 31: Implement Results Endpoint
- [ ] Create GET `/api/documents/{job_id}/results` endpoint
- [ ] Query DocumentJob with related elements
- [ ] Return 200 with results if COMPLETED
- [ ] Return 202 if still PROCESSING
- [ ] Return 500 if FAILED with error details
- [ ] Return 404 for non-existent jobs
- [ ] Serialize document graph
- [ ] Add caching for completed results
- [ ] Test with completed jobs

**File to modify:**
- `server/apps/documents/api.py`

**Endpoint:**
```python
@router.get('/{job_id}/results', response={200: ResultsResponse, 202: StatusResponse, 500: ErrorResponse, 404: ErrorResponse})
def get_job_results(request, job_id: str):
    """Get results of a completed document job"""
```

---

## Phase 5: Celery Tasks - Core Processing

### Task ### Task 31: Create Pipeline Chain Task
- [ ] Create `server/apps/documents/tasks.py`
- [ ] Import Celery app
- [ ] Create `process_document_pipeline()` task
- [ ] Chain all 5 processing steps
- [ ] Add error callbacks
- [ ] Add success callbacks
- [ ] Update job status at start
- [ ] Add logging
- [ ] Test pipeline execution

**File to modify:**
- `server/apps/documents/tasks.py`

**Task:**
```python
@shared_task
def process_document_pipeline(job_id: str):
    """Main pipeline that chains all processing steps"""
    chain(
        step1_pdf_to_images.si(job_id),
        step2_ml_inference.s(),
        step3_draw_bounding_boxes.s(),
        step4_text_extraction.s(),
        step5_assemble_graph.s(),
    ).apply_async()
```

---

### Task ### Task 31: Implement Step 1 - PDF to Images
- [ ] Create `step1_pdf_to_images()` task
- [ ] Download PDF from MinIO
- [ ] Use PyMuPDF to convert pages to images
- [ ] Resize images to 1280x1280
- [ ] Save images as PNG
- [ ] Upload images to MinIO step1/
- [ ] Update job with total_pages
- [ ] Update job status and current_step
- [ ] Add error handling and retries
- [ ] Add logging
- [ ] Test with various PDFs

**File to modify:**
- `server/apps/documents/tasks.py`

**Task signature:**
```python
@shared_task(bind=True, max_retries=3)
def step1_pdf_to_images(self, job_id: str) -> dict:
    """Convert PDF pages to images"""
    # Returns: {"job_id": str, "total_pages": int}
```

---

### Task ### Task 31: Create Model Loader Utility
- [ ] Create `server/apps/documents/utils/model_loader.py`
- [ ] Implement singleton model loader
- [ ] Load TorchScript model from file
- [ ] Move model to configured device (CPU/CUDA)
- [ ] Set model to eval mode
- [ ] Add model caching per worker
- [ ] Add error handling
- [ ] Test model loading

**File to create:**
- `server/apps/documents/utils/model_loader.py`

**Key functions:**
```python
def load_doclayout_model() -> torch.jit.ScriptModule:
    """Load and cache DocLayout-YOLO model"""

def get_model_device() -> torch.device:
    """Get configured device (cpu/cuda)"""
```

---

### Task ### Task 31: Create Image Preprocessing Utility
- [ ] Create `server/apps/documents/utils/image_processing.py`
- [ ] Implement `preprocess_for_inference()` function
- [ ] Resize image to 1280x1280
- [ ] Normalize to [0, 1] range
- [ ] Convert to CHW format (channels, height, width)
- [ ] Add batch dimension
- [ ] Convert to torch tensor
- [ ] Test preprocessing

**File to create:**
- `server/apps/documents/utils/image_processing.py`

**Key functions:**
```python
def preprocess_for_inference(image_path: str) -> torch.Tensor:
    """Preprocess image for ML inference"""

def postprocess_predictions(outputs, conf_threshold, iou_threshold):
    """Apply NMS and filter predictions"""
```

---

### Task 31: Implement Step 2 - ML Inference
- [ ] Create `step2_ml_inference()` task
- [ ] Load DocLayout-YOLO model
- [ ] Download images from MinIO step1/
- [ ] For each image:
  - [ ] Preprocess image
  - [ ] Run inference
  - [ ] Parse outputs (boxes, classes, scores)
  - [ ] Apply NMS
  - [ ] Save results as JSON
  - [ ] Upload JSON to MinIO step2/
- [ ] Update job status and current_step
- [ ] Add error handling and retries
- [ ] Add logging
- [ ] Test with sample images

**File to modify:**
- `server/apps/documents/tasks.py`

**Task signature:**
```python
@shared_task(bind=True, max_retries=2)
def step2_ml_inference(self, result: dict) -> dict:
    """Run ML inference on document images"""
    # Input: {"job_id": str, "total_pages": int}
    # Returns: {"job_id": str, "total_pages": int}
```

---

## Phase 6: Celery Tasks - ML & Text

### Task 31: Create Drawing Utilities
- [ ] Create `server/apps/documents/utils/visualization.py`
- [ ] Implement `draw_bounding_boxes()` function
- [ ] Define color map for each class
- [ ] Draw rectangles with class colors
- [ ] Add text labels with class name and confidence
- [ ] Adjust line thickness
- [ ] Save annotated image
- [ ] Test visualization

**File to create:**
- `server/apps/documents/utils/visualization.py`

**Key functions:**
```python
def draw_bounding_boxes(image_path: str, detections: list, output_path: str):
    """Draw bounding boxes on image with labels"""

CLASS_COLORS = {
    'title': (255, 0, 0),
    'plain_text': (0, 255, 0),
    # ... etc
}
```

---

### Task 31: Implement Step 3 - Draw Bounding Boxes
- [ ] Create `step3_draw_bounding_boxes()` task
- [ ] Download images from MinIO step1/
- [ ] Download inference results from MinIO step2/
- [ ] For each page:
  - [ ] Load image
  - [ ] Load detections JSON
  - [ ] Draw bounding boxes with labels
  - [ ] Save annotated image
  - [ ] Upload to MinIO step3/
- [ ] Update job status and current_step
- [ ] Add error handling
- [ ] Add logging
- [ ] Test with inference results

**File to modify:**
- `server/apps/documents/tasks.py`

**Task signature:**
```python
@shared_task(bind=True, max_retries=3)
def step3_draw_bounding_boxes(self, result: dict) -> dict:
    """Draw bounding boxes on images"""
    # Input: {"job_id": str, "total_pages": int}
    # Returns: {"job_id": str, "total_pages": int}
```

---

### Task 31: Create Text Extraction Utility
- [ ] Create `server/apps/documents/utils/text_extraction.py`
- [ ] Implement MinerU integration
- [ ] Implement `extract_text_from_region()` function
- [ ] Crop image to bounding box
- [ ] Run MinerU OCR on cropped region
- [ ] Parse text output
- [ ] Add confidence scores if available
- [ ] Test text extraction

**File to create:**
- `server/apps/documents/utils/text_extraction.py`

**Key functions:**
```python
def extract_text_from_region(image_path: str, bbox: list) -> dict:
    """Extract text from image region using MinerU"""
    # Returns: {"text": str, "confidence": float}

def crop_image_region(image_path: str, bbox: list) -> Image:
    """Crop image to bounding box coordinates"""
```

---

### Task 31: Implement Step 4 - Text Extraction
- [ ] Create `step4_text_extraction()` task
- [ ] Download images from MinIO step1/
- [ ] Download inference results from MinIO step2/
- [ ] For each page:
  - [ ] Load image
  - [ ] Load detections JSON
  - [ ] For each detected region:
    - [ ] Crop region from image
    - [ ] Run text extraction
    - [ ] Save cropped image
    - [ ] Save text + metadata as JSON
    - [ ] Upload to MinIO step4/page_XXXX/
- [ ] Update job status and current_step
- [ ] Add error handling
- [ ] Add logging
- [ ] Test with various document types

**File to modify:**
- `server/apps/documents/tasks.py`

**Task signature:**
```python
@shared_task(bind=True, max_retries=3)
def step4_text_extraction(self, result: dict) -> dict:
    """Extract text from detected regions"""
    # Input: {"job_id": str, "total_pages": int}
    # Returns: {"job_id": str, "total_pages": int}
```

---

### Task 31: Implement Step 5 - Assemble Document Graph
- [ ] Create `step5_assemble_graph()` task
- [ ] Download all element JSONs from MinIO step4/
- [ ] For each page:
  - [ ] Load all element data
  - [ ] Sort by reading order (top-to-bottom, left-to-right)
  - [ ] Create DocumentElement records in database
- [ ] Assemble complete document graph JSON
- [ ] Calculate metadata and statistics
- [ ] Store graph in DocumentJob.document_graph
- [ ] Update job status to COMPLETED
- [ ] Set processing_completed_at timestamp
- [ ] Add error handling
- [ ] Add logging
- [ ] Test graph assembly

**File to modify:**
- `server/apps/documents/tasks.py`

**Task signature:**
```python
@shared_task(bind=True, max_retries=3)
def step5_assemble_graph(self, result: dict) -> dict:
    """Assemble final document graph"""
    # Input: {"job_id": str, "total_pages": int}
    # Returns: {"job_id": str, "status": "COMPLETED"}
```

---

## Phase 7: Integration & Testing

### Task 31: Register API Router
- [ ] Import documents router in `server/api.py`
- [ ] Add router with prefix `/documents/`
- [ ] Test API documentation at `/api/docs`
- [ ] Verify all endpoints visible
- [ ] Test authentication

**File to modify:**
- `server/api.py`

**Code:**
```python
from server.apps.documents.api import router as documents_router
api.add_router("/documents/", documents_router)
```

---

### Task 31: Create Django Admin Interface
- [ ] Register DocumentJob in admin
- [ ] Add list_display fields
- [ ] Add filters (status, created_at)
- [ ] Add search fields
- [ ] Register DocumentElement in admin
- [ ] Add inline elements in DocumentJob admin
- [ ] Add readonly fields
- [ ] Test admin interface

**File to modify:**
- `server/apps/documents/admin.py`

---

### Task 31: Add Error Handling and Retries
- [ ] Review all tasks for error handling
- [ ] Add retry logic for transient errors
- [ ] Add proper exception logging
- [ ] Update job status on failures
- [ ] Store error messages in database
- [ ] Add error callbacks for pipeline
- [ ] Test failure scenarios

**Files to modify:**
- `server/apps/documents/tasks.py`
- All utility modules

---

### Task 31: Create Integration Tests
- [ ] Create test fixtures (sample PDFs)
- [ ] Test upload endpoint
- [ ] Test status endpoint
- [ ] Test results endpoint
- [ ] Test full pipeline end-to-end
- [ ] Test error scenarios
- [ ] Test MinIO operations
- [ ] Test model inference
- [ ] Run test suite

**Directory to create:**
- `tests/test_apps/test_documents/`

**Files to create:**
- `tests/test_apps/test_documents/test_api.py`
- `tests/test_apps/test_documents/test_tasks.py`
- `tests/test_apps/test_documents/test_models.py`

---

### Task 31: Documentation and Deployment
- [ ] Update README.md with setup instructions
- [ ] Document API endpoints
- [ ] Document environment variables
- [ ] Create deployment guide
- [ ] Document MinIO setup
- [ ] Document model setup
- [ ] Add example API calls
- [ ] Add troubleshooting section
- [ ] Create CHANGELOG entry

**Files to update:**
- `README.md`
- `CHANGELOG.md`

---

## Testing Checklist

### Unit Tests
- [ ] Test DocumentJob model creation
- [ ] Test DocumentElement model creation
- [ ] Test MinIO client operations
- [ ] Test storage path generation
- [ ] Test image preprocessing
- [ ] Test model loading
- [ ] Test text extraction

### Integration Tests
- [ ] Test full pipeline with 1-page PDF
- [ ] Test full pipeline with multi-page PDF
- [ ] Test upload with invalid file
- [ ] Test upload with oversized file
- [ ] Test status endpoint for non-existent job
- [ ] Test results endpoint for pending job
- [ ] Test results endpoint for failed job
- [ ] Test concurrent job processing

### Performance Tests
- [ ] Test with 10-page PDF
- [ ] Test with 50-page PDF
- [ ] Test with 100-page PDF
- [ ] Test concurrent uploads (10 simultaneous)
- [ ] Test memory usage
- [ ] Test MinIO bandwidth

---

## Dependencies Summary

### New Dependencies to Install
```toml
PyMuPDF = ">=1.24.0"         # PDF to image conversion
Pillow = ">=10.0.0"          # Image processing
minio = ">=7.2.0"            # Object storage client
python-magic = ">=0.4.27"    # File type detection
torch = ">=2.0.0"            # ML inference
torchvision = ">=0.15.0"     # Vision utilities
magic-pdf = ">=0.7.0"        # MinerU text extraction
```

### Existing Dependencies to Keep
- Django
- django-ninja
- Celery
- Redis
- PostgreSQL (psycopg)

---

## Configuration Summary

### Environment Variables
```bash
# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=document-processing
MINIO_USE_SSL=false

# Processing
MAX_PDF_SIZE_MB=50
PDF_TO_IMAGE_DPI=150
ML_MODEL_PATH=mlmodels/doclayout-yolo.mlpkg/doclayout_yolo_model.pt
ML_CONFIDENCE_THRESHOLD=0.25
ML_IOU_THRESHOLD=0.45
ML_DEVICE=cpu
```

### Docker Services Required
- PostgreSQL (pgvector)
- Redis
- RabbitMQ
- MinIO

---

## Progress Tracking

**Total Tasks**: 30
**Completed**: 0
**In Progress**: 0
**Remaining**: 30

**Estimated Time**: 12-16 hours
**Phase 1-3**: 3-4 hours
**Phase 4-5**: 4-5 hours
**Phase 6-7**: 5-7 hours

---

## Notes

- Model files must exist in `mlmodels/doclayout-yolo.mlpkg/` before starting Phase 5
- MinIO must be running before testing any tasks
- Celery workers must be running for background processing
- GPU is optional but recommended for ML inference (faster)
- MinerU setup may require additional system dependencies
