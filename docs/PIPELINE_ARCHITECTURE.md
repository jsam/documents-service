# Document Processing Pipeline Architecture

## Pipeline Overview

The document processing pipeline uses Celery with parallel execution to maximize throughput. After PDF conversion, ML inference and OCR processing run **in parallel**.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    PDF Upload                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Step 1: PDF to Images                           │
│              Convert PDF pages to PNG images                 │
│              Output: jobs/{job_id}/PDF to Images/            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
            ┌─────────┴─────────┐
            │                   │
            ▼                   ▼
┌───────────────────┐   ┌──────────────────┐
│   ML Chain        │   │   OCR Processing │
│   (Sequential)    │   │   (Parallel)     │
└───────────────────┘   └──────────────────┘
            │                   │
            │                   │
  ┌─────────┴───────┐          │
  │                 │          │
  ▼                 │          │
┌───────────────────┴─┐        │
│ Step 2: ML Inference│        │
│ DocLayout-YOLO      │        │
│ Output: ML Inference│        │
└───────────┬─────────┘        │
            │                  │
            ▼                  │
┌───────────────────────┐      │
│ Step 3: Draw Boxes    │      │
│ Annotate detections   │      │
│ Output: Draw Bounding │      │
│         Boxes         │      │
└───────────┬───────────┘      │
            │                  │
            ▼                  │
┌───────────────────────┐      │
│ Step 4: Text Extract  │      │
│ Extract text regions  │      │
│ Output: Text          │      │
│         Extraction    │      │
└───────────┬───────────┘      │
            │                  │
            │    ┌─────────────┴─────────────┐
            │    │ Step 6: OCR Processing    │
            │    │ DeepSeek-OCR via Rust     │
            │    │ Output: OCR Processing    │
            │    └─────────────┬─────────────┘
            │                  │
            └─────────┬────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │ Step 5: Assemble Graph │
         │ Combine all results    │
         │ Output: Assemble Graph │
         └────────────────────────┘
```

## Parallel Execution Details

### Celery Implementation

```python
ml_chain = chain(
    step2_ml_inference.s(job_id),
    step3_draw_bounding_boxes.s(job_id),
    step4_text_extraction.s(job_id),
)

ocr_task = step6_ocr_processing.s(job_id)

task_pipeline = chain(
    step1_pdf_to_images.s(job_id),
    chord(
        group(ml_chain, ocr_task),  # Parallel execution
        step5_assemble_graph.s(job_id)
    )
)
```

### Execution Pattern

1. **Sequential Start**: `PDF to Images` must complete first
2. **Parallel Branch**: After images are ready, two paths execute simultaneously:
   - **ML Chain**: ML Inference → Draw Boxes → Text Extraction (sequential within chain)
   - **OCR Processing**: Independent parallel path
3. **Synchronization**: `Assemble Graph` waits for both branches to complete

## MinIO Storage Structure

All output folders now match the UI display names (no "stepX" prefixes):

```
jobs/{job_id}/
├── Upload/
│   └── {original_filename}.pdf
├── PDF to Images/
│   ├── page_0001.png
│   ├── page_0002.png
│   └── ...
├── ML Inference/
│   ├── page_0001.json
│   ├── page_0002.json
│   └── ...
├── Draw Bounding Boxes/
│   ├── page_0001_annotated.png
│   ├── page_0002_annotated.png
│   └── ...
├── Text Extraction/
│   ├── page_0001_element_001.png
│   ├── page_0001_element_001.txt
│   ├── page_0001_element_002.png
│   ├── page_0001_element_002.txt
│   └── ...
├── OCR Processing/
│   ├── page_0001_ocr.json
│   ├── page_0002_ocr.json
│   └── ...
└── Assemble Graph/
    └── document_graph.json
```

## Worker Queues

### Default Queue (`celery`)
- PDF to Images
- Draw Bounding Boxes
- Text Extraction
- Assemble Graph

### ML Inference Queue (`ml_inference`)
- ML Inference (DocLayout-YOLO)
- Runs on GPU-enabled worker

### OCR Queue (`ocr_processing`)
- OCR Processing
- Runs on dedicated worker
- Sends requests to deepseek-ocr-server

## Performance Benefits

### Parallel Execution Gains

**Before (Sequential)**:
```
Total Time = PDF→Images + ML + Draw + Text + OCR + Assemble
           = 10s + 30s + 5s + 20s + 200s + 5s = 270s (4.5 minutes)
```

**After (Parallel)**:
```
ML Branch: 30s + 5s + 20s = 55s
OCR Branch: 200s (but now 3-5s with Rust server) = 5s

Total Time = PDF→Images + max(ML Branch, OCR Branch) + Assemble
           = 10s + max(55s, 5s) + 5s = 70s (1.2 minutes)
```

**Speedup**: ~3.8x faster (with old OCR) or ~5x faster (with Rust OCR)

### Scalability

Can scale workers independently:
- Scale `ml_inference` workers for more document throughput
- Scale `ocr_processing` workers for more OCR throughput
- Single `deepseek-ocr-server` handles all OCR requests efficiently

## Monitoring

### Check Pipeline Status

```python
from server.apps.documents.models import DocumentJob, ProcessingStep

job = DocumentJob.objects.get(id=job_id)
steps = ProcessingStep.objects.filter(job=job).order_by('created_at')

for step in steps:
    print(f"{step.step_name}: {step.status}")
```

### Expected Step States During Parallel Execution

```
PDF_TO_IMAGES:          COMPLETED
ML_INFERENCE:           IN_PROGRESS  ⎤
DRAW_BOUNDING_BOXES:    PENDING      ⎥ ML Chain
TEXT_EXTRACTION:        PENDING      ⎦
OCR_PROCESSING:         IN_PROGRESS  ← Parallel with ML Chain
ASSEMBLE_GRAPH:         PENDING
```

### Celery Monitoring

```bash
# View active tasks
celery -A server inspect active

# View task stats
celery -A server inspect stats

# View worker status
celery -A server status
```

## Error Handling

### Parallel Branch Failures

If either branch fails:
- The other branch continues execution
- `Assemble Graph` step fails when it tries to collect results
- Job status is marked as FAILED
- Error is recorded in `DocumentJob.error_message`

### Retry Logic

Each step has `max_retries=3`:
- Transient failures (network, GPU OOM) retry automatically
- Permanent failures (bad input, missing file) fail immediately

## Configuration

### Enable/Disable Parallel OCR

To run OCR sequentially (for debugging):

```python
# In pipeline.py
task_pipeline = chain(
    step1_pdf_to_images.s(job_id),
    step2_ml_inference.s(job_id),
    step3_draw_bounding_boxes.s(job_id),
    step4_text_extraction.s(job_id),
    step6_ocr_processing.s(job_id),
    step5_assemble_graph.s(job_id)
)
```

### Adjust Parallelism

Modify worker concurrency:

```yaml
# docker-compose.app.yml
ocr-worker:
  command: celery -A server worker -Q ocr_processing --concurrency=2

celery-doclayout-ml:
  command: celery -A server worker -Q ml_inference --concurrency=1
```

## Folder Name Mapping

The `get_step_folder()` function maps step names to UI-friendly folder names:

| Step Constant | UI Display Name | MinIO Folder Name |
|--------------|-----------------|-------------------|
| `PDF_TO_IMAGES` | PDF to Images | `PDF to Images` |
| `ML_INFERENCE` | ML Inference | `ML Inference` |
| `DRAW_BOUNDING_BOXES` | Draw Bounding Boxes | `Draw Bounding Boxes` |
| `TEXT_EXTRACTION` | Text Extraction | `Text Extraction` |
| `OCR_PROCESSING` | OCR Processing | `OCR Processing` |
| `ASSEMBLE_GRAPH` | Assemble Graph | `Assemble Graph` |

All folder names now match exactly what users see in the UI!

## Related Documentation

- [DeepSeek OCR Rust Setup](./DEEPSEEK_OCR_RUST_SETUP.md)
- [vLLM Parallel OCR Spec](./VLLM_PARALLEL_OCR_SPEC.md)
