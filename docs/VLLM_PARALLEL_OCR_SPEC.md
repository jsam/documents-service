# vLLM + Parallel Processing Implementation Specification

**Document Version:** 1.0  
**Date:** 2025-10-27  
**Author:** System Architecture Team  
**Status:** Proposal

---

## Executive Summary

This specification proposes migrating from the current PyTorch-based DeepSeek-OCR implementation to a vLLM-backed architecture with parallel page processing. Expected improvements:

- **Performance:** 10-15x faster OCR processing
- **Throughput:** 2,500 tokens/s on RTX 4090 (vs ~150 tokens/s current)
- **GPU Utilization:** 70-85% (vs 23% current)
- **Cost:** Better resource utilization, lower processing time per document

---

## 1. Current Architecture Analysis

### 1.1 Current Implementation

```
┌─────────────────────────────────────────┐
│   Celery Worker (ocr-worker)            │
│                                          │
│   ┌──────────────────────────────────┐  │
│   │  PyTorch + Transformers          │  │
│   │  - Load model per worker         │  │
│   │  - Sequential page processing    │  │
│   │  - File-based I/O                │  │
│   └──────────────────────────────────┘  │
│                                          │
│   GPU: RTX 4090 (23% util)              │
└─────────────────────────────────────────┘
```

### 1.2 Performance Bottlenecks

| Issue | Impact | Root Cause |
|-------|--------|------------|
| Sequential processing | 5-8x slower | `for page in pages:` loop |
| Crop mode enabled | 6-8x slower | `crop_mode=True` creates multiple crops |
| Low GPU utilization | Wasted capacity | Single-image inference |
| Cold start overhead | 15-20s | Model loads per task |

---

## 2. Proposed Architecture: vLLM + Parallel Processing

### 2.1 High-Level Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    vLLM Inference Server                        │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │  Continuous Batching Engine                              │ │
│   │  - Dynamic batching of requests                          │ │
│   │  - Paged Attention (KV cache optimization)               │ │
│   │  - Automatic memory management                           │ │
│   └──────────────────────────────────────────────────────────┘ │
│                                                                  │
│   DeepSeek-OCR Model (persistent, warm)                        │
│   GPU: RTX 4090 (target 70-85% util)                          │
└────────────────────────────────────────────────────────────────┘
                              ▲
                              │ HTTP/gRPC requests
                              │
┌─────────────────────────────┴──────────────────────────────────┐
│                  Celery Workers (Multiple)                       │
│                                                                  │
│  Worker 1          Worker 2          Worker 3                  │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐             │
│  │ Page 1-2 │      │ Page 3-4 │      │ Page 5-6 │             │
│  └──────────┘      └──────────┘      └──────────┘             │
│       │                 │                 │                     │
│       └─────────────────┴─────────────────┘                    │
│                  Parallel Execution                             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Design

#### 2.2.1 vLLM Inference Server

**Purpose:** Centralized, persistent model serving with optimized batching

**Key Features:**
- Single model instance serves all workers
- Automatic request batching and scheduling
- Paged attention for memory efficiency
- OpenAI-compatible API

**Configuration:**
```yaml
model: deepseek-ai/DeepSeek-OCR
dtype: bfloat16
gpu_memory_utilization: 0.90
max_model_len: 8192
tensor_parallel_size: 1
trust_remote_code: true
disable_log_stats: false
```

**Resource Requirements:**
- GPU VRAM: ~4-6GB (model) + 12-16GB (KV cache + batches)
- CPU: 4 cores
- RAM: 8GB

#### 2.2.2 Celery Worker Pool

**Purpose:** Parallel page processing with load distribution

**Configuration:**
```python
# New queue: ocr_page_processing
CELERY_ROUTES = {
    'server.apps.documents.tasks.pipeline.process_single_page_ocr': {
        'queue': 'ocr_page_processing'
    }
}

# Worker concurrency
CELERY_WORKER_CONCURRENCY = 4  # Tune based on RTX 4090 capacity
```

**Worker Responsibilities:**
1. Fetch page image from MinIO
2. Prepare request payload
3. Call vLLM HTTP API
4. Store result back to MinIO
5. Update progress tracking

---

## 3. Implementation Plan

### 3.1 Phase 1: vLLM Server Setup (Week 1)

#### 3.1.1 Docker Service Addition

**File:** `docker-compose.app.yml`

```yaml
vllm-server:
  image: vllm/vllm-openai:v0.8.5
  command: >
    --model deepseek-ai/DeepSeek-OCR
    --dtype bfloat16
    --gpu-memory-utilization 0.90
    --max-model-len 8192
    --trust-remote-code
    --port 8000
  environment:
    - CUDA_VISIBLE_DEVICES=0
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
      limits:
        memory: 20G
  networks: [web-net]
  ports:
    - "8000:8000"
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 120s
```

#### 3.1.2 Dependencies

**File:** `pyproject.toml`

```toml
dependencies = [
    # ... existing deps ...
    "openai>=1.0.0",  # vLLM client compatibility
    "httpx>=0.24.0",  # Async HTTP client
]
```

### 3.2 Phase 2: Client Implementation (Week 1-2)

#### 3.2.1 vLLM Client Wrapper

**File:** `server/apps/documents/utils/vllm_client.py`

```python
import base64
import httpx
from typing import List, Dict
from PIL import Image
from io import BytesIO

class VLLMClient:
    def __init__(self, base_url: str = "http://vllm-server:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    async def process_image(
        self, 
        image: Image.Image, 
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown. "
    ) -> str:
        """Process single image with DeepSeek-OCR."""
        image_b64 = self._encode_image(image)
        
        payload = {
            "model": "deepseek-ai/DeepSeek-OCR",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.0,
        }
        
        response = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    async def process_batch(
        self, 
        images: List[Image.Image], 
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown. "
    ) -> List[str]:
        """Process multiple images in parallel."""
        import asyncio
        tasks = [self.process_image(img, prompt) for img in images]
        return await asyncio.gather(*tasks)
```

#### 3.2.2 Single Page Processing Task

**File:** `server/apps/documents/tasks/step6_ocr_processing_vllm.py`

```python
import asyncio
from celery import shared_task
from PIL import Image
from io import BytesIO
from uuid import UUID
import json

from server.apps.documents.utils.vllm_client import VLLMClient
from server.apps.documents.utils.minio_client import download_file, upload_file
from server.apps.documents.utils.storage import (
    get_page_image_path,
    get_ocr_result_path,
)
from server.apps.documents.models import DocumentJob, ProcessingStep

import logging
logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, queue='ocr_page_processing')
def process_single_page_ocr(self, job_id: str, page_num: int):
    """Process a single page with vLLM-backed DeepSeek-OCR."""
    
    job = DocumentJob.objects.get(id=UUID(job_id))
    
    logger.info(f'[VLLM-OCR] Processing page {page_num} for job {job_id}')
    
    try:
        # Download page image
        page_image_path = get_page_image_path(job.id, page_num)
        image_bytes = download_file(job.minio_bucket, page_image_path)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Process with vLLM
        client = VLLMClient()
        prompt = '<image>\n<|grounding|>Convert the document to markdown. '
        
        # Run async in sync context
        response = asyncio.run(client.process_image(image, prompt))
        
        logger.info(f'[VLLM-OCR] Page {page_num} processed: {len(response)} chars')
        
        # Save result
        result_data = {
            'page_number': page_num,
            'prompt': prompt,
            'response': response
        }
        
        result_path = get_ocr_result_path(job.id, page_num)
        result_json = json.dumps(result_data, indent=2).encode('utf-8')
        
        upload_file(
            bucket=job.minio_bucket,
            object_name=result_path,
            file_data=result_json,
            content_type='application/json',
        )
        
        return {'page': page_num, 'success': True, 'result_path': result_path}
        
    except Exception as e:
        logger.error(f'[VLLM-OCR] Page {page_num} failed: {e}', exc_info=True)
        raise


@shared_task(bind=True, max_retries=3, queue='ocr_processing')
def step6_ocr_processing_parallel(self, previous_result, job_id: str):
    """Coordinate parallel OCR processing of all pages."""
    from celery import chord, group
    
    job = DocumentJob.objects.get(id=UUID(job_id))
    step = ProcessingStep.objects.get(job=job, step_name='OCR_PROCESSING')
    
    step.mark_in_progress(self.request.id)
    job.current_step = 'OCR_PROCESSING'
    job.save(update_fields=['current_step'])
    
    total_pages = job.total_pages
    step.update_progress(0, total_pages)
    
    logger.info(f'[VLLM-OCR] Starting parallel processing of {total_pages} pages')
    
    # Create parallel tasks for all pages
    page_tasks = group(
        process_single_page_ocr.s(job_id, page_num)
        for page_num in range(1, total_pages + 1)
    )
    
    # Aggregate results when all pages complete
    callback = finalize_ocr_processing.s(job_id)
    
    return chord(page_tasks)(callback)


@shared_task
def finalize_ocr_processing(results, job_id: str):
    """Finalize OCR processing after all pages complete."""
    job = DocumentJob.objects.get(id=UUID(job_id))
    step = ProcessingStep.objects.get(job=job, step_name='OCR_PROCESSING')
    
    # Extract result paths
    result_paths = [r['result_path'] for r in results if r['success']]
    
    step.status = ProcessingStep.STATUS_COMPLETED
    step.completed_at = timezone.now()
    step.result_data = {
        'result_paths': result_paths,
        'total_pages': len(results)
    }
    step.save(update_fields=['status', 'completed_at', 'result_data'])
    
    logger.info(f'[VLLM-OCR] Job {job_id} completed: {len(result_paths)} pages')
    
    return {'success': True, 'total_pages': len(results), 'result_paths': result_paths}
```

### 3.3 Phase 3: Pipeline Integration (Week 2)

#### 3.3.1 Update Pipeline

**File:** `server/apps/documents/tasks/pipeline.py`

```python
# Import new task
from server.apps.documents.tasks.step6_ocr_processing_vllm import (
    step6_ocr_processing_parallel
)

# Update pipeline task reference
@shared_task(bind=True, max_retries=3, queue='ocr_processing')
def step6_ocr_processing(self, previous_result, job_id: str):
    """Route to vLLM-based parallel OCR processing."""
    return step6_ocr_processing_parallel(self, previous_result, job_id)
```

### 3.4 Phase 4: Worker Configuration (Week 2)

#### 3.4.1 New Worker Service

**File:** `docker-compose.app.yml`

```yaml
# Replace old ocr-worker with new configuration
ocr-page-workers:
  <<: *django-service
  command: celery -A server worker -l INFO -Q ocr_page_processing --concurrency=4 --prefetch-multiplier=2
  environment:
    - VLLM_SERVER_URL=http://vllm-server:8000
  depends_on:
    vllm-server:
      condition: service_healthy
    migrate:
      condition: service_completed_successfully
  deploy:
    replicas: 2  # Can scale horizontally
  healthcheck:
    test: celery -A server inspect ping -d celery@$$HOSTNAME
    interval: 60s
    timeout: 10s
    retries: 3

# Keep coordinator worker
ocr-coordinator:
  <<: *django-service
  command: celery -A server worker -l INFO -Q ocr_processing --concurrency=1
  depends_on:
    migrate:
      condition: service_completed_successfully
```

---

## 4. Performance Optimization

### 4.1 Concurrency Tuning

**RTX 4090 Capacity Analysis:**
- Total VRAM: 24GB
- Model size: ~4GB
- KV cache per request: ~500MB-1GB (depends on image size)
- **Optimal batch size:** 8-12 concurrent requests

**Worker Configuration:**
```python
# Celery worker concurrency per container
CELERY_WORKER_CONCURRENCY = 4

# Number of worker containers
WORKER_REPLICAS = 2

# Total concurrent pages: 4 × 2 = 8 pages
```

### 4.2 vLLM Configuration Tuning

```yaml
vllm-server:
  command: >
    --model deepseek-ai/DeepSeek-OCR
    --dtype bfloat16
    --gpu-memory-utilization 0.85
    --max-model-len 8192
    --max-num-seqs 8            # Max concurrent requests
    --max-num-batched-tokens 8192
    --trust-remote-code
```

### 4.3 Image Processing Parameters

**Optimal settings for speed vs quality:**
```python
# For standard documents (recommended)
base_size = 640
image_size = 640
crop_mode = False  # Disable multi-crop for speed

# For high-quality scans (if needed)
base_size = 1024
image_size = 1024
crop_mode = False  # Still avoid multi-crop
```

---

## 5. Monitoring & Observability

### 5.1 Metrics to Track

| Metric | Target | Current | Tool |
|--------|--------|---------|------|
| Pages/second | 2-4 | 0.02 | Prometheus |
| GPU Utilization | 70-85% | 23% | nvidia-smi |
| Request latency (p95) | <15s | ~45s | vLLM metrics |
| Queue depth | <10 | N/A | Celery |
| Memory usage | <22GB | ~3GB | nvidia-smi |

### 5.2 vLLM Metrics Endpoint

```yaml
# Expose vLLM metrics
vllm-server:
  ports:
    - "8000:8000"  # API
    - "9090:9090"  # Metrics (Prometheus)
```

### 5.3 Logging

```python
# Enhanced logging for debugging
logger.info('[VLLM-OCR] Page %d: Image size=%s, Response time=%.2fs, Response length=%d',
            page_num, image.size, response_time, len(response))
```

---

## 6. Rollout Strategy

### 6.1 Feature Flag

```python
# server/settings/components/common.py
USE_VLLM_OCR = os.environ.get('USE_VLLM_OCR', 'false').lower() == 'true'

# server/apps/documents/tasks/pipeline.py
if USE_VLLM_OCR:
    from .step6_ocr_processing_vllm import step6_ocr_processing_parallel as ocr_task
else:
    from .step6_ocr_processing import execute_step6 as ocr_task
```

### 6.2 Migration Plan

1. **Week 1:** Deploy vLLM server alongside existing system
2. **Week 1-2:** Implement and test new client code
3. **Week 2:** Run A/B test (10% traffic to vLLM)
4. **Week 3:** Gradually increase to 50%
5. **Week 4:** Full migration, deprecate old implementation

### 6.3 Rollback Plan

```bash
# Emergency rollback
docker-compose -f docker-compose.app.yml up -d --scale vllm-server=0
export USE_VLLM_OCR=false
docker-compose -f docker-compose.app.yml restart ocr-coordinator
```

---

## 7. Resource Requirements

### 7.1 Hardware

| Component | Current | Proposed | Change |
|-----------|---------|----------|--------|
| GPU VRAM | 2.7GB (11%) | 18-20GB (80%) | +650% |
| System RAM | 4GB | 12GB | +8GB |
| CPU cores | 2 | 8 | +6 cores |

### 7.2 Cost Analysis

**Time savings per 100-page document:**
- Current: ~75 minutes (45s/page)
- Proposed: ~5-7 minutes (3-4s/page)
- **Savings: ~68 minutes** (91% reduction)

**Cost per document (assuming $0.50/GPU-hour):**
- Current: $0.625
- Proposed: $0.058
- **Savings: $0.567** (91% reduction)

---

## 8. Testing & Validation

### 8.1 Unit Tests

```python
# tests/test_vllm_ocr.py
def test_vllm_client_single_image():
    client = VLLMClient()
    image = Image.new('RGB', (640, 480), color='white')
    result = asyncio.run(client.process_image(image))
    assert len(result) > 0

def test_parallel_page_processing():
    job_id = create_test_job(pages=5)
    result = step6_ocr_processing_parallel.apply(args=[None, str(job_id)])
    assert result['total_pages'] == 5
```

### 8.2 Integration Tests

```python
# tests/integration/test_ocr_pipeline.py
def test_end_to_end_ocr_pipeline():
    # Upload test PDF
    job = upload_pdf('test_document.pdf')
    
    # Run pipeline
    process_document_pipeline.apply_async(args=[str(job.id)])
    
    # Wait for completion
    wait_for_job_completion(job.id, timeout=300)
    
    # Verify OCR results
    step = ProcessingStep.objects.get(job=job, step_name='OCR_PROCESSING')
    assert step.status == ProcessingStep.STATUS_COMPLETED
    assert len(step.result_data['result_paths']) == job.total_pages
```

### 8.3 Load Tests

```bash
# Locust load test
# tests/load/test_ocr_throughput.py
class OCRUser(HttpUser):
    @task
    def upload_and_process(self):
        # Upload PDF
        # Monitor processing time
        # Verify results
```

---

## 9. Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| vLLM OOM crashes | High | Medium | GPU memory monitoring, automatic restarts |
| Network latency to vLLM | Medium | Low | Deploy on same host, use Unix sockets |
| Cold start delays | Low | Low | Health checks, pre-warming |
| Concurrent page failures | Medium | Medium | Per-page retry logic, exponential backoff |
| vLLM version incompatibility | High | Low | Pin versions, test upgrades in staging |

---

## 10. Success Criteria

### 10.1 Performance Targets

- ✅ 10x faster processing (45s → 4-5s per page)
- ✅ 70%+ GPU utilization
- ✅ 2000+ tokens/second throughput
- ✅ <5% failure rate

### 10.2 Quality Targets

- ✅ OCR accuracy parity with current implementation
- ✅ No data loss or corruption
- ✅ Consistent results across sequential/parallel execution

### 10.3 Operational Targets

- ✅ <10 minutes deployment time
- ✅ Automated monitoring and alerting
- ✅ <1 hour MTTR (Mean Time To Recovery)

---

## 11. Future Enhancements

### 11.1 Multi-GPU Support

```yaml
vllm-server:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 2  # Tensor parallelism across 2 GPUs
```

### 11.2 Model Quantization

```bash
# INT8 quantization for 2x speed boost
--quantization awq
```

### 11.3 Dynamic Batching Optimization

```python
# Adaptive batch sizing based on image dimensions
def calculate_optimal_batch_size(image_sizes):
    total_pixels = sum(w * h for w, h in image_sizes)
    return min(12, int(50_000_000 / total_pixels))
```

---

## 12. Appendix

### 12.1 References

- vLLM Documentation: https://docs.vllm.ai/
- DeepSeek-OCR GitHub: https://github.com/deepseek-ai/DeepSeek-OCR
- vLLM DeepSeek-OCR Recipe: https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html

### 12.2 Related Documents

- `docs/OCR_GPU_REQUIREMENT.md` - GPU requirements
- `server/apps/documents/tasks/step6_ocr_processing.py` - Current implementation
- `docker-compose.app.yml` - Docker services configuration

### 12.3 Glossary

- **vLLM:** High-throughput LLM inference engine with PagedAttention
- **Continuous Batching:** Dynamic batching technique that adds/removes requests during execution
- **Paged Attention:** Memory-efficient attention mechanism that reduces fragmentation
- **KV Cache:** Key-Value cache for attention mechanism optimization

---

**End of Specification**
