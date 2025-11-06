# OCR Parallel Processing - Root Cause Analysis & Fix

## Problem Statement

The OCR processing task (`step6_ocr_processing`) was not starting immediately after PDF-to-images completion. It appeared to remain in PENDING status while the ML chain (steps 2-4) ran sequentially.

## Root Cause

**The task WAS actually running in parallel, but it was failing silently due to a missing dependency.**

### Evidence from Logs

```
celery-1  | [2025-10-27 20:15:04,332: ERROR/ForkPoolWorker-2] Chord '123ba43c-1c77-40cd-bef7-8e9135cd8d5c' raised: ChordError('Dependency 930f0504-620d-4cbf-8081-b6c27767358c raised ModuleNotFoundError("No module named \'openai\'")')
```

The task with ID `930f0504-620d-4cbf-8081-b6c27767358c` was `step6_ocr_processing`, and it was being invoked but immediately crashed because the `openai` Python package was not installed in the container.

### Why It Appeared Sequential

1. The ProcessingStep status tracking requires the task to call `step.mark_in_progress()` at the beginning
2. The task was crashing during import before it could update its status
3. The UI showed it as PENDING because the status was never updated from the initial state
4. The chord was waiting for all tasks to complete, but one was failing silently

## Pipeline Architecture

The current pipeline structure is:

```python
task_pipeline = chain(
    step1_pdf_to_images.s(job_id),           # Step 1: Convert PDF to images
    chord(
        group(                                # These run in PARALLEL:
            chain(                             # ML Chain (sequential within):
                step2_ml_inference.s(job_id),    # - ML inference
                step3_draw_bounding_boxes.s(...),# - Draw boxes
                step4_text_extraction.s(...),    # - Extract text
            ),
            step6_ocr_processing.si(job_id)   # OCR task (independent)
        ),
        step5_assemble_graph.s(job_id),      # Step 5: Runs after ALL above complete
    ),
)
```

### Key Points:

1. **`chord(group(...), callback)`** - Waits for ALL tasks in the group to complete, then runs the callback
2. **`group(task1, task2)`** - Runs task1 and task2 in PARALLEL immediately
3. **`.si(job_id)`** - Immutable signature (doesn't take previous result as input)
4. **`.s(job_id)`** - Normal signature (takes previous result)

## The Fix

### 1. Added `openai` dependency to pyproject.toml

```toml
dependencies = [
    ...
    "openai>=1.0.0",
]
```

### 2. Installed in running containers (temporary fix)

```bash
docker compose -f docker-compose.app.yml exec celery bash -c "cd /code && uv pip install openai"
```

### 3. Proper fix: Rebuild containers

To ensure the dependency is permanent:

```bash
docker compose -f docker-compose.app.yml down
docker compose -f docker-compose.app.yml build
docker compose -f docker-compose.app.yml up -d
```

## Changes Made to OCR Task

### File: `server/apps/documents/tasks/step6_ocr_processing.py`

**Key changes:**

1. **Replaced httpx with OpenAI client:**
   ```python
   from openai import OpenAI
   
   client = OpenAI(
       base_url=f'{DEEPSEEK_OCR_SERVER_URL}/v1',
       api_key='dummy-key',
   )
   ```

2. **Parallel processing with ThreadPoolExecutor:**
   ```python
   with ThreadPoolExecutor(max_workers=max_workers) as executor:
       futures = {
           executor.submit(process_single_page, ...): page_num
           for page_num in range(1, total_pages + 1)
       }
   ```

3. **Correct OpenAI vision message format:**
   ```python
   messages=[
       {
           'role': 'user',
           'content': [
               {
                   'type': 'image_url',
                   'image_url': {'url': f'data:image/png;base64,{image_b64}'},
               },
               {
                   'type': 'text',
                   'text': '<|grounding|>Convert the document to markdown.',
               },
           ],
       },
   ]
   ```

4. **Assembles markdown output:**
   - Processes all pages in parallel
   - Combines into single `full_document.md` file
   - Each page separated with `---` markdown divider

### File: `server/apps/documents/tasks/pipeline.py`

**Changes:**

1. **Removed queue specification:**
   ```python
   @shared_task(bind=True, max_retries=3)  # No queue='ocr_processing'
   def step6_ocr_processing(self, job_id: str):
   ```

2. **Made task signature immutable:**
   ```python
   ocr_task = step6_ocr_processing.si(job_id)  # .si() not .s()
   ```

3. **Added debug logging:**
   ```python
   logger.info(f'[PIPELINE] Building pipeline for job {job_id}')
   logger.info(f'[PIPELINE] step6_ocr_processing STARTED for job {job_id}')
   ```

### File: `server/celery.py`

**Changes:**

1. **Removed ocr_processing queue:**
   ```python
   app.conf.task_queues = (
       Queue('celery', routing_key='celery'),
       Queue('ml_inference', routing_key='ml_inference'),
       # Removed: Queue('ocr_processing', routing_key='ocr_processing'),
   )
   ```

## Verification

After the fix, both tasks should start immediately after PDF_TO_IMAGES completes:

1. **ML_INFERENCE** - Starts on `ml_inference` queue (celery-doclayout-ml worker)
2. **OCR_PROCESSING** - Starts on `celery` queue (celery worker)

Both run in parallel, and ASSEMBLE_GRAPH waits for both to complete.

## Lessons Learned

1. **Silent failures are dangerous** - The task was failing before it could update its status
2. **Check container dependencies** - pyproject.toml changes require container rebuilds
3. **The pipeline structure was correct all along** - The chord/group setup was working as intended
4. **Always check actual error logs** - The Celery logs contained the real error

## Future Improvements

1. Add health checks that verify all required packages are installed
2. Add better error handling in task entry points to catch import errors
3. Consider adding task status updates BEFORE imports
4. Add integration tests that verify dependencies are available
