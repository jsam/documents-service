# Step Naming Refactor

## Overview

Removed the `stepX_` prefix from all task files, functions, and storage paths for cleaner, more descriptive naming.

## Changes Made

### 1. Task Files Renamed

| Old Name | New Name |
|----------|----------|
| `step1_pdf_to_images.py` | `pdf_to_images.py` |
| `step2_ml_inference.py` | `ml_inference.py` |
| `step3_draw_bounding_boxes.py` | `draw_bounding_boxes.py` |
| `step4_text_extraction.py` | `text_extraction.py` |
| `step5_assemble_graph.py` | `assemble_graph.py` |
| `step6_ocr_processing.py` | `ocr_processing.py` |

### 2. Function Names Updated

| Old Name | New Name |
|----------|----------|
| `execute_step1()` | `execute_pdf_to_images()` |
| `execute_step2()` | `execute_ml_inference()` |
| `execute_step3()` | `execute_draw_bounding_boxes()` |
| `execute_step4()` | `execute_text_extraction()` |
| `execute_step5()` | `execute_assemble_graph()` |
| `execute_step6()` | `execute_ocr_processing()` |

### 3. Celery Task Names Updated

| Old Name | New Name |
|----------|----------|
| `step1_pdf_to_images` | `pdf_to_images` |
| `step2_ml_inference` | `ml_inference` |
| `step3_draw_bounding_boxes` | `draw_bounding_boxes` |
| `step4_text_extraction` | `text_extraction` |
| `step5_assemble_graph` | `assemble_graph` |
| `step6_ocr_processing` | `ocr_processing` |

### 4. Storage Path Functions Renamed

In `server/apps/documents/utils/storage.py`:

| Old Name | New Name |
|----------|----------|
| `get_step1_images_path()` | `get_images_path()` |
| `get_step2_detections_path()` | `get_detections_path()` |
| `get_step3_annotated_path()` | `get_annotated_path()` |
| `get_step4_text_path()` | `get_text_extraction_path()` |
| `get_step5_graph_path()` | `get_graph_path()` |
| `get_step6_ocr_path()` | `get_ocr_path()` |

### 5. MinIO Storage Paths (Unchanged)

The actual folder names in MinIO remain unchanged for backward compatibility:
- `jobs/{job_id}/PDF to Images/`
- `jobs/{job_id}/ML Inference/`
- `jobs/{job_id}/Draw Bounding Boxes/`
- `jobs/{job_id}/Text Extraction/`
- `jobs/{job_id}/Assemble Graph/`
- `jobs/{job_id}/OCR Processing/`

## Files Modified

1. **Task Files** (6 files renamed + internal functions updated)
   - `server/apps/documents/tasks/pdf_to_images.py`
   - `server/apps/documents/tasks/ml_inference.py`
   - `server/apps/documents/tasks/draw_bounding_boxes.py`
   - `server/apps/documents/tasks/text_extraction.py`
   - `server/apps/documents/tasks/assemble_graph.py`
   - `server/apps/documents/tasks/ocr_processing.py`

2. **Pipeline** (updated imports and task references)
   - `server/apps/documents/tasks/pipeline.py`

3. **Storage Utilities** (updated function names)
   - `server/apps/documents/utils/storage.py`

4. **Models** (updated comment)
   - `server/apps/documents/models.py`

## Verification

All changes verified:
- ✅ Python syntax check passed
- ✅ All imports updated
- ✅ Services restarted successfully
- ✅ Celery registered all tasks with new names
- ✅ No broken references found

## Impact

- **Backward Compatibility**: MinIO folder paths remain unchanged, so existing jobs are unaffected
- **Code Clarity**: Task names now clearly describe what they do
- **Maintainability**: Easier to understand the codebase without numeric prefixes
- **No Data Migration Required**: Database models reference step names by constant (e.g., `PDF_TO_IMAGES`), which remain unchanged

## Registered Tasks (After Refactor)

```
. server.apps.documents.tasks.pipeline.assemble_graph
. server.apps.documents.tasks.pipeline.draw_bounding_boxes
. server.apps.documents.tasks.pipeline.ml_inference
. server.apps.documents.tasks.pipeline.ocr_processing
. server.apps.documents.tasks.pipeline.pdf_to_images
. server.apps.documents.tasks.pipeline.process_document_pipeline
. server.apps.documents.tasks.pipeline.text_extraction
```
