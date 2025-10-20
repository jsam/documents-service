# POINTS OCR GPU Requirement

## Overview

The POINTS-Reader OCR step (`step6_points_ocr`) **requires GPU/CUDA support** and cannot run on CPU-only environments.

## Technical Details

### Why GPU is Required

The POINTS-Reader model uses the Qwen2VL vision encoder, which has a known incompatibility when running on CPU:

1. **Position Embeddings Issue**: The rotary position embedding layer in `Qwen2VLAttention` returns `None` on CPU, causing inference to fail with:
   ```
   TypeError: cannot unpack non-iterable NoneType object
   ```

2. **Device Placement**: Even with patching to replace `.cuda()` calls with `.to(self.device)`, the underlying transformers library has hardcoded assumptions about GPU availability in the Qwen2VL attention mechanism.

3. **Attempted Workarounds That Failed**:
   - Using `torch.float32` instead of `bfloat16`
   - Explicit device mapping with `device_map={'': 'cpu'}`
   - Patching flash_attention to use eager attention
   - Replacing `.cuda()` calls with `.to(self.device)`

## Development Setup (macOS without CUDA)

### Option 1: Skip POINTS OCR (Recommended)

The service uses Docker Compose profiles to make the POINTS worker optional:

```bash
# Start all services EXCEPT celery-points-ml
docker-compose -f docker-compose.app.yml up -d

# The celery-points-ml service has profile: with-gpu
# It won't start unless explicitly requested
```

The document processing pipeline will:
- Complete all other steps successfully (PDF to images, layout detection, bounding boxes, text extraction, graph assembly)
- Skip the POINTS OCR step
- Mark the job as complete without OCR results

### Option 2: Use GPU Profile (Requires CUDA)

If you have a CUDA-enabled GPU:

```bash
docker-compose -f docker-compose.app.yml --profile with-gpu up -d
```

## Production Setup (with GPU)

### Requirements

1. **CUDA-enabled GPU** (NVIDIA)
2. **nvidia-docker runtime** installed
3. **CUDA drivers** compatible with PyTorch 2.8.0

### Configuration

1. Uncomment the GPU reservations in `docker-compose.app.yml`:

```yaml
celery-points-ml:
  environment:
    - POINTS_USE_GPU=true
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
      limits:
        memory: 8G
```

2. Start with GPU profile:

```bash
docker-compose -f docker-compose.app.yml --profile with-gpu up -d
```

### Verification

Check that the worker loaded successfully:

```bash
docker logs documents-service-celery-points-ml-1
```

You should see:
```
[POINTS] Loading model on GPU with device_map=auto
[POINTS] Model loaded successfully
```

## Error Handling

If POINTS_USE_GPU=false (CPU mode), the worker will immediately fail with a clear error message:

```
RuntimeError: POINTS OCR requires GPU. Set POINTS_USE_GPU=true in production with CUDA support.
```

This is intentional to prevent silent failures and make the GPU requirement explicit.

## Alternative Solutions (Future)

If CPU support becomes critical, consider:

1. **Using a different OCR engine** for CPU environments (e.g., EasyOCR, Tesseract, PaddleOCR)
2. **Creating a hybrid approach**: Use GPU for POINTS when available, fall back to alternative OCR on CPU
3. **Waiting for Qwen2VL fixes**: Monitor the transformers library for CPU compatibility improvements
4. **Using ONNX Runtime**: Convert the model to ONNX format which may have better CPU support

## Files Modified

- `server/apps/documents/tasks/step6_points_ocr.py`: Added GPU requirement check in `load_points_model()`
- `docker-compose.app.yml`: Added profile for GPU-only services, updated documentation
- `scripts/patch_points_model.py`: Enhanced patching for vision encoder
- `docker/django/start_points_worker.sh`: Pre-patches model files at worker startup
