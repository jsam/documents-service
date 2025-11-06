# DeepSeek OCR Rust Server Setup

## Overview

The document processing pipeline uses [deepseek-ocr.rs](https://github.com/TimmyOVO/deepseek-ocr.rs), a Rust implementation of DeepSeek-OCR with CUDA GPU acceleration for high-performance document OCR processing.

## Architecture

```
┌─────────────────────────────────────────────┐
│   DeepSeek OCR Rust Server                  │
│   (deepseek-ocr-server)                     │
│                                              │
│   - Loads model at startup (~6.3GB)         │
│   - CUDA GPU acceleration (f16)             │
│   - OpenAI-compatible HTTP API              │
│   - Port: 8001                              │
└─────────────────────────────────────────────┘
                    ▲
                    │ HTTP requests
                    │
┌─────────────────────────────────────────────┐
│   OCR Worker (Celery)                       │
│   - Processes pages in parallel             │
│   - Sends images via HTTP                   │
│   - No local model loading required         │
└─────────────────────────────────────────────┘
```

## CUDA GPU Configuration

### Requirements
- NVIDIA GPU with CUDA Compute Capability 7.0+
- CUDA Toolkit 12.2+
- NVIDIA Container Toolkit installed on host
- At least 16GB VRAM (20GB recommended)

### Docker Configuration

The `deepseek-ocr-server` service is configured with:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
    limits:
      memory: 20G
```

### Environment Variables

- `CUDA_VISIBLE_DEVICES=0` - GPU device selection
- `RUST_LOG=info` - Logging level
- `HF_HOME=/app/.cache/huggingface` - Model cache location

### Model Loading

The model is **loaded at startup** before the HTTP server begins accepting requests:

1. Server starts and checks for CUDA availability
2. Downloads model weights (~6.3GB) if not cached
3. Loads model onto GPU with f16 precision
4. Initializes tokenizer
5. Starts HTTP server on port 8001
6. Health check confirms model is ready

### Startup Logs

You should see output like:

```
=========================================
DeepSeek OCR Rust Server Startup
=========================================
Checking NVIDIA GPU availability...
Tesla T4, 535.104.05, 15360 MiB
✓ CUDA GPU detected

Starting server with CUDA device...
Command: /app/deepseek-ocr-server --host 0.0.0.0 --port 8001 --device cuda --dtype f16 ...

[INFO] Using configuration ... (active model `deepseek-ocr`)
[INFO] Loading model on CUDA device with f16 dtype
[INFO] Model loaded successfully
[INFO] Server ready on 0.0.0.0:8001 (deepseek-ocr)
```

## Model Caching

Model weights are persisted in a Docker volume:

```yaml
volumes:
  deepseek-model-cache:/app/.cache
```

**First run**: Downloads ~6.3GB from HuggingFace/ModelScope (takes 5-10 minutes)  
**Subsequent runs**: Uses cached model (starts in 30-60 seconds)

To clear the cache and re-download:
```bash
docker volume rm documents-service_deepseek-model-cache
```

## Configuration Options

The server is configured via command-line arguments:

| Flag | Value | Purpose |
|------|-------|---------|
| `--device` | `cuda` | Use NVIDIA GPU acceleration |
| `--dtype` | `f16` | Use FP16 precision (faster, less VRAM) |
| `--max-new-tokens` | `4096` | Maximum output tokens per request |
| `--crop-mode` | `false` | Disable multi-crop (faster, single pass) |
| `--host` | `0.0.0.0` | Bind to all interfaces |
| `--port` | `8001` | HTTP server port |

### Why `--crop-mode false`?

Crop mode creates multiple high-resolution tiles per image, which:
- ✗ 6-8x slower processing time
- ✗ Higher VRAM usage
- ✓ Better quality for complex layouts

For most documents, single-pass processing (`crop-mode=false`) provides good quality at much higher speed.

## Building the Container

```bash
# Build with CUDA support (takes 10-15 minutes first time)
docker compose -f docker-compose.app.yml build deepseek-ocr-server
```

The Dockerfile:
1. Uses `nvidia/cuda:12.2.0-devel-ubuntu22.04` for building
2. Installs Rust 1.78.0
3. Builds with `--features cuda` for GPU support
4. Creates lightweight runtime image with CUDA runtime

## Running the Server

```bash
# Start just the OCR server
docker compose -f docker-compose.app.yml up deepseek-ocr-server

# Start full stack (including OCR worker)
docker compose -f docker-compose.app.yml up -d
```

## Health Checks

The service includes a health check that verifies:
- Server is responding on port 8001
- `/v1/models` endpoint returns successfully
- Model is loaded and ready

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8001/v1/models"]
  interval: 30s
  timeout: 10s
  retries: 5
  start_period: 180s  # 3 minutes for model loading
```

## API Endpoints

### GET /v1/models
List available models.

```bash
curl http://localhost:8001/v1/models
```

### POST /v1/chat/completions
OpenAI-compatible chat completions endpoint.

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ocr",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "<image>\n<|grounding|>Convert the document to markdown."},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
      }
    ],
    "max_tokens": 4096,
    "temperature": 0.0
  }'
```

## Integration with OCR Worker

The Python OCR worker sends requests via HTTP:

```python
import httpx
from PIL import Image

client = httpx.Client(timeout=300.0)
response = client.post(
    f'{DEEPSEEK_OCR_SERVER_URL}/v1/chat/completions',
    json={
        'model': 'deepseek-ocr',
        'messages': [...],
        'max_tokens': 4096,
        'temperature': 0.0
    }
)
```

## Troubleshooting

### Model not loading
Check logs:
```bash
docker compose -f docker-compose.app.yml logs deepseek-ocr-server
```

Look for CUDA availability message and model loading progress.

### CUDA not available
Verify NVIDIA Container Toolkit:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi
```

### Out of memory
Reduce memory usage:
- Use `--dtype f16` (already configured)
- Ensure `--crop-mode false` (already configured)
- Reduce `--max-new-tokens` if needed
- Check other GPU processes: `nvidia-smi`

### Slow inference
- Verify GPU is being used (check logs for "CUDA device")
- Ensure model is cached (second run should be faster)
- Check GPU utilization: `nvidia-smi`

## Performance

Expected performance with RTX 4090:
- **Cold start**: 30-60 seconds (model loading)
- **Inference**: 3-5 seconds per page (1024x1024 image)
- **Throughput**: ~2000-2500 tokens/second
- **GPU utilization**: 70-85%

## Monitoring

Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

Monitor server logs:
```bash
docker compose -f docker-compose.app.yml logs -f deepseek-ocr-server
```

Check health status:
```bash
docker compose -f docker-compose.app.yml ps deepseek-ocr-server
```

## References

- [deepseek-ocr.rs GitHub](https://github.com/TimmyOVO/deepseek-ocr.rs)
- [DeepSeek-OCR Model](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
