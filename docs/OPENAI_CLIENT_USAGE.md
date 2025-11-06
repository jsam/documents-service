# OpenAI Client Usage for Local DeepSeek OCR API

This document describes how to use the OpenAI Python client to communicate with the local DeepSeek OCR API server running at `http://0.0.0.0:8001/v1`.

## Installation

The `openai` package is already included in the project dependencies:

```bash
uv pip install -e .
```

## Configuration

The DeepSeek OCR server runs as an OpenAI-compatible API at:
- **Base URL**: `http://0.0.0.0:8001/v1`
- **Model Name**: `deepseek-ocr`
- **API Key**: Any dummy value (e.g., `dummy-key`)

## Basic Usage

### Text-Only Request

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://0.0.0.0:8001/v1',
    api_key='dummy-key',
)

response = client.chat.completions.create(
    model='deepseek-ocr',
    messages=[
        {
            'role': 'user',
            'content': 'Hello! Can you tell me what you are?',
        },
    ],
)

print(response.choices[0].message.content)
```

### Image + Text Request (OCR)

To send an image with your prompt, use the vision-style message format with base64-encoded images:

```python
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url='http://0.0.0.0:8001/v1',
    api_key='dummy-key',
)

# Read and encode the image
image_path = Path('document.png')
with open(image_path, 'rb') as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

# Send the request
response = client.chat.completions.create(
    model='deepseek-ocr',
    messages=[
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Extract all the text you see in this image.',
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/png;base64,{image_data}',
                    },
                },
            ],
        },
    ],
)

print(response.choices[0].message.content)
```

## Supported Image Formats

The API supports standard image formats including:
- PNG: `data:image/png;base64,{base64_data}`
- JPEG: `data:image/jpeg;base64,{base64_data}`
- JPG: `data:image/jpg;base64,{base64_data}`

## Example Use Cases

### OCR Processing
```python
response = client.chat.completions.create(
    model='deepseek-ocr',
    messages=[
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Extract all text from this document.'},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{image_data}'}},
            ],
        },
    ],
)
```

### Structured Data Extraction
```python
response = client.chat.completions.create(
    model='deepseek-ocr',
    messages=[
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Extract the invoice number, date, and total amount from this invoice.',
                },
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{image_data}'}},
            ],
        },
    ],
)
```

## Testing

A test script is available at `test_openai_local.py`:

```bash
python test_openai_local.py
```

This script tests both text-only and image-based requests to the API.

## Notes

- The API server must be running before making requests
- The server uses CUDA GPU acceleration and requires GPU support
- Model weights (~6.3GB) are cached on first run
- The API is fully compatible with OpenAI's chat completion format
