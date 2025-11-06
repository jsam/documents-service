#!/usr/bin/env python3
import httpx
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def test_ocr():
    # Create a realistic test document image
    image = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(image)
    
    # Add some text content
    draw.text((50, 50), 'INVOICE', fill='black')
    draw.text((50, 100), 'Date: October 27, 2025', fill='black')
    draw.text((50, 150), 'Invoice #: INV-2025-001', fill='black')
    draw.text((50, 200), '', fill='black')
    draw.text((50, 220), 'Items:', fill='black')
    draw.text((70, 250), '1. Widget A - $25.00', fill='black')
    draw.text((70, 280), '2. Widget B - $35.00', fill='black')
    draw.text((50, 330), '', fill='black')
    draw.text((50, 350), 'Total: $60.00', fill='black')
    
    image_b64 = encode_image_to_base64(image)
    
    content = f'data:image/png;base64,{image_b64}\n\n<|grounding|>Convert the document to markdown.'
    
    payload = {
        'model': 'deepseek-ocr',
        'messages': [{
            'role': 'user',
            'content': content
        }],
        'max_tokens': 2048
    }
    
    print('=' * 60)
    print('DeepSeek OCR Rust Server - Final Integration Test')
    print('=' * 60)
    print()
    print(f'Server URL: http://localhost:8001')
    print(f'Image size: {image.size}')
    print(f'Base64 length: {len(image_b64)} bytes')
    print()
    print('Sending OCR request...')
    
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                'http://localhost:8001/v1/chat/completions',
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            text = result['choices'][0]['message']['content']
            tokens = result['usage']
            
            print()
            print('✓ OCR SUCCESS!')
            print('=' * 60)
            print('Response:')
            print('-' * 60)
            print(text)
            print('-' * 60)
            print()
            print(f'Token usage:')
            print(f'  Prompt tokens: {tokens["prompt_tokens"]}')
            print(f'  Completion tokens: {tokens["completion_tokens"]}')
            print(f'  Total tokens: {tokens["total_tokens"]}')
            print()
            print('✓ Integration test passed!')
            
            return True
            
    except Exception as e:
        print()
        print(f'✗ ERROR: {e}')
        return False

if __name__ == '__main__':
    success = test_ocr()
    exit(0 if success else 1)
