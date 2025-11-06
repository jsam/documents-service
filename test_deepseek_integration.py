import base64
import json
from io import BytesIO

import httpx
from PIL import Image


def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def test_deepseek_server_connection():
    server_url = 'http://localhost:8001'
    
    print(f'Testing connection to {server_url}...')
    
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f'{server_url}/v1/models')
            response.raise_for_status()
            print('✓ Server is reachable')
            print(f'Response: {response.json()}')
    except Exception as e:
        print(f'✗ Server connection failed: {e}')
        return False
    
    return True


def test_deepseek_ocr_inference():
    server_url = 'http://localhost:8001'
    
    print('\nTesting OCR inference...')
    
    test_image = Image.new('RGB', (640, 480), color='white')
    
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(test_image)
    draw.text((50, 50), 'Test Document\n\nThis is a test.', fill='black')
    
    image_b64 = encode_image_to_base64(test_image)
    
    payload = {
        'model': 'deepseek-ocr',
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': '<image>\n<|grounding|>Convert the document to markdown. ',
                    },
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/png;base64,{image_b64}'},
                    },
                ],
            }
        ],
        'max_tokens': 4096,
    }
    
    try:
        with httpx.Client(timeout=300.0) as client:
            print('Sending OCR request...')
            response = client.post(
                f'{server_url}/v1/chat/completions', json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            print('✓ OCR inference successful')
            print(f'Response length: {len(content)} chars')
            print(f'Response preview: {content[:200]}...')
            
    except Exception as e:
        print(f'✗ OCR inference failed: {e}')
        return False
    
    return True


if __name__ == '__main__':
    print('DeepSeek OCR Integration Test')
    print('=' * 50)
    
    print('\nPrerequisite: Make sure deepseek-ocr-server is running')
    print('Run: docker compose -f docker-compose.app.yml up deepseek-ocr-server\n')
    
    if test_deepseek_server_connection():
        test_deepseek_ocr_inference()
