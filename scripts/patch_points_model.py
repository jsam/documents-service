#!/usr/bin/env python3
"""
Patch POINTS-Reader model to use eager attention instead of flash_attention_2.
This is needed for CPU inference where flash-attn is not available.
For GPU mode with CUDA, flash_attention_2 can be used if flash-attn is installed.
"""
import os
import sys
from pathlib import Path


def patch_points_model():
    use_gpu = os.environ.get('POINTS_USE_GPU', 'false').lower() == 'true'
    
    cache_dir = Path.home() / '.cache/huggingface/modules/transformers_modules/tencent/POINTS_hyphen_Reader'
    
    if not cache_dir.exists():
        print('[PATCH] POINTS model not yet downloaded, skipping patch')
        return
    
    model_files = list(cache_dir.glob('*/modeling_pointsv15_chat.py'))
    
    if not model_files:
        print('[PATCH] No model files found, skipping patch')
        return
    
    if use_gpu:
        print('[PATCH] GPU mode enabled - skipping flash_attention_2 patch')
        print('[PATCH] Model will use flash_attention_2 if available, or fall back to eager')
        return
    
    print('[PATCH] CPU mode enabled - patching to use eager attention and CPU device')
    
    for model_file in model_files:
        print(f'[PATCH] Patching {model_file}')
        
        content = model_file.read_text()
        original_content = content
        
        content = content.replace(
            'config.llm_config._attn_implementation = "flash_attention_2"',
            'config.llm_config._attn_implementation = "eager"'
        )
        content = content.replace(
            'attn_implementation="flash_attention_2"',
            'attn_implementation="eager"'
        )
        
        content = content.replace(
            'config.vision_config._attn_implementation = "flash_attention_2"',
            'config.vision_config._attn_implementation = "eager"'
        )
        
        content = content.replace(
            '.cuda()',
            '.to(self.device)'
        )
        
        if content != original_content:
            model_file.write_text(content)
            print(f'[PATCH] Successfully patched {model_file}')
        else:
            print(f'[PATCH] File {model_file} already patched or no changes needed')


if __name__ == '__main__':
    patch_points_model()
