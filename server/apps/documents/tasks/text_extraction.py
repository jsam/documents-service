import csv
import json
import logging
import re
from io import BytesIO, StringIO
from uuid import UUID

import easyocr
import numpy as np
import torch
from django.conf import settings
from django.utils import timezone
from PIL import Image

from server.apps.documents.models import DocumentElement, DocumentJob, ProcessingStep
from server.apps.documents.utils.minio_client import download_file, upload_file
from server.apps.documents.utils.storage import (
    get_element_image_path,
    get_page_detections_path,
    get_page_image_path,
)

logger = logging.getLogger(__name__)

reader = easyocr.Reader(['en'], gpu=False)


# Global variable to store model in worker process
_olmocr_model = None
_olmocr_processor = None

def extract_table_with_olmocr_direct(image: Image.Image) -> dict | None:
    """
    Extract table using olmOCR model directly (called from table_extraction worker).
    Returns dict with 'html' and 'data' (2D array) or None if failed.
    """
    global _olmocr_model, _olmocr_processor
    
    try:
        import torch
        
        logger.info('[OLMOCR] Extracting table structure with olmOCR2 (direct call)')
        
        # Use preloaded model from worker init
        if _olmocr_model is None or _olmocr_processor is None:
            logger.warning('[OLMOCR] Model not preloaded in worker, loading now...')
            from server.apps.documents.utils.model_loader import load_olmocr_model
            _olmocr_model, _olmocr_processor = load_olmocr_model()
            logger.info('[OLMOCR] Model loaded successfully in worker process')
        else:
            logger.info('[OLMOCR] Using preloaded model from worker process')
        
        model = _olmocr_model
        processor = _olmocr_processor
        
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Convert this table to HTML format using <table>, <tr>, <th>, and <td> tags. Only output the HTML table, no explanation.'},
                    {'type': 'image', 'image': image},
                ],
            }
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors='pt',
        )
        inputs = {key: value.to(settings.OLMOCR_DEVICE) for key, value in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=True,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        logger.info(f'[OLMOCR] Extracted output length: {len(output_text)} chars')
        logger.info(f'[OLMOCR] Extracted output preview: {output_text[:500]}...')
        logger.info(f'[OLMOCR] Contains <table>: {"<table>" in output_text}')
        logger.info(f'[OLMOCR] Contains pipe |: {"| " in output_text}')
        
        if '<table>' in output_text:
            logger.info('[OLMOCR] Parsing HTML table')
            table_data = parse_html_table_to_array(output_text)
            if table_data:
                logger.info(f'[OLMOCR] HTML parsing successful: {len(table_data)} rows, {len(table_data[0]) if table_data else 0} cols')
                return {
                    'html': output_text,
                    'data': table_data,
                }
            else:
                logger.warning('[OLMOCR] HTML parsing returned empty table_data')
        
        if '|' in output_text:
            logger.info('[OLMOCR] Parsing markdown table')
            table_data = parse_markdown_table_to_array(output_text)
            if table_data:
                logger.info(f'[OLMOCR] Markdown parsing successful: {len(table_data)} rows, {len(table_data[0]) if table_data else 0} cols')
                return {
                    'html': output_text,
                    'data': table_data,
                }
            else:
                logger.warning('[OLMOCR] Markdown parsing returned empty table_data')
        
        logger.error('[OLMOCR] No valid table format detected in output')
        return None
            
    except Exception as e:
        logger.error(f'[OLMOCR] Error: {e}', exc_info=True)
        return None


def parse_html_table_to_array(html: str) -> list[list[str]]:
    """
    Parse HTML table string to 2D array.
    """
    try:
        from html.parser import HTMLParser
        
        class TableParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.tables = []
                self.current_table = []
                self.current_row = []
                self.current_cell = []
                self.in_table = False
                self.in_row = False
                self.in_cell = False
            
            def handle_starttag(self, tag, attrs):
                if tag == 'table':
                    self.in_table = True
                    self.current_table = []
                elif tag == 'tr' and self.in_table:
                    self.in_row = True
                    self.current_row = []
                elif tag in ['td', 'th'] and self.in_row:
                    self.in_cell = True
                    self.current_cell = []
            
            def handle_endtag(self, tag):
                if tag == 'table':
                    if self.current_table:
                        self.tables.append(self.current_table)
                    self.in_table = False
                elif tag == 'tr':
                    if self.current_row:
                        self.current_table.append(self.current_row)
                    self.in_row = False
                elif tag in ['td', 'th']:
                    cell_text = ''.join(self.current_cell).strip()
                    self.current_row.append(cell_text)
                    self.in_cell = False
            
            def handle_data(self, data):
                if self.in_cell:
                    self.current_cell.append(data)
        
        parser = TableParser()
        parser.feed(html)
        
        if parser.tables:
            table_data = parser.tables[0]
            logger.info(f'[TABLE_PARSER] Parsed {len(table_data)} rows from HTML')
            return table_data
        
        return []
        
    except Exception as e:
        logger.error(f'[TABLE_PARSER] Error parsing HTML: {e}', exc_info=True)
        return []


def parse_markdown_table_to_array(markdown: str) -> list[list[str]]:
    """
    Parse markdown table string to 2D array.
    Markdown format: | Col1 | Col2 |
                     |------|------|
                     | Val1 | Val2 |
    """
    try:
        lines = markdown.strip().split('\n')
        table_data = []
        
        for line in lines:
            line = line.strip()
            if re.match(r'^\|?[-:| ]+\|?$', line):
                continue
            
            if '|' in line:
                cells = [cell.strip() for cell in line.strip('|').split('|')]
                if cells and any(cell for cell in cells):
                    table_data.append(cells)
        
        logger.info(f'[TABLE_PARSER] Parsed {len(table_data)} rows from markdown')
        return table_data
        
    except Exception as e:
        logger.error(f'[TABLE_PARSER] Error parsing markdown: {e}', exc_info=True)
        return []


def extract_text_from_region(image: Image.Image) -> str:
    """
    Extract text using EasyOCR (fast, no Ollama fallback for regular text).
    """
    try:
        logger.info(f'[OCR] Processing image size: {image.size}')
        result = reader.readtext(np.array(image))
        
        logger.info(f'[OCR] Found {len(result)} text regions')
        
        lines = []
        for detection in result:
            bbox, text, conf = detection
            lines.append((text, conf))
            logger.info(f'[OCR] Extracted: "{text}" (conf: {conf:.2f})')
        
        full_text = ' '.join(t for t, _ in lines) if lines else ''
        logger.info(f'[OCR] Final text: "{full_text}" ({len(lines)} lines)')
        return full_text.strip()
    except Exception as e:
        logger.error(f'[OCR] EasyOCR error: {e}', exc_info=True)
        return ''


def execute_text_extraction(job_id: str, celery_task_id: str) -> dict:
    job = DocumentJob.objects.get(id=UUID(job_id))
    step = ProcessingStep.objects.get(job=job, step_name='TEXT_EXTRACTION')
    
    step.mark_in_progress(celery_task_id)
    job.current_step = 'TEXT_EXTRACTION'
    job.save(update_fields=['current_step'])
    
    try:
        total_pages = job.total_pages
        step.update_progress(0, total_pages)
        
        total_elements = 0
        
        for page_num in range(1, total_pages + 1):
            image_path = get_page_image_path(job.id, page_num)
            image_bytes = download_file(job.minio_bucket, image_path)
            
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            img_width, img_height = image.size
            
            detections_path = get_page_detections_path(job.id, page_num)
            detections_bytes = download_file(job.minio_bucket, detections_path)
            detections_data = json.loads(detections_bytes.decode('utf-8'))
            
            detections = detections_data['detections']
            
            logger.info(f'[TEXT] Page {page_num}: {len(detections)} detections')
            
            detections.sort(key=lambda d: (d['bbox']['y1'], d['bbox']['x1']))
            
            page_elements = []
            
            for idx, detection in enumerate(detections):
                bbox = detection['bbox']
                element_type = detection['element_type']
                confidence = detection['confidence']
                
                x1 = int(bbox['x1'] * img_width)
                y1 = int(bbox['y1'] * img_height)
                x2 = int(bbox['x2'] * img_width)
                y2 = int(bbox['y2'] * img_height)
                
                logger.info(f'[TEXT] Element {idx}: type={element_type}, bbox=({x1},{y1},{x2},{y2})')
                
                cropped = image.crop((x1, y1, x2, y2))
                
                element_img_path = get_element_image_path(job.id, page_num, idx)
                
                cropped_bytes = BytesIO()
                cropped.save(cropped_bytes, format='PNG')
                cropped_bytes.seek(0)
                
                upload_file(
                    bucket=job.minio_bucket,
                    object_name=element_img_path,
                    file_data=cropped_bytes,
                    content_type='image/png',
                )
                
                extracted_text = None
                table_html = None
                table_data = None
                
                if element_type == 'table':
                    logger.info(f'[TEXT] Extracting table structure with olmOCR for element {idx}')
                    table_result = extract_table_with_olmocr_direct(cropped)
                    if table_result and table_result.get('data'):
                        table_html = table_result.get('html', '')
                        table_data = table_result['data']
                        extracted_text = '\\n'.join([','.join(str(cell) for cell in row) for row in table_data])
                        logger.info(f'[TEXT] Table extracted: {len(table_data)} rows, {len(table_data[0]) if table_data else 0} columns')
                    else:
                        logger.warning(f'[TEXT] olmOCR table extraction failed for element {idx}, using EasyOCR fallback')
                        extracted_text = extract_text_from_region(cropped)
                
                # Handle text elements with fast EasyOCR (no Ollama)
                elif element_type in ['title', 'plain_text', 'figure_caption', 'table_caption', 'table_footnote', 'formula_caption']:
                    logger.info(f'[TEXT] Running EasyOCR on element {idx}')
                    extracted_text = extract_text_from_region(cropped)
                    logger.info(f'[TEXT] OCR result: "{extracted_text}"')
                
                # Skip OCR for figures and other non-text elements
                else:
                    logger.info(f'[TEXT] Skipping OCR for element type: {element_type}')
                
                element_metadata = {
                    'page_number': page_num,
                    'element_index': idx,
                    'bbox': {
                        'x1': bbox['x1'],
                        'y1': bbox['y1'],
                        'x2': bbox['x2'],
                        'y2': bbox['y2'],
                    },
                    'bbox_pixels': {
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                    },
                    'element_type': element_type,
                    'confidence': confidence,
                    'extracted_text': extracted_text,
                    'table_html': table_html,
                    'table_data': table_data,
                    'image_path': element_img_path,
                    'sequence': idx,
                }
                
                page_elements.append(element_metadata)
                
                # Save table as CSV file to MinIO if it's a table
                if element_type == 'table' and table_data:
                    csv_path = f'jobs/{job.id}/text_extraction/page_{page_num:04d}_element_{idx:03d}_table.csv'
                    csv_buffer = StringIO()
                    csv_writer = csv.writer(csv_buffer)
                    csv_writer.writerows(table_data)
                    csv_bytes = csv_buffer.getvalue().encode('utf-8')
                    
                    upload_file(
                        bucket=job.minio_bucket,
                        object_name=csv_path,
                        file_data=csv_bytes,
                        content_type='text/csv',
                    )
                    
                    # Store CSV path in metadata
                    element_metadata['csv_path'] = csv_path
                    
                    logger.info(f'[TEXT] Saved table CSV to {csv_path}')
                
                DocumentElement.objects.create(
                    job=job,
                    page_number=page_num,
                    bbox_x1=bbox['x1'],
                    bbox_y1=bbox['y1'],
                    bbox_x2=bbox['x2'],
                    bbox_y2=bbox['y2'],
                    element_type=element_type,
                    confidence=confidence,
                    extracted_text=extracted_text,
                    table_html=table_html,
                    table_data=table_data,
                    minio_image_key=element_img_path,
                    sequence=idx,
                )
                
                total_elements += 1
            
            metadata_path = f'jobs/{job.id}/text_extraction/page_{page_num}_elements.json'
            metadata_json = json.dumps({
                'page_number': page_num,
                'total_elements': len(page_elements),
                'image_dimensions': {'width': img_width, 'height': img_height},
                'elements': page_elements,
            }, indent=2).encode('utf-8')
            
            upload_file(
                bucket=job.minio_bucket,
                object_name=metadata_path,
                file_data=metadata_json,
                content_type='application/json',
            )
            
            logger.info(f'[TEXT] Saved metadata for page {page_num} with {len(page_elements)} elements')
            
            step.update_progress(page_num, total_pages)
        
        step.status = ProcessingStep.STATUS_COMPLETED
        step.completed_at = timezone.now()
        step.result_data = {'total_elements': total_elements, 'total_pages': total_pages}
        step.save(update_fields=['status', 'completed_at', 'result_data'])
        
        return {'success': True, 'total_pages': total_pages, 'total_elements': total_elements}
    
    except Exception as e:
        step.status = ProcessingStep.STATUS_FAILED
        step.error_message = str(e)
        step.retry_count += 1
        step.save(update_fields=['status', 'error_message', 'retry_count'])
        
        job.mark_failed(str(e), 'TEXT_EXTRACTION')
        
        raise
