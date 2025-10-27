from pathlib import Path
from uuid import UUID


def get_job_base_path(job_id: UUID) -> str:
    return f'jobs/{job_id}'


def get_original_pdf_path(job_id: UUID, filename: str) -> str:
    return f'{get_job_base_path(job_id)}/upload/{filename}'


def get_images_path(job_id: UUID) -> str:
    return f'{get_job_base_path(job_id)}/PDF to Images'


def get_page_image_path(job_id: UUID, page_number: int) -> str:
    return f'{get_images_path(job_id)}/page_{page_number:04d}.png'


def get_detections_path(job_id: UUID) -> str:
    return f'{get_job_base_path(job_id)}/ML Inference'


def get_page_detections_path(job_id: UUID, page_number: int) -> str:
    return f'{get_detections_path(job_id)}/page_{page_number:04d}.json'


def get_annotated_path(job_id: UUID) -> str:
    return f'{get_job_base_path(job_id)}/Draw Bounding Boxes'


def get_annotated_image_path(job_id: UUID, page_number: int) -> str:
    return f'{get_annotated_path(job_id)}/page_{page_number:04d}_annotated.png'


def get_text_extraction_path(job_id: UUID) -> str:
    return f'{get_job_base_path(job_id)}/Text Extraction'


def get_element_image_path(job_id: UUID, page_number: int, element_idx: int) -> str:
    return f'{get_text_extraction_path(job_id)}/page_{page_number:04d}_element_{element_idx:03d}.png'


def get_element_text_path(job_id: UUID, page_number: int, element_idx: int) -> str:
    return f'{get_text_extraction_path(job_id)}/page_{page_number:04d}_element_{element_idx:03d}.txt'


def get_graph_path(job_id: UUID) -> str:
    return f'{get_job_base_path(job_id)}/Assemble Graph/document_graph.json'


def get_ocr_path(job_id: UUID) -> str:
    return f'{get_job_base_path(job_id)}/OCR Processing'


def get_ocr_result_path(job_id: UUID, page_number: int) -> str:
    return f'{get_ocr_path(job_id)}/page_{page_number:04d}_ocr.json'


def get_step_folder(job_id: UUID, step_name: str) -> str:
    step_map = {
        'UPLOAD': 'Upload',
        'PDF_TO_IMAGES': 'PDF to Images',
        'ML_INFERENCE': 'ML Inference',
        'DRAW_BOUNDING_BOXES': 'Draw Bounding Boxes',
        'TEXT_EXTRACTION': 'Text Extraction',
        'ASSEMBLE_GRAPH': 'Assemble Graph',
        'OCR_PROCESSING': 'OCR Processing',
    }
    
    folder_name = step_map.get(step_name, step_name.replace('_', ' ').title())
    return f'{get_job_base_path(job_id)}/{folder_name}'
