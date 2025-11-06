from pathlib import Path
from uuid import UUID


def get_job_base_path(job_id: UUID) -> str:
    return f'jobs/{job_id}'


def get_original_pdf_path(job_id: UUID, filename: str) -> str:
    return f'{get_job_base_path(job_id)}/upload/{filename}'


def get_images_path(job_id: UUID) -> str:
    return f'{get_job_base_path(job_id)}/pdf_to_images'


def get_page_image_path(job_id: UUID, page_number: int) -> str:
    return f'{get_images_path(job_id)}/page_{page_number:04d}.png'


def get_detections_path(job_id: UUID) -> str:
    return f'{get_job_base_path(job_id)}/page_segmentation'


def get_page_detections_path(job_id: UUID, page_number: int) -> str:
    return f'{get_detections_path(job_id)}/page_{page_number:04d}.json'


def get_annotated_path(job_id: UUID) -> str:
    return f'{get_job_base_path(job_id)}/draw_bounding_boxes'


def get_annotated_image_path(job_id: UUID, page_number: int) -> str:
    return f'{get_annotated_path(job_id)}/page_{page_number:04d}_annotated.png'


def get_text_extraction_path(job_id: UUID) -> str:
    return f'{get_job_base_path(job_id)}/text_extraction'


def get_element_image_path(job_id: UUID, page_number: int, element_idx: int) -> str:
    return f'{get_text_extraction_path(job_id)}/page_{page_number:04d}_element_{element_idx:03d}.png'


def get_element_text_path(job_id: UUID, page_number: int, element_idx: int) -> str:
    return f'{get_text_extraction_path(job_id)}/page_{page_number:04d}_element_{element_idx:03d}.txt'


def get_graph_path(job_id: UUID) -> str:
    return f'{get_job_base_path(job_id)}/assemble_graph/document_graph.json'


def get_step_folder(job_id: UUID, step_name: str) -> str:
    step_map = {
        'UPLOAD': 'upload',
        'PDF_TO_IMAGES': 'pdf_to_images',
        'PAGE_SEGMENTATION': 'page_segmentation',
        'DRAW_BOUNDING_BOXES': 'draw_bounding_boxes',
        'TEXT_EXTRACTION': 'text_extraction',
        'ASSEMBLE_GRAPH': 'assemble_graph',
    }
    
    folder_name = step_map.get(step_name, step_name.lower())
    return f'{get_job_base_path(job_id)}/{folder_name}'
