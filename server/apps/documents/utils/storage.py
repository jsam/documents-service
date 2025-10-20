from pathlib import Path
from uuid import UUID


def get_job_base_path(job_id: UUID) -> str:
    return f'jobs/{job_id}'


def get_original_pdf_path(job_id: UUID, filename: str) -> str:
    return f'{get_job_base_path(job_id)}/step0_upload/{filename}'


def get_step1_images_path(job_id: UUID) -> str:
    return f'{get_job_base_path(job_id)}/step1_images'


def get_page_image_path(job_id: UUID, page_number: int) -> str:
    return f'{get_step1_images_path(job_id)}/page_{page_number:04d}.png'


def get_step2_detections_path(job_id: UUID) -> str:
    return f'{get_job_base_path(job_id)}/step2_detections'


def get_page_detections_path(job_id: UUID, page_number: int) -> str:
    return f'{get_step2_detections_path(job_id)}/page_{page_number:04d}.json'


def get_step3_annotated_path(job_id: UUID) -> str:
    return f'{get_job_base_path(job_id)}/step3_annotated'


def get_annotated_image_path(job_id: UUID, page_number: int) -> str:
    return f'{get_step3_annotated_path(job_id)}/page_{page_number:04d}_annotated.png'


def get_step4_text_path(job_id: UUID) -> str:
    return f'{get_job_base_path(job_id)}/step4_text'


def get_element_image_path(job_id: UUID, page_number: int, element_idx: int) -> str:
    return f'{get_step4_text_path(job_id)}/page_{page_number:04d}_element_{element_idx:03d}.png'


def get_element_text_path(job_id: UUID, page_number: int, element_idx: int) -> str:
    return f'{get_step4_text_path(job_id)}/page_{page_number:04d}_element_{element_idx:03d}.txt'


def get_step5_graph_path(job_id: UUID) -> str:
    return f'{get_job_base_path(job_id)}/step5_graph/document_graph.json'


def get_step6_points_path(job_id: UUID) -> str:
    return f'{get_job_base_path(job_id)}/step6_points'


def get_points_result_path(job_id: UUID, page_number: int) -> str:
    return f'{get_step6_points_path(job_id)}/page_{page_number:04d}_points.json'


def get_step_folder(job_id: UUID, step_name: str) -> str:
    step_map = {
        'UPLOAD': 'step0_upload',
        'PDF_TO_IMAGES': 'step1_images',
        'ML_INFERENCE': 'step2_detections',
        'DRAW_BOUNDING_BOXES': 'step3_annotated',
        'TEXT_EXTRACTION': 'step4_text',
        'ASSEMBLE_GRAPH': 'step5_graph',
        'POINTS_OCR': 'step6_points',
    }
    
    folder_name = step_map.get(step_name, step_name.lower())
    return f'{get_job_base_path(job_id)}/{folder_name}'
