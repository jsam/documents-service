#!/usr/bin/env python3
"""
Inference script for DocLayout-YOLO model.
Processes images from inputs/ folder and saves visualized results to outputs/ folder.
"""

import os
import json
import argparse
from pathlib import Path
import torch
import torchvision
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from doclayout_yolo import YOLOv10


def load_colormap(N=256):
    """Generate colormap for visualization."""
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << (7 - j))
            g = g | (bitget(c, 1) << (7 - j))
            b = b | (bitget(c, 2) << (7 - j))
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    
    return cmap


def preprocess_image(image_path, target_size=1280):
    """
    Load and preprocess image for model input.
    
    Args:
        image_path: Path to input image
        target_size: Target size for resizing (default: 1280)
    
    Returns:
        Preprocessed tensor and original image
    """
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    original_size = image.size
    image_resized = image.resize((target_size, target_size))
    
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_tensor = torch.from_numpy(img_array).unsqueeze(0)
    
    return img_tensor, original_image, original_size


def extract_predictions(output, conf_threshold=0.25, iou_threshold=0.45, num_classes=10):
    """
    Extract predictions from YOLOv10 model output and apply NMS.
    
    Args:
        output: Model output dictionary with 'one2one' and 'one2many' keys
        conf_threshold: Confidence threshold
        iou_threshold: IOU threshold for NMS
        num_classes: Number of classes (default: 10)
    
    Returns:
        Filtered boxes, scores, and classes
    """
    if isinstance(output, dict) and 'one2one' in output:
        pred = output['one2one']
        if isinstance(pred, (tuple, list)) and len(pred) > 0:
            pred = pred[0]
    elif isinstance(output, (list, tuple)):
        pred = output[0]
    else:
        pred = output
    
    if not isinstance(pred, torch.Tensor):
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    if pred.dim() == 3:
        pred = pred[0]
    
    pred = pred.transpose(0, 1)
    
    if pred.shape[-1] < 4:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    boxes = pred[:, :4]
    
    if pred.shape[-1] >= 4 + num_classes:
        class_scores = pred[:, 4:4+num_classes]
        scores, classes = class_scores.max(dim=1)
    elif pred.shape[-1] >= 5:
        scores = pred[:, 4]
        if pred.shape[-1] > 5:
            classes = pred[:, 5:].argmax(dim=1)
        else:
            classes = torch.zeros(len(pred), dtype=torch.long)
    else:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]
    
    if len(boxes) > 0:
        indices = torchvision.ops.nms(boxes, scores, iou_threshold)
        boxes = boxes[indices]
        scores = scores[indices]
        classes = classes[indices]
    
    return boxes, scores, classes


def scale_boxes(boxes, original_size, model_size=1280):
    """
    Scale bounding boxes from model coordinates to original image coordinates.
    
    Args:
        boxes: Tensor of boxes in xyxy format
        original_size: Tuple of (width, height) of original image
        model_size: Size of model input (default: 1280)
    
    Returns:
        Scaled boxes
    """
    if len(boxes) == 0:
        return boxes
    
    orig_w, orig_h = original_size
    scale_x = orig_w / model_size
    scale_y = orig_h / model_size
    
    boxes = boxes.clone()
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    
    return boxes


def visualize_predictions(image, boxes, classes, scores, class_mapping, alpha=0.3):
    """
    Visualize predictions on image with bounding boxes and labels.
    
    Args:
        image: Original image as numpy array (RGB)
        boxes: Bounding boxes in xyxy format
        classes: Class IDs
        scores: Confidence scores
        class_mapping: Dictionary mapping class IDs to names
        alpha: Transparency for filled boxes
    
    Returns:
        Visualized image as numpy array (RGB)
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = image.copy()
    
    cmap = load_colormap(N=len(class_mapping))
    
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        class_id = int(classes[i])
        score = float(scores[i])
        
        class_name = class_mapping.get(str(class_id), f"class_{class_id}")
        label = f"{class_name}: {score:.3f}"
        
        color = tuple(int(c) for c in cmap[class_id])
        
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(
            image,
            (x_min, y_min - text_height - baseline - 5),
            (x_min + text_width, y_min),
            color,
            -1
        )
        cv2.putText(
            image,
            label,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def infer(
    model_dir='mlpkg',
    input_dir='inputs',
    output_dir='outputs',
    conf_threshold=None,
    iou_threshold=None,
    device='cpu'
):
    """
    Run inference on all images in input directory.
    
    Args:
        model_dir: Directory containing exported model
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        conf_threshold: Confidence threshold (None = use config default)
        iou_threshold: IOU threshold (None = use config default)
        device: Device to run inference on ('cpu' or 'cuda')
    """
    print("=" * 70)
    print("DocLayout-YOLO Inference")
    print("=" * 70)
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[1/6] Loading configuration from {model_dir}/...")
    with open(os.path.join(model_dir, 'config.json')) as f:
        config = json.load(f)
    
    with open(os.path.join(model_dir, 'class_mapping.json')) as f:
        class_mapping = json.load(f)
    
    if conf_threshold is None:
        conf_threshold = config['inference']['conf_threshold']
    if iou_threshold is None:
        iou_threshold = config['inference']['iou_threshold']
    
    model_size = config['inference']['imgsz']
    
    print(f"  Model: {config['model_name']}")
    print(f"  Classes: {config['num_classes']}")
    print(f"  Confidence threshold: {conf_threshold}")
    print(f"  IOU threshold: {iou_threshold}")
    print(f"  Input size: {model_size}x{model_size}")
    print()
    
    print(f"[2/6] Loading model...")
    model_path = 'models/DocLayout-YOLO-DocStructBench-imgsz1280-2501/doclayout_yolo_docstructbench_imgsz1280_2501.pt'
    if not os.path.exists(model_path):
        print(f"  Model weights not found at {model_path}")
        print(f"  Attempting to use exported model from {model_dir}/...")
        checkpoint = torch.load(
            os.path.join(model_dir, 'doclayout_yolo_model.pt'),
            weights_only=False,
            map_location=device
        )
        model_obj = checkpoint['model']
        temp_path = '/tmp/temp_model.pt'
        torch.save(model_obj, temp_path)
        model = YOLOv10(temp_path)
    else:
        model = YOLOv10(model_path)
    
    print(f"  Device: {device}")
    print(f"  Model loaded successfully")
    print()
    
    print(f"[3/6] Scanning input directory: {input_dir}/")
    if not os.path.exists(input_dir):
        print(f"  Error: Input directory '{input_dir}' does not exist!")
        return
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    
    if not image_files:
        print(f"  Warning: No images found in '{input_dir}'")
        return
    
    print(f"  Found {len(image_files)} images")
    print()
    
    print(f"[4/6] Running inference...")
    results_summary = []
    
    for image_file in tqdm(image_files, desc="  Processing"):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        
        try:
            original_image = np.array(Image.open(input_path).convert('RGB'))
            
            det_results = model.predict(
                input_path,
                imgsz=model_size,
                conf=conf_threshold,
                device=device,
            )
            
            if len(det_results) == 0:
                results_summary.append({
                    'image': image_file,
                    'detections': 0,
                    'success': True
                })
                Image.fromarray(original_image).save(output_path, quality=95)
                continue
            
            det_res = det_results[0]
            boxes = det_res.__dict__['boxes'].xyxy
            classes = det_res.__dict__['boxes'].cls
            scores = det_res.__dict__['boxes'].conf
            
            indices = torchvision.ops.nms(
                boxes=torch.Tensor(boxes),
                scores=torch.Tensor(scores),
                iou_threshold=iou_threshold
            )
            boxes, scores, classes = boxes[indices], scores[indices], classes[indices]
            
            if len(boxes.shape) == 1:
                boxes = np.expand_dims(boxes, 0)
                scores = np.expand_dims(scores, 0)
                classes = np.expand_dims(classes, 0)
            
            vis_image = visualize_predictions(
                original_image,
                boxes.cpu().numpy() if hasattr(boxes, 'cpu') else boxes,
                classes.cpu().numpy() if hasattr(classes, 'cpu') else classes,
                scores.cpu().numpy() if hasattr(scores, 'cpu') else scores,
                class_mapping
            )
            
            Image.fromarray(vis_image).save(output_path, quality=95)
            
            results_summary.append({
                'image': image_file,
                'detections': len(boxes),
                'success': True
            })
            
        except Exception as e:
            print(f"\n  Error processing {image_file}: {e}")
            import traceback
            traceback.print_exc()
            results_summary.append({
                'image': image_file,
                'detections': 0,
                'success': False,
                'error': str(e)
            })
    
    print()
    print(f"[5/6] Saving results to {output_dir}/")
    
    results_json_path = os.path.join(output_dir, 'inference_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"  Saved inference results to: {results_json_path}")
    print()
    
    print(f"[6/6] Summary:")
    successful = sum(1 for r in results_summary if r['success'])
    total_detections = sum(r['detections'] for r in results_summary if r['success'])
    
    print(f"  Processed: {successful}/{len(image_files)} images")
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections per image: {total_detections/successful if successful > 0 else 0:.1f}")
    
    class_counts = {}
    print()
    print("  Detection breakdown by class:")
    for result in results_summary:
        if result['success'] and result['detections'] > 0:
            input_path = os.path.join(input_dir, result['image'])
            try:
                det_results = model.predict(
                    input_path,
                    imgsz=model_size,
                    conf=conf_threshold,
                    device=device,
                    verbose=False
                )
                if len(det_results) > 0:
                    det_res = det_results[0]
                    boxes = det_res.__dict__['boxes'].xyxy
                    classes_pred = det_res.__dict__['boxes'].cls
                    scores = det_res.__dict__['boxes'].conf
                    
                    indices = torchvision.ops.nms(
                        boxes=torch.Tensor(boxes),
                        scores=torch.Tensor(scores),
                        iou_threshold=iou_threshold
                    )
                    classes_pred = classes_pred[indices]
                    
                    for cls in classes_pred.cpu().numpy():
                        cls_name = class_mapping.get(str(int(cls)), f"class_{int(cls)}")
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            except:
                pass
    
    for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {cls_name}: {count}")
    
    print()
    print("=" * 70)
    print("✓ Inference completed successfully!")
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on images using exported DocLayout-YOLO model'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='mlpkg',
        help='Directory containing exported model (default: mlpkg)'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='inputs',
        help='Directory containing input images (default: inputs)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory to save output images (default: outputs)'
    )
    
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=None,
        help='Confidence threshold (default: use config value)'
    )
    
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=None,
        help='IOU threshold for NMS (default: use config value)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run inference on (default: cpu)'
    )
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    infer(
        model_dir=args.model_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device
    )


if __name__ == "__main__":
    main()
