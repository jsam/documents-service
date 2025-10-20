# DocLayout-YOLO Model Package

This package contains the exported DocLayout-YOLO model for document layout analysis.

## Contents

- `doclayout_yolo_model.pt`: TorchScript traced model
- `postprocessor.pt`: TorchScript postprocessor module (optional)
- `config.json`: Complete model configuration and metadata
- `class_mapping.json`: Mapping from class IDs to class names

## Model Information

- **Task**: Document Layout Analysis (Object Detection)
- **Input Size**: 1280x1280
- **Classes**: 10 document element types
- **Framework**: PyTorch (TorchScript)

## Classes

- 0: title
- 1: plain text
- 2: abandon
- 3: figure
- 4: figure_caption
- 5: table
- 6: table_caption
- 7: table_footnote
- 8: isolate_formula
- 9: formula_caption

## Usage

### Loading the Model

```python
import torch

# Load the traced model
model = torch.jit.load('doclayout_yolo_model.pt')
model.eval()

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
```

### Preprocessing

```python
import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=1280):
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize to target size
    image = image.resize((target_size, target_size))
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Convert to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_tensor = torch.from_numpy(img_array).unsqueeze(0)
    
    return img_tensor
```

### Inference

```python
# Preprocess image
input_tensor = preprocess_image('document.jpg')
input_tensor = input_tensor.to(device)

# Run inference
with torch.no_grad():
    outputs = model(input_tensor)

# Note: You will need to parse the outputs based on the model's output format
# The outputs typically contain bounding boxes, confidence scores, and class predictions
```

### Postprocessing

```python
import torchvision

def postprocess_predictions(boxes, scores, classes, conf_threshold=0.25, iou_threshold=0.45):
    # Filter by confidence
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]
    
    # Apply NMS
    if len(boxes) > 0:
        indices = torchvision.ops.nms(boxes, scores, iou_threshold)
        boxes = boxes[indices]
        scores = scores[indices]
        classes = classes[indices]
    
    return boxes, scores, classes
```

## Configuration

Default inference parameters:
- Confidence Threshold: 0.25
- IOU Threshold: 0.45
- Input Size: 1280x1280

See `config.json` for complete configuration details.

## Notes

- The model expects RGB images resized to 1280x1280 pixels
- Input values should be normalized to [0, 1] range
- Bounding boxes are in xyxy format (x_min, y_min, x_max, y_max)
- Coordinates are relative to the 1280x1280 input size

## Requirements

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy
- PIL/Pillow

## Export Information

Exported on: Mo 13 Okt 2025 17:46:25 CEST
Device: cpu
PyTorch Version: 2.8.0
