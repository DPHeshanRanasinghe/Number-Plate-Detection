# Automatic Number Plate Recognition (ANPR) System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ADD8.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

A complete Automatic Number Plate Recognition system combining YOLOv8 for license plate detection and EasyOCR for text extraction. This end-to-end solution achieves high accuracy in detecting and reading vehicle license plates from images and video streams.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Real-Time Deployment](#real-time-deployment)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Results](#results)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Overview

This ANPR system implements a two-stage pipeline:

1. **Detection Stage**: YOLOv8 nano model trained to detect license plates in images
2. **Recognition Stage**: EasyOCR extracts text characters from detected plate regions

The system is designed for real-time processing and can be deployed on edge devices, servers, or cloud infrastructure for various applications including parking management, toll collection, and access control systems.

## Features

- **High Accuracy Detection**: 97.8% mAP@50, 99.1% precision, 94.7% recall
- **Real-Time Processing**: Optimized for fast inference on GPU and CPU
- **End-to-End Pipeline**: Complete workflow from image input to text output
- **Multiple Export Formats**: ONNX, TensorRT, TFLite for deployment flexibility
- **Batch Processing**: Process multiple images or video frames efficiently
- **Configurable Confidence Thresholds**: Adjust detection sensitivity based on use case
- **Comprehensive Visualization**: Bounding boxes with confidence scores and extracted text

## Model Performance

Trained on 10,125 images over 50 epochs (4.4 hours on NVIDIA GTX 1650):

| Metric | Value |
|--------|-------|
| mAP@50 | 97.8% |
| mAP@50-95 | 71.6% |
| Precision | 99.1% |
| Recall | 94.7% |
| Inference Speed | 4.6ms per image |

## Installation

### Prerequisites

- Python 3.11+
- CUDA 11.8+ (for GPU acceleration)
- Anaconda or Miniconda (recommended)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/Number-Plate-Detection.git
cd Number-Plate-Detection

# Create conda environment
conda create -n anpr python=3.11 -y
conda activate anpr

# Install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install ultralytics easyocr opencv-python pandas matplotlib seaborn
```

### Alternative: pip-only Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics easyocr opencv-python pandas matplotlib seaborn pillow pyyaml
```

## Usage

### Basic Detection and Recognition

```python
from ultralytics import YOLO
import easyocr
import cv2

# Load trained model
model = YOLO('runs/detect/license_plate_detection/yolov8n_run12/weights/best.pt')

# Initialize OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Detect and read license plate
results = model.predict('path/to/image.jpg', conf=0.25)

for result in results:
    for box in result.boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # Extract plate region
        img = cv2.imread('path/to/image.jpg')
        plate_img = img[int(y1):int(y2), int(x1):int(x2)]
        
        # Perform OCR
        ocr_result = reader.readtext(plate_img)
        plate_text = ' '.join([text[1] for text in ocr_result])
        
        print(f"Detected Plate: {plate_text}")
        print(f"Confidence: {float(box.conf[0]):.2f}")
```

### Batch Processing

```python
from pathlib import Path

# Process multiple images
image_dir = Path('path/to/images')
results = model.predict(str(image_dir), conf=0.25, save=True)

# Results saved to runs/detect/predict/
```

### Training Custom Model

```python
from ultralytics import YOLO

# Initialize model
model = YOLO('yolov8n.pt')

# Train on custom dataset
results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,  # GPU device
    project='license_plate_detection',
    name='custom_run',
    patience=10
)
```

## Real-Time Deployment

### Video Stream Processing

```python
import cv2
from ultralytics import YOLO
import easyocr

model = YOLO('best.pt')
reader = easyocr.Reader(['en'], gpu=True)

# Open video stream (webcam, RTSP, or video file)
cap = cv2.VideoCapture(0)  # 0 for webcam, or 'rtsp://...' for IP camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect plates
    results = model.predict(frame, conf=0.4, verbose=False)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Extract and recognize plate
            plate_img = frame[y1:y2, x1:x2]
            if plate_img.size > 0:
                ocr_result = reader.readtext(plate_img)
                text = ' '.join([r[1] for r in ocr_result])
                
                # Draw results
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('ANPR', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### RTSP Camera Integration

```python
# Connect to IP camera
rtsp_url = "rtsp://username:password@192.168.1.100:554/stream"
cap = cv2.VideoCapture(rtsp_url)

# Process stream with same logic as above
```

### Model Export for Deployment

```python
# Export to ONNX (cross-platform)
model.export(format='onnx')

# Export to TensorRT (NVIDIA GPUs - fastest)
model.export(format='engine')

# Export to TFLite (mobile/edge devices)
model.export(format='tflite')
```

### API Server Deployment

```python
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import easyocr
import numpy as np
import cv2

app = FastAPI()
model = YOLO('best.pt')
reader = easyocr.Reader(['en'], gpu=True)

@app.post("/detect-plate/")
async def detect_plate(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detect and recognize
    results = model.predict(img, conf=0.25)
    plates = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            plate_img = img[y1:y2, x1:x2]
            ocr_result = reader.readtext(plate_img)
            text = ' '.join([r[1] for r in ocr_result])
            
            plates.append({
                "text": text,
                "confidence": float(box.conf[0]),
                "bbox": [x1, y1, x2, y2]
            })
    
    return {"plates": plates}

# Run with: uvicorn api:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
Number-Plate-Detection/
├── notebook.ipynb                      # Complete ANPR implementation
├── number-plate-detection.ipynb        # Main project notebook
├── README.md                           # Project documentation
├── .gitignore                          # Git ignore rules
├── runs/                               # Training results (excluded from git)
│   └── detect/
│       └── license_plate_detection/
│           └── yolov8n_run12/
│               ├── weights/
│               │   ├── best.pt         # Best model weights
│               │   ├── best.onnx       # ONNX export
│               │   └── last.pt         # Last epoch weights
│               ├── results.png         # Training metrics
│               ├── confusion_matrix.png
│               └── val_batch0_pred.jpg # Validation predictions
└── data.yaml                           # Dataset configuration (if training)
```

## Technical Details

### YOLOv8 Architecture

- **Model**: YOLOv8 Nano (lightweight, fast inference)
- **Input Size**: 640x640 pixels
- **Parameters**: 3.01M parameters
- **GFLOPs**: 8.2
- **Backbone**: CSPDarknet with C2f modules
- **Neck**: PANet (Path Aggregation Network)
- **Head**: Decoupled detection head

### Training Configuration

```yaml
epochs: 50
batch_size: 16
image_size: 640
optimizer: MuSGD (lr=0.002, momentum=0.9)
patience: 10 (early stopping)
augmentations: 
  - Mosaic
  - RandomFlip
  - HSV augmentation
  - Random affine transforms
```

### Data Preprocessing

- Image resizing to 640x640
- Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- YOLO format labels (normalized coordinates)

### OCR Configuration

- **Engine**: EasyOCR
- **Language**: English
- **GPU Acceleration**: Enabled
- **Character Recognition**: Alphanumeric + special characters

## Results

### Detection Examples

The trained model successfully detects license plates in various conditions:

- Different lighting conditions (day/night)
- Various viewing angles
- Multiple plates in single image
- Partially occluded plates
- Different plate sizes and aspect ratios

### Training Metrics

Training progression over 50 epochs:

| Epoch | Box Loss | mAP@50 | mAP@50-95 | Precision | Recall |
|-------|----------|---------|-----------|-----------|--------|
| 1     | 1.234    | 93.1%   | 62.3%     | 96.3%     | 89.1%  |
| 25    | 0.983    | 97.8%   | 70.8%     | 98.6%     | 94.7%  |
| 50    | 0.838    | 97.7%   | 71.4%     | 99.1%     | 94.7%  |

### Performance Benchmarks

| Hardware | Inference Time | FPS |
|----------|----------------|-----|
| NVIDIA GTX 1650 | 4.6ms | 217 |
| NVIDIA RTX 3090 | 2.1ms | 476 |
| Intel i7 CPU | 45ms | 22 |

## Requirements

### Core Dependencies

```
torch>=2.5.1
torchvision>=0.20.1
ultralytics>=8.4.7
easyocr>=1.7.2
opencv-python>=4.12.0
numpy>=1.26.4
pandas>=2.2.3
matplotlib>=3.9.1
seaborn>=0.13.2
pillow>=10.4.0
pyyaml>=6.0.3
```

### Optional Dependencies

```
fastapi>=0.115.0        # For API deployment
uvicorn>=0.34.0         # ASGI server
python-multipart>=0.0.20 # File uploads
tensorrt>=8.6.0         # NVIDIA TensorRT (GPU optimization)
onnxruntime-gpu>=1.20.0 # ONNX runtime with GPU
```

## Dataset Format

For training custom models, organize data in YOLO format:

```
dataset/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── labels/
│       ├── img1.txt
│       └── img2.txt
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

**data.yaml**:
```yaml
path: /path/to/dataset
train: train/images
val: valid/images
test: test/images

nc: 1  # number of classes
names: ['license-plate']
```

**Label format** (normalized coordinates):
```
class_id center_x center_y width height
0 0.5 0.5 0.2 0.1
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection framework
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for optical character recognition
- Dataset sourced from [Kaggle - License Plate Detection Dataset](https://www.kaggle.com/datasets/barkataliarbab/license-plate-detection-dataset-10125-images)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{anpr_yolov8_2026,
  title={Automatic Number Plate Recognition System using YOLOv8 and EasyOCR},
  author={Heshan Ranasinghe},
  year={2026},
  url={https://github.com/DPHeshanRanasinghe/Number-Plate-Detection}
}
```

## Author

**Heshan Ranasinghe**  
Electronic and Telecommunication Engineering Undergraduate

- Email: hranasinghe505@gmail.com
- GitHub: [@DPHeshanRanasinghe](https://github.com/DPHeshanRanasinghe)
- LinkedIn: [Heshan Ranasinghe](https://www.linkedin.com/in/heshan-ranasinghe-988b00290)