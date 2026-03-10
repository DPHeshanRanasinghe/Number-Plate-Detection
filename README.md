# Automatic Number Plate Recognition (ANPR) System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ADD8.svg)](https://github.com/ultralytics/ultralytics)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

An end-to-end Automatic Number Plate Recognition system that combines YOLOv8 for license plate detection and EasyOCR for text extraction. The system achieves 97.8% detection accuracy with real-time processing capabilities.

## Overview

This project implements a complete ANPR pipeline with two main stages:

1. **Detection**: YOLOv8 nano model identifies and localizes license plates in images
2. **Recognition**: EasyOCR extracts alphanumeric text from detected plate regions

The system can process images and video streams, making it suitable for applications like parking systems, toll collection, and access control.

## Key Features

- High accuracy license plate detection (97.8% mAP@50)
- Fast inference speed (4.6ms per image on GPU)
- End-to-end pipeline from image input to text output
- Support for batch processing of multiple images
- Configurable detection confidence thresholds
- Visualization with bounding boxes and extracted text

## Model Performance

Trained on 10,125 images for 50 epochs (4.4 hours training time):

| Metric | Value |
|--------|-------|
| mAP@50 | 97.8% |
| mAP@50-95 | 71.6% |
| Precision | 99.1% |
| Recall | 94.7% |
| Inference Speed | 4.6ms per image |
| Model Size | 6.2 MB |

## Installation

### Prerequisites

- Python 3.11 or higher
- CUDA 11.8+ (for GPU acceleration)
- 4GB+ GPU memory (NVIDIA recommended)

### Environment Setup

**Option 1: Using Conda (Recommended)**

```bash
# Clone the repository
git clone https://github.com/DPHeshanRanasinghe/Number-Plate-Detection.git
cd Number-Plate-Detection

# Create and activate environment
conda create -n ml_env_fixed python=3.11 -y
conda activate ml_env_fixed

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install ultralytics easyocr opencv-python pandas matplotlib seaborn ipywidgets
```

**Option 2: Using pip**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics easyocr opencv-python pandas matplotlib seaborn pillow pyyaml
```

## Usage

The complete implementation is available in the Jupyter notebooks. To use the system:

1. Open `notebook.ipynb` for the full ANPR pipeline
2. Run cells sequentially to:
   - Load and explore the dataset
   - Train the YOLOv8 detection model
   - Test license plate detection
   - Extract text using EasyOCR
   - Visualize results

The trained model weights are saved in:
```
runs/detect/license_plate_detection/yolov8n_run12/weights/best.pt
```

## Project Structure

```
Number-Plate-Detection/
├── notebook.ipynb                      # Complete ANPR implementation
├── number-plate-detection.ipynb        # Main project notebook
├── README.md                           # Documentation
├── .gitignore                          # Git ignore rules
└── runs/                               # Training results (not in git)
    └── detect/
        └── license_plate_detection/
            └── yolov8n_run12/
                ├── weights/
                │   ├── best.pt         # Best model checkpoint
                │   ├── best.onnx       # ONNX export
                │   └── last.pt         # Last epoch checkpoint
                ├── results.png         # Training curves
                ├── confusion_matrix.png
                └── val_batch0_pred.jpg # Sample predictions
```

## Technical Details

### Model Architecture

- **Base Model**: YOLOv8 Nano
- **Input Resolution**: 640x640 pixels
- **Parameters**: 3.01 million
- **Architecture**: CSPDarknet backbone with PANet neck

### Training Configuration

- Dataset: 10,125 license plate images
- Split: 70% train, 20% validation, 10% test
- Epochs: 50 with early stopping (patience=10)
- Batch Size: 16
- Optimizer: MuSGD (lr=0.002, momentum=0.9)
- Augmentations: Mosaic, random flip, HSV color, affine transforms

### OCR Setup

- Engine: EasyOCR
- Language: English
- Character Set: Alphanumeric + special characters
- GPU Acceleration: Enabled

## Training Results

The model shows consistent improvement across training:

| Epoch | Box Loss | mAP@50 | Precision | Recall |
|-------|----------|---------|-----------|--------|
| 1     | 1.234    | 93.1%   | 96.3%     | 89.1%  |
| 25    | 0.983    | 97.8%   | 98.6%     | 94.7%  |
| 50    | 0.838    | 97.7%   | 99.1%     | 94.7%  |

The model successfully handles:
- Various lighting conditions
- Different viewing angles
- Multiple plates in one image
- Partially occluded plates
- Different plate sizes and formats


## Dependencies

Core libraries used in this project:

```
torch==2.5.1
ultralytics==8.4.7
easyocr==1.7.2
opencv-python==4.12.0
numpy==1.26.4
pandas==2.2.3
matplotlib==3.9.1
seaborn==0.13.2
pillow==10.4.0
```

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - OCR engine
- [Kaggle Dataset](https://www.kaggle.com/datasets/barkataliarbab/license-plate-detection-dataset-10125-images) - Training data source

## Author

**Heshan Ranasinghe**  
Electronic and Telecommunication Engineering Undergraduate

- Email: hranasinghe505@gmail.com
- GitHub: [@DPHeshanRanasinghe](https://github.com/DPHeshanRanasinghe)
- LinkedIn: [Heshan Ranasinghe](https://www.linkedin.com/in/heshan-ranasinghe-988b00290)
