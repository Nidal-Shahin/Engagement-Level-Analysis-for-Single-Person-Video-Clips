---
title: Engagement Level Analysis For Single Person Video Clips
emoji: 📊
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Engagement Analysis System

An optimized deep learning application for analyzing and visualizing human engagement levels in video content using **Vision Transformers (ViT)** and **GRU** architectures.

---

## 🚀 Features

- **Adversarial ViT Backbone**: High-accuracy facial feature extraction  
- **Temporal Analysis**: GRU integration for consistent engagement tracking over time  
- **Real-time Visualization**: Dynamic bounding boxes with color-coded engagement levels  
- **Performance Optimized**: Batch processing and frame sampling for faster inference  

---

## 🛠️ Installation

### 1. Requirements

Ensure you have **Python 3.10+** installed.  
Install the necessary dependencies using:

```bash
pip install -r requirements.txt
```

### 2. File Structure

The system expects the following data structure (configured for Kaggle environments):

```bash
/kaggle/input/
├── adversarial-vit-with-discriminator-and-gru/
│   └── best_progressive_model.pth
├── yunet-facial-landmarks-extractor/
│   └── face_detection_yunet (1).onnx
└── test-samples/
    └── Class_X_Example.mp4
```

## 📊 Engagement Levels

The system classifies engagement into four categories based on the calculated score 
***𝐿***: 
```bash
Level	Range	Visualization Color
Very High	
𝐿
≥
2.5
L≥2.5	Green
High	
1.5
≤
𝐿
<
2.5
1.5≤L<2.5	Yellow / Cyan
Low	
0.5
≤
𝐿
<
1.5
0.5≤L<1.5	Orange
Very Low	
𝐿
<
0.5
L<0.5	Red
```

## 💻 Usage
### Running the UI

Execute the main script to launch the Gradio web interface:
```bash
python app.py
```

### Advanced Settings

1. **Batch Size:** Balance speed vs. VRAM usage (Default: 12)

2. **Smoothness:** Control the temporal averaging filter (Default: 5)

3. **Analysis FPS:** Adjust the density of inference frames (Default: 5)

## 📜 Requirements List
```bash
gradio
opencv-python-headless
torch
torchvision
timm
albumentations
numpy
```