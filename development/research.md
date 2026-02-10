# 🧪 Development & Research Roadmap

This folder contains the complete evolution of the project, from initial data preprocessing to the final Adversarial ViT implementation. These notebooks were developed and executed on Kaggle.

---

## 🏗️ Phase 1: Preprocessing & Detection
* **[Data Preprocessing](https://www.kaggle.com/code/nidalshahin/part-0-data-preprocessing)**: Initial cleaning, formatting, and organization of the raw video dataset.
* **[Face Detection & Bounding Boxes](https://www.kaggle.com/code/nidalshahin/part-1-1-face-detection-and-bbox-generation)**: Implementation of YuNet for robust facial localization and coordinate generation.
* **[Face Cropping & Storage](https://www.kaggle.com/code/nidalshahin/part-1-2-face-cropping-and-saving)**: Automated extraction and saving of facial regions to streamline model input.

## 🔍 Phase 2: Architectural Benchmarking (Old Path)
Before settling on the final architecture, several backbones were tested for performance and efficiency:
* **[ConvNeXt Tiny](https://www.kaggle.com/code/nidalshahin/part-2-1-trying-convnext-tiny)**: Testing modern pure-CNN architectures.
* **[Swin Transformer](https://www.kaggle.com/code/nidalshahin/part-2-2-fine-tuning-swin-transformer)**: Exploring hierarchical vision transformers.
* **[ShuffleNet (Lightweight)](https://www.kaggle.com/code/nidalshahin/part-2-3-fine-tuning-lightweight-shufflenet-cnn)**: Investigating mobile-friendly CNN alternatives.

## 🔬 Phase 3: Optimization & Feature Engineering
* **[Class Imbalance Correction](https://www.kaggle.com/code/nidalshahin/part-2-4-1-fixing-class-imbalance-from-the-root)**: Strategic oversampling and loss adjustment to handle skewed engagement distributions.
* **[Label Refinement](https://www.kaggle.com/code/nidalshahin/part-2-4-2-feature-engineering-label-refinement)**: Enhancing ground-truth quality through engineered feature sets.
* **[Visualization & Error Analysis](https://www.kaggle.com/code/nidalshahin/part-2-4-3-visualization-and-analysis)**: Deep dive into model predictions to identify edge cases.
* **[MobileNet Experiment](https://www.kaggle.com/code/nidalshahin/part-3-1-mobilenet)**: Testing standard lightweight backbones.
* **[ResNet34 + Temporal Attention](https://www.kaggle.com/code/nidalshahin/part-3-2-resnet34-with-temporal-attention)**: Early experiments with temporal modeling using attention mechanisms.

## 🏆 Phase 4: Final Production Pipeline
* **[Uniform Frame Extraction](https://www.kaggle.com/code/nidalshahin/part-4-0-uniform-frames-extraction)**: Standardizing video input by extracting uniform temporal samples for the ViT-GRU pipeline.
* **[Adversarial ViT Training](https://www.kaggle.com/code/nidalshahin/part-4-1-adversarial-vit)**: The core training script for the final production model utilized in this Space.
