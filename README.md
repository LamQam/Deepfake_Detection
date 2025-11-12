# Deepfake Detection

A deep learning-based project for detecting deepfake videos and images, focusing on robustness under social media compression. This project evaluates multiple state-of-the-art CNN models for deepfake detection and provides tools for model training and evaluation.

## Features

- Multiple model architectures: XceptionNet, EfficientNet (B4, B7), MesoNet, and ResNet
- Support for different datasets including DeepfakeTIMIT and WildDeepfake
- Pre-processing utilities for video frame extraction and data balancing
- Model evaluation and comparison tools
- Pre-trained models for quick deployment

## Datasets

### DeepfakeTIMIT (LQ)
- **Source**: [IDIAP Research Institute](https://www.idiap.ch/en/scientific-research/data/deepfaketimit)
- **Description**: Low-quality (64×64) deepfake videos
- **Content**: Manipulated videos based on VidTIMIT dataset

### VidTIMIT
- **Source**: [VidTIMIT](https://conradsanderson.id.au/vidtimit/)
- **Description**: Source dataset with real videos
- **Content**: Videos of 43 consented volunteers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Deepfake_Detection.git
cd Deepfake_Detection
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Deepfake_Detection/
├── VidTIMIT/               # Code for models trained on VidTIMIT dataset
│   ├── ResNet152.py        # ResNet model implementation
│   ├── efficientNetB4_model.py  # EfficientNetB4 implementation
│   ├── efficientNetB7.py   # EfficientNetB7 implementation
│   ├── mesoNet.py          # MesoNet implementation
│   └── xception_model.py   # Xception model implementation
├── VidTIMIT_deepfake_dataset/  # DeepfakeTIMIT dataset
├── WDFtest/                # WildDeepfake test implementations
├── WildDeepfake/           # Code for models trained on WildDeepfake dataset
├── weights/                # Trained model weights
│   ├──  VidTIMIT
│   └──  WD
├── pre_processing/         # Data preprocessing utilities
│   ├── balance_frames.py   # Utility for balancing dataset
│   └── extract_frame.py    # Frame extraction from videos
├── comparison.py           # Model comparison utilities
└── kaggle.py               # WildDeepfake competition utilities
└── charts/                 # Charts for model comparison
```

## Usage

### Data Preparation

1. **Extract Frames from Videos**:
```bash
python pre_processing/extract_frame.py --input_dir /path/to/videos --output_dir /path/to/save/frames
```

2. **Balance Dataset**:
```bash
python pre_processing/balance_frames.py --data_dir /path/to/frames --output_dir /path/to/balanced_data
```

### Training a Model

Example: Training Xception model on VidTIMIT dataset
```bash
python VidTIMIT/xception_model.py --data_dir /path/to/training_data --model_save_path /path/to/save/model
```

### Evaluating a Model

Example: Evaluating a pre-trained model
```bash
python VidTIMIT/xception_test.py --model_path /path/to/model --test_data /path/to/test_data
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- DeepfakeTIMIT dataset provided by IDIAP Research Institute
- VidTIMIT dataset
