# Deepfake Detection

A deep learning-based project for detecting deepfake videos and images, focusing on robustness under social media compression. Evaluates state-of-the-art models (XceptionNet, EfficientNetB4, EfficientNetB7, MesoNet, ResNet) on the DeepfakeTIMIT (LQ) dataset and explores hybrid approaches for real-world deployment.

## Datasets
- **[DeepfakeTIMIT]** (https://www.idiap.ch/en/scientific-research/data/deepfaketimit) (LQ)**: Low-quality (64×64) deepfake videos 
- **[VidTIMIT]** (https://conradsanderson.id.au/vidtimit/): Source dataset with real videos of 43 consented volunteers

## Installation
```bash
git clone https://github.com/LamQam/Deepfake_Detection.git
cd Deepfake_Detection
pip install -r requirements.txt