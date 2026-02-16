# Training HRF Segmentation Models

This directory contains the training scripts for both U-Net and Attention U-Net models.

## Files

- hrfunet.py - Complete training pipeline for U-Net
- hrf-aunet.py - Complete training pipeline for Attention U-Net
- requirements.txt - Python dependencies

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Organize your data in the following structure:

```
dataset/
├── images/
│   ├── image001.jpeg
│   ├── image002.jpeg
│   └── ...
└── masks/
    ├── image001_HRF.ome.tiff
    ├── image002_HRF.ome.tiff
    └── ...
```

### 3. Train U-Net

```bash
python hrfunet.py
```

### 4. Train Attention U-Net

```bash
python hrf-aunet.py
```

## Configuration

Both scripts include configuration sections at the top where you can modify:

- Data paths: Update IMAGES_DIR and MASKS_DIR
- Hyperparameters:
  - Learning rate
  - Batch size
  - Number of epochs
  - Train/val/test split ratios
- Model architecture:
  - Base filters
  - Use bilinear upsampling vs. transposed convolutions

## Training Details

### U-Net (hrfunet.py)

- Architecture: Classic U-Net with 5 levels
- Preprocessing: Simple 0-255 normalization
- Loss: Binary Cross-Entropy + Dice Loss
- Optimizer: Adam (lr=1e-4)
- Output: unet_best_model.pth

### Attention U-Net (hrf-aunet.py)

- Architecture: U-Net with attention gates
- Preprocessing: CLAHE + Z-Score normalization (ImageNet stats)
- Loss: Binary Cross-Entropy + Dice Loss
- Optimizer: Adam (lr=1e-4)
- Output: aunet_best_model.pth

## Hardware Requirements

- GPU: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- RAM: 16GB+ recommended
- Storage: ~10GB for checkpoints and logs

## Monitoring Training

Both scripts print training progress including:
- Epoch number
- Training loss
- Validation loss
- Validation Dice coefficient
- Best model checkpointing

## Outputs

After training, you'll find:
- {model_name}_best_model.pth - Best model checkpoint (based on validation Dice)
- Training logs in console output

## Tips for Better Results

1. Data Augmentation: Both scripts include random flips, rotations, and elastic transforms
2. Early Stopping: Models save only when validation Dice improves
3. Learning Rate: Start with 1e-4, reduce if loss plateaus
4. Batch Size: Adjust based on your GPU memory (4-16 typical)

## Troubleshooting

### Out of Memory Errors
- Reduce batch size
- Reduce image resolution
- Use gradient checkpointing

### Poor Convergence
- Check data normalization
- Verify masks are binary (0 and 1)
- Increase number of epochs
- Try different learning rate

## Pre-trained Models

Don't want to train from scratch? Download our pre-trained weights:

[**Google Drive - Pre-trained Models**](https://drive.google.com/drive/folders/1J78I28EzEXMD3jNrusWfuQoCFeFhtZsT)
