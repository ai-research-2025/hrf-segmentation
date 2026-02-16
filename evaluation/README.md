# HRF Segmentation Evaluation

This directory contains the evaluation notebook for testing U-Net and Attention U-Net models on the HRF segmentation task.

## Files

- HRF_Segmentation_Evaluation.ipynb - Complete evaluation pipeline
- requirements.txt - Python dependencies

## Quick Start (Local Jupyter)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Pre-trained Models**
   Download weights from [Google Drive](https://drive.google.com/drive/folders/1J78I28EzEXMD3jNrusWfuQoCFeFhtZsT) and place them in your working directory.

3. **Start Jupyter**
   ```bash
   jupyter notebook HRF_Segmentation_Evaluation.ipynb
   ```

4. **Run Evaluation**
   Update the image and mask paths in the configuration cell and run all cells.

## What You'll Get

The evaluation notebook generates:

### 1. Quantitative Metrics

- Dice Coefficient
- IoU (Jaccard Index)
- Precision, Recall, F1-Score
- Specificity, Sensitivity
- AUC-ROC

### 2. Visualizations

- ROC curves comparing both models
- Confusion matrices
- Side-by-side predictions
- Performance comparison bar charts
- Prediction overlays on original images

### 3. Detailed Results

- Per-image metrics
- Statistical summaries
- Confusion matrix breakdown (TP, TN, FP, FN)

## Requirements

- Pre-trained Models: Download from [Google Drive](https://drive.google.com/drive/folders/1J78I28EzEXMD3jNrusWfuQoCFeFhtZsT)
- Test Images: OCT retinal images in JPEG format
- Ground Truth Masks: Corresponding HRF masks in .ome.tiff format

## Dataset Structure

```
your_data/
├── HRF_IMAGES/
│   ├── image01.jpeg
│   ├── image02.jpeg
│   └── ...
└── HRF_MASKS/
    ├── image01_HRF.ome.tiff
    ├── image02_HRF.ome.tiff
    └── ...
```

## Configuration

In the notebook, update these paths:

```python
BASE_DIR = 'path/to/your/data'
IMAGES_DIR = os.path.join(BASE_DIR, 'HRF_IMAGES')
MASKS_DIR = os.path.join(BASE_DIR, 'HRF_MASKS')
UNET_MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints/unet_best_model.pth')
AUNET_MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints/aunet_best_model.pth')
```

## Key Features

### Preprocessing Fixes Applied

The notebook includes the latest preprocessing fixes:
- U-Net: Simple 0-255 normalization (float32)
- Attention U-Net: CLAHE + Z-Score normalization (float32)

### Model-Specific Inference

Each model uses its own preprocessing pipeline matching how it was trained.

### Comprehensive Metrics

All standard segmentation metrics plus ROC analysis and visualizations.

## Outputs

Results are saved to the evaluation_results directory:

- confusion_matrices.png
- roc_curves.png
- metrics_comparison.png
- predictions_visualization.png
- evaluation_summary.txt

## Tips

1. Use GPU: Ensure you have a CUDA-enabled GPU for faster inference.
2. Test Split: Ensure your test data matches the model's expectations.
3. Batch Processing: Notebook processes all test images efficiently.
4. Reproducibility: Random seed is set to 42.

## Troubleshooting

### Import Errors
```bash
pip install --upgrade torch torchvision
```

### CUDA Out of Memory
- Use smaller images or reduce processing batch size if applicable.
- Close other GPU-intensive applications.

### Model Loading Errors
- Verify checkpoint paths are correct.
- Ensure model files are not corrupted.
- Check PyTorch version compatibility.

## Example Output

```
==========================================================
EVALUATION RESULTS
==========================================================
Metric               U-Net                Attention U-Net     
----------------------------------------------------------
Dice                 0.2065               0.5017
IoU                  0.1151               0.3348
Precision            0.3842               0.5725
Recall               0.1412               0.4464
F1                   0.2065               0.5017
Specificity          0.9999               0.9999
AUC                  0.3054               0.6045
----------------------------------------------------------
```
