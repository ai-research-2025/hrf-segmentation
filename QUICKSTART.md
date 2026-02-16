# Quick Start Guide - HRF Segmentation

Get started with HRF segmentation in 3 minutes!

## Choose Your Path

### Path 1: Try the Live Demo (Fastest)

**No installation required!**

[Open HuggingFace Space](https://huggingface.co/spaces/AI-RESEARCHER-2024/HRF_Segmentation)

1. Upload an OCT image
2. Select model (U-Net or Attention U-Net)
3. Click "Run Segmentation"
4. View and download results

---

### Path 2: Run Evaluation (5 minutes)

**Evaluate models locally using Jupyter Notebooks**

1. Clone the repository: `git clone https://github.com/ai-research-2025/hrf-segmentation.git`
2. Install dependencies: `pip install -r evaluation/requirements.txt`
3. Download pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1J78I28EzEXMD3jNrusWfuQoCFeFhtZsT)
4. Open the notebook: `jupyter notebook evaluation/HRF_Segmentation_Evaluation.ipynb`
5. Update paths and run all cells to get metrics and visualizations!

**What you need:**
- Python 3.8+
- Test images in JPEG format
- Ground truth masks in .ome.tiff format
- Pre-trained models (.pth)

---

### Path 3: Train Your Own Model (Advanced)

**Train from scratch on your dataset**

```bash
# Clone repository
git clone https://github.com/ai-research-2025/hrf-segmentation.git
cd hrf-segmentation

# Install dependencies
cd training
pip install -r requirements.txt

# Prepare your data (images/ and masks/ folders)
# Then train!
python hrfunet.py        # For U-Net
python hrf-aunet.py      # For Attention U-Net
```

**Requirements:**
- NVIDIA GPU (8GB+ VRAM)
- Python 3.8+
- Training dataset with images and masks

---

## Download Pre-trained Models

**All trained models available here:**

[**Google Drive - Model Weights**](https://drive.google.com/drive/folders/1J78I28EzEXMD3jNrusWfuQoCFeFhtZsT)

Contains:
- unet_best_model.pth (~355 MB)
- aunet_best_model.pth (~361 MB)
- Training scripts

---

## Learn More

- **Full Documentation**: See main [README.md](README.md)
- **Training Guide**: [training/README.md](training/README.md)
- **Evaluation Guide**: [evaluation/README.md](evaluation/README.md)

---

## Need Help?

- Check the [main README](README.md) for detailed instructions
- [Open an issue](https://github.com/ai-research-2025/hrf-segmentation/issues)
- [Start a discussion](https://github.com/ai-research-2025/hrf-segmentation/discussions)

---

## Star This Repo

Found this helpful? Give it a star to show your support!
