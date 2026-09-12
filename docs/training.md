# Model Training Pipeline & Reproducibility Guide

**System**: USIP — Underwater Sonar Intelligence Platform (SIH26057)

---

## 1. Local Training Pipeline

The supervised target detector is trained on the compiled unified sonar dataset using the local GPU:

### Command
```powershell
$env:PYTHONPATH="."
D:\usip_env\Scripts\python.exe -m ml.training.train_detector --epochs 15 --batch 8 --device 0
```

### Execution Details
- **Architecture**: YOLOv8n (Decoupled anchor-free detection head, 3.2M parameters)
- **Dataset YAML**: `D:/USIP_DATA/unified_sonar_dataset/dataset.yaml`
- **Trained Classes**:
  1. `Shipwreck` (Class index 0)
  2. `Pipeline` (Class index 1)
  3. `Debris` (Class index 2)
- **Checkpoint Location**: `ml/detection/weights/best.pt`
- **Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)

---

## 2. Cloud GPU Training Bridge (Google Colab)

For multi-gigabyte or extended training runs (100+ epochs) on high-end enterprise GPUs (Tesla T4 / A100), the repository provides an end-to-end Jupyter notebook:
- **Notebook Path**: `notebooks/USIP_Colab_Model_Training.ipynb`
- **Workflow**:
  1. Open Google Colab and select T4/A100 GPU runtime.
  2. Upload `notebooks/USIP_Colab_Model_Training.ipynb`.
  3. Mount dataset zip archives (`AI4Shipwrecks.zip`, `AquaScan-1K.zip`, `SubPipeMini.zip`).
  4. Run all cells to execute training and export `best.pt`.

