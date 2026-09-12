# Model Architecture Selection & Technical Trade-Off Analysis

**Project**: USIP — Underwater Sonar Intelligence Platform  
**Problem Statement**: SIH26057 — AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery  
**Document**: Architecture Justification & Comparative Study  

---

## 1. Operational & Environmental Constraints

Side-scan sonar target detection operates under fundamentally different physical and computing regimes than terrestrial RGB computer vision:
1. **Compute Envelope**: Marine operations deploy models on Autonomous Underwater Vehicles (AUVs) or towed sonars with low-power embedded processors (NVIDIA Jetson / compact GPUs) and constrained battery power.
2. **Real-Time Along-Track Processing**: Pings arrive continuously at 10–30 Hz; high-latency batch models cause buffer overrun or missed target re-acquisition passes.
3. **Multi-Source Unification**: The target taxonomy spans three distinct physical signatures:
   - *Shipwrecks*: Large-scale spatial extent (tens to hundreds of meters) with multi-point highlights and stern acoustic cast shadows.
   - *Subsea Pipelines*: Long, continuous, high-aspect-ratio linear structures with continuous shadow tracks.
   - *Anthropogenic Debris*: Compact, discrete, high-backscatter anomalies against varying sediment textures.
4. **Local Hardware Constraint**: The deployment environment features an NVIDIA RTX 3050 Laptop GPU (4,096 MB VRAM). Any candidate architecture must comfortably fit within 4 GB VRAM during training and inference without thrashing or requiring out-of-core memory swapping.

---

## 2. Comparative Evaluation of Candidate Architectures

| Criteria | Faster R-CNN (ResNet-50 FPN) | RT-DETR (ResNet-50-vd) | YOLOv8 / YOLO11 Nano (`yolov8n`) | YOLOv8 Small (`yolov8s`) |
| :--- | :--- | :--- | :--- | :--- |
| **Model Footprint** | ~160 MB | ~120 MB | **~6.5 MB** | ~22 MB |
| **Parameters** | 41.5 M | 32 M | **3.2 M** | 11.2 M |
| **FLOPs (at 640×640)** | ~180 GFLOPs | ~86 GFLOPs | **8.7 GFLOPs** | 28.6 GFLOPs |
| **Training VRAM (Batch 8)** | ~7.2 GB (OOM on 4GB) | ~6.5 GB (OOM on 4GB) | **~2.4 GB (Clean fit)** | ~4.1 GB (Borderline) |
| **Edge Inference Latency** | 45–65 ms (CPU: 280 ms) | 22–35 ms (CPU: 180 ms) | **4–7 ms (CPU: 32 ms)** | 8–14 ms (CPU: 65 ms) |
| **Multi-Scale Feature Hierarchy** | Feature Pyramid (FPN) | Hybrid Encoder + Deformable Attention | **Path Aggregation Neck (PANet) with C2f** | PANet with C2f |
| **Small Object Sensitivity** | High (Two-stage RPN) | High (Transformer Query) | **High (Anchor-free decoupled head)** | High |
| **Acoustic Generalization** | Prone to overfitting on small sonar datasets | Requires massive data for attention convergence | **Resilient with acoustic-safe augmentation** | Resilient |

---

## 3. Detailed Justification for Selected Model (`YOLOv8n` / `YOLO11n`)

### Why NOT Faster R-CNN?
- While Faster R-CNN's Region Proposal Network (RPN) is historically reliable for terrestrial imagery, its 41.5M parameters exceed our 4 GB VRAM limit during backpropagation with batch size $\ge 4$.
- Its two-stage design introduces high inference latency (>45 ms on GPU, >280 ms on CPU), which violates the real-time continuous ping requirement of AUV forward-looking and side-scan sonars.
- Two-stage anchor proposal heads tend to overfit when trained on hundreds rather than tens of thousands of examples.

### Why NOT RT-DETR?
- Real-Time DEtection TRansformer (RT-DETR) provides excellent global context modeling via deformable attention.
- However, transformer-based query decoders require substantially larger training datasets to learn spatial inductive biases from scratch. On our 4,453 unified acoustic samples, transformer queries struggle with convergence stability without massive self-supervised pretraining.
- VRAM requirements for attention keys and values exceed the local 4 GB hardware envelope during multi-scale feature fusion.

### Why `YOLOv8n` / `YOLO11n` is the Scientifically Defensible Choice:
1. **Anchor-Free Decoupled Head**: Decouples classification from bounding box regression. Sonar specular highlights frequently have diffuse or asymmetric acoustic shadows where anchor-based priors fail. The anchor-free regression head accurately predicts non-standard acoustic footprints.
2. **Parameter Efficiency**: At only 3.2M parameters and 8.7 GFLOPs, it trains in under 2.4 GB VRAM, allowing training with batch size 8 or 16 on our NVIDIA RTX 3050 GPU.
3. **C2f Cross-Stage Partial Neck**: The multi-scale gradient flow paths retain subtle low-intensity backscatter contrast from small debris while resolving large-scale hull structures across multiple downsampling tiers.
4. **Latency Profile**: Achieves <7 ms inference on GPU and ~32 ms on standard dual-core CPU, enabling true edge deployment and asynchronous queuing in FastAPI without blocking worker threads.

---

## 4. Training Strategy & Hyperparameters

- **Base Architecture**: `yolov8n.pt` pretrained backbone
- **Target Resolution**: 640 × 640 pixels
- **Batch Size**: 8 (optimized for 4GB VRAM)
- **Optimizer**: AdamW (weight decay $0.0005$, momentum $0.937$)
- **Learning Rate**: Initial $\text{lr}_0 = 0.001$, Cosine Annealing to $\text{lr}_f = 0.01 \cdot \text{lr}_0$
- **Acoustic Augmentation**:
  - Horizontal flip (Port/Starboard symmetry): $p = 0.5$
  - Vertical flip: $p = 0.0$ (strictly disabled; violates along-track temporal sequence)
  - Color jitter / HSV: minimal ($h=0.015, s=0.0, v=0.2$) to preserve grayscale acoustic backscatter
  - Mosaic: disabled during final epochs to avoid artificial edge discontinuities in acoustic shadows
- **Leakage Prevention**: Strictly disjoint evaluation against unseen shipwreck sites, unseen survey dates, and unseen pipeline chunks.

