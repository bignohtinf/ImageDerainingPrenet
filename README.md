# PReNet: Progressive Image Restoration Network for Rain Removal

A comprehensive implementation of progressive image restoration networks with multiple architectural enhancements for single image rain removal. This project implements and compares several variants of PReNet on two benchmark datasets: **Rain100H** and **Rain1400**.

## Overview

This repository contains implementations of:
- **PReNet** - Progressive Image Restoration Network (baseline)
- **PReNet + CBAM** - Enhanced with Convolutional Block Attention Module
- **PReNet + Perceptual Loss** - With perceptual loss for better visual quality
- **PReNet + CBAM + Perceptual Loss** - Combined attention and perceptual improvements
- **PReNet + CBAM + GAN** - Adversarial training with perceptual enhancement

All models are trained and evaluated on two standard rain removal benchmarks:
- **Rain100H** - 100 high-resolution rainy images with ground truth
- **Rain1400** - 1400 rainy images for comprehensive evaluation

## Project Structure

```
.
├── models/                    # Model implementations
│   ├── PReNet.py             # Base PReNet architecture
│   ├── PReNet_CBAM.py        # PReNet with CBAM attention
│   ├── CBAM.py               # Convolutional Block Attention Module
│   ├── perceptual_loss.py    # Perceptual loss implementation
│   ├── discriminator.py      # GAN discriminator for adversarial training
│   └── __init__.py
├── training/                  # Training scripts
│   ├── train_PreNet_rain1400.py  # Main training script
│   └── __init__.py
├── inference/                 # Inference and testing
│   ├── PreNet_1400.py        # Inference script
│   ├── test/                 # Test images
│   ├── output/               # Inference outputs
│   └── __init__.py
├── evaluation/               # Evaluation metrics
│   ├── evaluate.py           # Evaluation pipeline
│   ├── metric_calculator.py  # PSNR, SSIM calculation
│   ├── psnr.py              # PSNR metric
│   ├── ssim.py              # SSIM metric
│   ├── batch_PSNR.py        # Batch PSNR evaluation
│   └── __init__.py
├── scripts/                  # Utility scripts
│   ├── data/                # Data processing
│   │   ├── dataset.py       # Dataset loading and preparation
│   │   ├── augmentation.py  # Data augmentation pipeline
│   │   └── __init__.py
│   ├── checkpoint/          # Checkpoint management
│   │   ├── load_checkpoint.py
│   │   ├── save_training_history.py
│   │   ├── load_training_history.py
│   │   ├── cleanup_old_models.py
│   │   └── __init__.py
│   └── __init__.py
├── utils/                    # Utility functions
│   ├── helpers.py           # Helper functions
│   ├── image_utils.py       # Image processing utilities
│   ├── metrics.py           # Metric utilities
│   ├── print_network.py     # Network architecture printing
│   └── __init__.py
├── configs/                  # Configuration files
│   └── config.py            # Training configuration
├── data/                     # Dataset directory
│   ├── raw/                 # Raw datasets
│   │   ├── rain100H/
│   │   └── rain1400/
│   └── processed/           # Processed datasets
├── notebooks/               # Jupyter notebooks
│   └── visualization.ipynb  # Results visualization
├── reference/               # Reference papers
└── requirements.txt         # Python dependencies
```

## Key Features

### Model Variants

1. **PReNet (Baseline)**
   - Progressive recurrent restoration with 6 iterations
   - Recurrent convolutional units with gating mechanisms
   - Multi-stage supervision for improved convergence

2. **PReNet + CBAM**
   - Integrates Convolutional Block Attention Module
   - Channel and spatial attention mechanisms
   - Improved feature representation learning

3. **PReNet + Perceptual Loss**
   - VGG-based perceptual loss for better visual quality
   - Combines L1 loss with perceptual similarity
   - Better preservation of image structure and texture

4. **PReNet + CBAM + Perceptual Loss**
   - Combines attention mechanisms with perceptual loss
   - Enhanced feature learning with attention
   - Superior visual quality and PSNR/SSIM metrics

5. **PReNet + CBAM + GAN**
   - Adversarial training with discriminator
   - Perceptual enhancement through adversarial loss
   - State-of-the-art visual quality

### Training Features

- **Multi-stage Supervision**: Loss computed at all recurrent iterations
- **Curriculum Learning**: Progressive augmentation strategy across training stages
- **Data Augmentation**: 
  - Basic augmentation (rotation, flip)
  - Rain streak augmentation
  - MixUp augmentation
- **Learning Rate Scheduling**: MultiStepLR with milestone-based decay
- **Checkpoint Management**: Automatic model saving and cleanup
- **Training History Tracking**: Epoch-wise metrics logging

### Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- Batch evaluation support

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- PyTorch 1.9+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ctthong18/ImageDerainingPrenet.git
cd ImageDerainingPrenet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets:
   - Rain100H: [Download Link](https://github.com/csdwren/PReNet)
   - Rain1400: [Download Link](https://github.com/csdwren/PReNet)

4. Extract datasets to `data/raw/`:
```
data/raw/
├── rain100H/
│   ├── rain/
│   └── norain/
└── rain1400/
    ├── rain/
    └── norain/
```

## Configuration

Edit `configs/config.py` to customize training parameters:

```python
# Data paths
data_path = 'data/raw/rain1400'
train_data_path = 'data/processed/rain1400'

# Training parameters
batch_size = 8
epochs = 100
lr = 0.001
recurrent_iter = 6

# Augmentation
use_augmentation = True
use_curriculum = True

# Hardware
use_gpu = True

# Model saving
save_path = 'inference/'
save_freq = 10
```

## Training

### Train PReNet on Rain1400:

```bash
python training/train_PreNet_rain1400.py
```

### Train with Custom Configuration:

Edit `configs/config.py` and run:
```bash
python training/train_PreNet_rain1400.py
```

### Resume Training:

The training script automatically resumes from the latest checkpoint if available.

## Inference

### Single Image Inference:

```bash
python inference/PreNet_1400.py --input <image_path> --model <model_path> --output <output_path>
```

### Batch Inference:

```bash
python inference/PreNet_1400.py --input_dir <directory> --model <model_path> --output_dir <output_directory>
```

## Demo Examples

### Example 1: Quick Inference with Pre-trained Model

```python
import torch
from models.PReNet import PReNet
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = PReNet(recurrent_iter=6, use_GPU=True)
model.load_state_dict(torch.load('inference/net_best.pth'))
model.eval()

# Load and preprocess image
image_path = 'inference/test/rainy_image.jpg'
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.ToTensor(),
])
input_tensor = transform(image).unsqueeze(0).cuda()

# Inference
with torch.no_grad():
    output, _ = model(input_tensor)
    output = torch.clamp(output, 0, 1)

# Save result
output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
output_image.save('inference/output/derrained_image.jpg')
print("✓ Derrained image saved to: inference/output/derrained_image.jpg")
```

### Example 2: Batch Processing with Metrics

```python
import torch
import os
from models.PReNet import PReNet
from evaluation.metric_calculator import MetricCalculator
from PIL import Image
import torchvision.transforms as transforms

# Initialize
model = PReNet(recurrent_iter=6, use_GPU=True)
model.load_state_dict(torch.load('inference/net_best.pth'))
model.eval()
metric_calc = MetricCalculator()

# Process batch
test_dir = 'inference/test'
output_dir = 'inference/output'
os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([transforms.ToTensor()])
psnr_list = []
ssim_list = []

for img_name in os.listdir(test_dir):
    if img_name.endswith(('.jpg', '.png')):
        # Load image
        img_path = os.path.join(test_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).cuda()
        
        # Inference
        with torch.no_grad():
            output, _ = model(input_tensor)
            output = torch.clamp(output, 0, 1)
        
        # Save result
        output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
        output_path = os.path.join(output_dir, f'derrained_{img_name}')
        output_image.save(output_path)
        
        print(f"✓ Processed: {img_name}")

print(f"\n✓ All images processed and saved to: {output_dir}")
```

### Example 3: Compare Model Variants

```python
import torch
from models.PReNet import PReNet
from models.PReNet_CBAM import PReNet_CBAM
from PIL import Image
import torchvision.transforms as transforms
import time

# Load test image
image_path = 'inference/test/rainy_image.jpg'
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([transforms.ToTensor()])
input_tensor = transform(image).unsqueeze(0).cuda()

# Model variants to compare
models = {
    'PReNet': ('inference/net_best.pth', PReNet),
    'PReNet_CBAM': ('inference/net_cbam_best.pth', PReNet_CBAM),
}

print("Comparing model variants...\n")
print(f"{'Model':<20} {'Inference Time (ms)':<20} {'Output Size':<15}")
print("-" * 55)

for model_name, (model_path, model_class) in models.items():
    # Load model
    model = model_class(recurrent_iter=6, use_GPU=True)
    if torch.cuda.is_available() and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Measure inference time
    with torch.no_grad():
        start = time.time()
        output, _ = model(input_tensor)
        end = time.time()
        
        inference_time = (end - start) * 1000  # Convert to ms
        output_size = output.shape
        
        print(f"{model_name:<20} {inference_time:<20.2f} {str(output_size):<15}")

print("\n✓ Comparison complete")
```

### Example 4: Real-time Inference from Webcam (Optional)

```python
import torch
import cv2
from models.PReNet import PReNet
import torchvision.transforms as transforms

# Load model
model = PReNet(recurrent_iter=6, use_GPU=True)
model.load_state_dict(torch.load('inference/net_best.pth'))
model.eval()

# Prepare transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit, 's' to save frame")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    input_tensor = transform(rgb_frame).unsqueeze(0).cuda()
    
    # Inference
    with torch.no_grad():
        output, _ = model(input_tensor)
        output = torch.clamp(output, 0, 1)
    
    # Convert back to numpy
    output_np = (output.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    # Display
    cv2.imshow('Original', frame)
    cv2.imshow('Derrained', output_bgr)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('inference/output/frame.jpg', output_bgr)
        print("✓ Frame saved")

cap.release()
cv2.destroyAllWindows()
```

## Evaluation

### Evaluate on Test Set:

```bash
python evaluation/evaluate.py --model <model_path> --test_dir <test_directory>
```

### Batch PSNR Calculation:

```bash
python evaluation/batch_PSNR.py --restored_dir <restored_images> --reference_dir <reference_images>
```

## Results

Results will be documented here after training completion. Metrics include:
- PSNR (dB)
- SSIM
- Visual quality comparison
- Inference time

## Architecture Details

### PReNet Core Components

- **Recurrent Convolutional Units**: 6 iterations of progressive refinement
- **Gating Mechanisms**: Input and forget gates for recurrent connections
- **Multi-scale Processing**: Feature extraction at multiple scales
- **Residual Connections**: Skip connections for improved gradient flow

### CBAM Attention Module

- **Channel Attention**: Adaptive feature recalibration
- **Spatial Attention**: Spatial feature refinement
- **Lightweight Design**: Minimal computational overhead

### Perceptual Loss

- **VGG-based Features**: Pre-trained VGG19 for perceptual similarity
- **Multi-layer Loss**: Loss computed at multiple VGG layers
- **Content Preservation**: Better structural and textural fidelity

## Training Details

- **Optimizer**: Adam (β₁=0.9, β₂=0.999)
- **Learning Rate**: 0.001 with MultiStepLR scheduling
- **Loss Function**: Multi-stage L1 + MSE (0.5 each)
- **Batch Size**: 8
- **Epochs**: 100
- **Data Augmentation**: Progressive curriculum learning

## Performance Considerations

- **GPU Memory**: ~4GB for batch size 8 with 256×256 images
- **Training Time**: ~24-48 hours per model on single GPU
- **Inference Speed**: ~50-100ms per image (256×256)

## References

- **PReNet**: [Progressive Image Restoration Network for Single Image Deraining](https://github.com/csdwren/PReNet)
- **CBAM**: [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- **Perceptual Loss**: [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- **GAN**: [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)

## Citation

If you use this code in your research, please cite the original PReNet paper:

```bibtex
@inproceedings{ren2019progressive,
  title={Progressive Image Restoration through Conditional Treatment},
  author={Ren, Dongwei and Zuo, Wangmei and Hu, Qinghua and Zhu, Pengfei and Meng, Deyu},
  booktitle={CVPR},
  year={2019}
}
```

## License

This project is provided for research and educational purposes.

## Acknowledgments

- Original PReNet implementation: [csdwren/PReNet](https://github.com/csdwren/PReNet)
- CBAM implementation inspired by official repository
- Evaluation metrics based on standard image quality assessment

## Contact & Support

For questions or issues, please open an issue in the repository.

---

**Last Updated**: January 2026
