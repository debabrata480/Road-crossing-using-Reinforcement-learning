# GPU Setup Guide

## Overview

The entire project has been modified to support GPU acceleration. All training scripts will automatically detect and use CUDA if available, falling back to CPU if not.

## Changes Made

### 1. **agent.py** - Core Agent
- ✅ Enhanced GPU device detection
- ✅ Automatic CUDA selection when available
- ✅ All tensors moved to correct device
- ✅ Models automatically placed on GPU

### 2. **train_crossy.py** - Main Training Script
- ✅ GPU status information printing at startup
- ✅ Explicit device specification
- ✅ Shows GPU name, CUDA version, and memory

### 3. **experiment_progress_reward.py** - Experiment Script
- ✅ GPU detection and device setup
- ✅ GPU information printed once at start
- ✅ All models and tensors use GPU when available

### 4. **improve_success_rate.py** - Enhanced Training
- ✅ GPU support with device parameter
- ✅ GPU information displayed in configuration
- ✅ EnhancedDQNAgent uses GPU properly

### 5. **test_crossy.py** - Testing Script
- ✅ GPU support when loading models
- ✅ Models loaded onto GPU if available
- ✅ Faster inference on GPU

### 6. **model.py** - Neural Network Model
- ✅ Models automatically moved to device
- ✅ All forward passes use correct device

## How It Works

### Automatic GPU Detection

The code automatically detects GPU availability:

```python
if torch.cuda.is_available():
    device = torch.device('cuda')
    # Uses GPU
else:
    device = torch.device('cpu')
    # Falls back to CPU
```

### GPU Information Display

When training starts, you'll see:

```
============================================================
DEVICE INFORMATION
============================================================
✓ CUDA Available: Yes
  Device: NVIDIA GeForce RTX 3080
  CUDA Version: 12.1
  Total Memory: 10.00 GB
============================================================
```

Or if CUDA is not available:

```
============================================================
DEVICE INFORMATION
============================================================
✗ CUDA Available: No
  Using CPU for training
============================================================
```

## Installing CUDA Support

### For PyTorch with CUDA

1. **Check your NVIDIA GPU**:
   ```bash
   nvidia-smi
   ```

2. **Install PyTorch with CUDA support**:

   For CUDA 11.8:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   For CUDA 12.1:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   Or using conda:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

3. **Verify Installation**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```

   Should output:
   ```
   True
   NVIDIA GeForce RTX ...
   ```

## Performance Benefits

### GPU vs CPU

- **Training Speed**: 5-50x faster depending on GPU
- **Batch Processing**: More efficient parallel computation
- **Model Updates**: Faster gradient calculations
- **Large Networks**: Significant speedup with wider/deeper networks

### Expected Speedup

- **CPU (baseline)**: ~1x
- **Entry-level GPU (GTX 1060)**: ~3-5x
- **Mid-range GPU (RTX 3060)**: ~5-10x
- **High-end GPU (RTX 3080/3090)**: ~10-20x
- **Latest GPU (RTX 4090)**: ~20-50x

## Running with GPU

### All scripts automatically use GPU:

```bash
# Training - will use GPU if available
python train_crossy.py

# Experiments - will use GPU if available
python experiment_progress_reward.py

# Enhanced training - will use GPU if available
python improve_success_rate.py

# Testing - will use GPU if available
python test_crossy.py
```

No code changes needed! The scripts automatically detect and use GPU.

## Forcing CPU Usage

If you want to force CPU (even with GPU available), you can modify scripts:

```python
# In any training script, change:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To:
device = torch.device('cpu')
```

## Troubleshooting

### Issue: "CUDA out of memory"

**Solutions**:
1. Reduce batch size:
   ```python
   agent = DQNAgent(model, batch_size=32)  # Instead of 64
   ```

2. Reduce network size:
   ```python
   model = DQN(obs_shape, n_actions, hidden=256)  # Instead of 512
   ```

3. Clear GPU cache:
   ```python
   torch.cuda.empty_cache()
   ```

### Issue: CUDA version mismatch

**Solution**: Reinstall PyTorch with matching CUDA version:
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: DLL errors on Windows

**Solution**: Ensure CUDA toolkit is installed and matches PyTorch CUDA version.

## Monitoring GPU Usage

### During Training

Check GPU utilization:
```bash
# On Windows/Linux
nvidia-smi -l 1  # Updates every second
```

### Memory Usage

PyTorch will automatically manage GPU memory. Monitor with:
```python
import torch
if torch.cuda.is_available():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## Notes

- All models are automatically moved to GPU during initialization
- All tensors are created on the correct device
- Model saving/loading handles device mapping automatically
- If no GPU is available, everything falls back to CPU seamlessly
- No code changes needed to enable GPU - it's automatic!

## Verification

Test GPU availability:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
```

Your project is now fully GPU-enabled! 🚀

