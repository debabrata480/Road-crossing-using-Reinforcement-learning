# Installation Guide

## Python Version Requirement

This project requires **Python 3.8 through 3.12**. 

**Python 3.13 and 3.14 are NOT compatible** because:
- Ursina game engine depends on Panda3D
- Panda3D does not support Python 3.13+
- Latest Panda3D release: supports up to Python 3.12

## Quick Setup

### Option 1: Using Python 3.11 or 3.12 (Recommended)

1. **Install Python 3.11 or 3.12**
   - Download from [python.org](https://www.python.org/downloads/)
   - Or use the Windows Store version
   - Make sure to add to PATH during installation

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import ursina, torch, numpy, matplotlib; print('All dependencies installed!')"
   ```

### Option 2: Using pyenv (Multiple Python Versions)

If you already have Python 3.14 and want to keep it:

1. **Install pyenv-win** (Windows)
   ```powershell
   git clone https://github.com/pyenv-win/pyenv-win.git %USERPROFILE%\.pyenv
   ```

2. **Install Python 3.12**
   ```bash
   pyenv install 3.12.7
   pyenv local 3.12.7
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Option 3: Using Conda/Miniconda

1. **Create environment with Python 3.11**
   ```bash
   conda create -n crossy python=3.11
   conda activate crossy
   ```

2. **Install PyTorch from conda** (optional, better CUDA support)
   ```bash
   conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

3. **Install other dependencies**
   ```bash
   pip install ursina numpy matplotlib pillow
   ```

## Troubleshooting

### Issue: "No module named 'panda3d'"
**Solution**: Install panda3d explicitly:
```bash
pip install panda3d
```

### Issue: "No module named 'ursina'"
**Solution**: Install ursina and its dependencies:
```bash
pip install ursina panda3d
```

### Issue: ImportError for asset files
**Solution**: Make sure you're running from the project root directory:
```bash
cd path/to/draft1
python game_crossy.py
```

### Issue: PyTorch CUDA errors
**Solution**: If you don't have CUDA, PyTorch will use CPU automatically. To force CPU-only:
```python
# This happens automatically if CUDA is not available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Issue: Ursina window doesn't open
**Solution**: 
- Check that display/GUI is available
- On remote servers, this won't work (use headless mode)
- On Windows, make sure graphics drivers are up to date

## Verification Steps

After installation, verify everything works:

1. **Test game environment**
   ```bash
   python game_crossy.py
   ```
   - Should open a 3D window with the game
   - Press ESC or close window to exit

2. **Test training** (just a few episodes)
   ```bash
   python train_crossy.py
   ```
   - Will train for 1000 episodes by default
   - Reduce episodes in the script if just testing

3. **Test with pre-trained model**
   ```bash
   python test_crossy.py
   ```
   - Should load model and play visually
   - Model file should exist: `models/dqn_crossy.pth`

## System Requirements

- **OS**: Windows 10+, Linux, or macOS
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: Optional but recommended for faster training
- **Disk**: ~1GB for Python packages and models

## Next Steps

Once installed, see [README.md](README.md) for usage instructions.




