# Setup Instructions for Your System

## Current Status

✅ **Core DQN Components**: Working perfectly!
- PyTorch, NumPy, Matplotlib installed and tested
- Model and Agent code imports successfully
- Neural network forward pass works

❌ **Game Visualization**: Not available (Ursina/Panda3D)
- Python 3.14 is not supported
- Need Python 3.8-3.12 for game visualization

## Solution: Install Python 3.12

Since you have Python 3.14 installed, you need to add Python 3.12 alongside it.

### Recommended: Install Python 3.12

1. **Download Python 3.12**
   - Go to: https://www.python.org/downloads/release/python-3127/
   - Download: "Windows installer (64-bit)"
   
2. **Install Python 3.12**
   - Run the installer
   - ✅ **Important**: Check "Add Python 3.12 to PATH"
   - ✅ Check "Install for all users" (optional)
   - Click "Install Now"

3. **Verify Installation**
   ```bash
   py -3.12 --version
   ```
   Should show: `Python 3.12.7`

4. **Create Virtual Environment with 3.12**
   ```bash
   py -3.12 -m venv venv312
   ```

5. **Activate Virtual Environment**
   ```bash
   venv312\Scripts\activate
   ```

6. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

7. **Test Everything**
   ```bash
   python quick_test.py
   ```
   Should show all tests passing!

8. **Run the Game**
   ```bash
   python game_crossy.py
   ```

### Alternative: Use Python Launcher

After installing Python 3.12, use the `py` launcher:

```bash
# Run with Python 3.12 specifically
py -3.12 game_crossy.py
py -3.12 train_crossy.py
py -3.12 test_crossy.py
```

## Quick Commands Summary

Once Python 3.12 is installed:

```bash
# Check what Python versions you have
py -0

# Use Python 3.12 for this project
py -3.12 -m pip install -r requirements.txt

# Run tests
py -3.12 quick_test.py

# Train model
py -3.12 train_crossy.py

# Test model
py -3.12 test_crossy.py
```

## Troubleshooting

### Issue: "No module named 'panda3d'"
- Make sure you're using Python 3.11 or 3.12
- Run: `py -3.12 -m pip install panda3d`

### Issue: "No module named 'ursina'"
- Run: `py -3.12 -m pip install ursina`

### Issue: Command prompt doesn't recognize 'py'
- Python wasn't installed correctly
- Reinstall Python with "Add to PATH" option
- Or use full path: `C:\Python312\python.exe`

## Why Python 3.12?

- **Stable**: Most stable recent release
- **Compatible**: Works with all dependencies
- **Ursina**: Supported by Panda3D (which Ursina needs)
- **PyTorch**: Fully supported
- **Community**: Widely used, good documentation

## Need Help?

See [INSTALLATION.md](INSTALLATION.md) for detailed troubleshooting.




