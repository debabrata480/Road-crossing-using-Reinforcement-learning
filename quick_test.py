#!/usr/bin/env python
"""
Quick test to verify core dependencies work without Ursina.
This allows testing the DQN components even if Ursina is unavailable.
"""

import sys
import torch

def test_dependencies():
    """Test if core dependencies are available."""
    print("Testing dependencies...")
    
    try:
        import numpy as np
        print(f"[OK] numpy {np.__version__}")
    except ImportError as e:
        print(f"[FAIL] numpy not available: {e}")
        return False
    
    try:
        print(f"[OK] torch {torch.__version__}")
    except ImportError as e:
        print(f"[FAIL] torch not available: {e}")
        return False
    
    try:
        import matplotlib
        print(f"[OK] matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"[FAIL] matplotlib not available: {e}")
        return False
    
    try:
        from model import DQN
        print("[OK] model.py imports successfully")
    except Exception as e:
        print(f"[FAIL] model.py error: {e}")
        return False
    
    try:
        from agent import DQNAgent
        print("[OK] agent.py imports successfully")
    except Exception as e:
        print(f"[FAIL] agent.py error: {e}")
        return False
    
    print("\n" + "="*50)
    print("Ursina dependency test:")
    print("="*50)
    
    try:
        import ursina
        version = getattr(ursina, '__version__', 'installed')
        print(f"[OK] ursina {version}")
        print("[OK] Full game environment available!")
        return True
    except ImportError as e:
        print(f"[FAIL] ursina not available: {e}")
        print("\n[WARNING] Game visualization requires Python 3.8-3.12")
        print("   Your Python version:", sys.version)
        print("\n   You can still train models if Ursina works with manual install")
        print("   See INSTALLATION.md for details")
        return False

def test_model():
    """Test if the DQN model works."""
    print("\nTesting DQN model...")
    try:
        from model import DQN
        
        obs_shape = (12,)
        n_actions = 5
        model = DQN(obs_shape, n_actions)
        
        # Test forward pass
        test_input = torch.randn(1, 12)
        output = model(test_input)
        
        assert output.shape == (1, 5), f"Expected (1, 5), got {output.shape}"
        print(f"[OK] Model forward pass works: input {test_input.shape} -> output {output.shape}")
        return True
    except Exception as e:
        print(f"[FAIL] Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent():
    """Test if the DQN agent works."""
    print("\nTesting DQN agent...")
    try:
        from model import DQN
        from agent import DQNAgent
        
        obs_shape = (12,)
        n_actions = 5
        model = DQN(obs_shape, n_actions)
        agent = DQNAgent(model, lr=1e-3)
        
        # Test action selection
        test_state = torch.randn(12).numpy()
        action = agent.select_action(test_state, n_actions, eps=0.1)
        
        assert 0 <= action < n_actions, f"Action {action} out of range [0, {n_actions})"
        print(f"[OK] Agent action selection works: action={action}")
        return True
    except Exception as e:
        print(f"[FAIL] Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*50)
    print("Crossy DQN Project - Quick Test")
    print("="*50)
    print(f"Python version: {sys.version}")
    print()
    
    deps_ok = test_dependencies()
    model_ok = test_model()
    agent_ok = test_agent()
    
    print("\n" + "="*50)
    if deps_ok and model_ok and agent_ok:
        print("[SUCCESS] ALL TESTS PASSED!")
        print("\nYou can now run:")
        print("  - python game_crossy.py (to test game environment)")
        print("  - python train_crossy.py (to train the agent)")
        print("  - python test_crossy.py (to test trained model)")
    else:
        print("[WARNING] SOME TESTS FAILED")
        print("\nPlease check the errors above and:")
        print("  1. Make sure Python version is 3.8-3.12 for Ursina")
        print("  2. Run: pip install -r requirements.txt")
        print("  3. See INSTALLATION.md for detailed instructions")
    print("="*50)

