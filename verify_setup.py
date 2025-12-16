"""
Setup verification script for Deep Unfolded BM3D
Run this after installing dependencies to verify everything works.
"""
import sys

def check_imports():
    """Check if all required packages are installed"""
    print("Checking package imports...")
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'yaml': 'PyYAML',
        'skimage': 'scikit-image',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm',
        'PIL': 'Pillow'
    }
    
    missing = []
    for pkg, name in packages.items():
        try:
            __import__(pkg)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT FOUND")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All packages installed successfully!")
    return True


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")


def check_config():
    """Check config file"""
    print("\nChecking configuration...")
    try:
        import yaml
        from pathlib import Path
        
        config_path = Path('config.yaml')
        if not config_path.exists():
            print("✗ config.yaml not found")
            return False
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        print("✓ config.yaml loaded successfully")
        
        # Check required sections
        required = ['model', 'training', 'data', 'paths', 'dataset']
        for section in required:
            if section not in config:
                print(f"✗ Missing config section: {section}")
                return False
        
        print("✓ All config sections present")
        return True
        
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False


def check_modules():
    """Check project modules"""
    print("\nChecking project modules...")
    modules = ['models', 'data', 'utils', 'train', 'evaluate']
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}.py")
        except Exception as e:
            print(f"✗ {module}.py - Error: {e}")
            return False
    
    print("✓ All project modules loaded successfully")
    return True


def main():
    """Run all checks"""
    print("="*60)
    print("Deep Unfolded BM3D - Setup Verification")
    print("="*60)
    
    checks = [
        ("Package Installation", check_imports),
        ("CUDA Support", check_cuda),
        ("Configuration", check_config),
        ("Project Modules", check_modules)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append(result if result is not None else True)
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    if all(results):
        print("✅ Setup verification PASSED - Ready to use!")
        print("\nNext steps:")
        print("  1. Prepare your data in data/train, data/val, data/test")
        print("  2. Run: python train.py --config config.yaml")
        print("  3. Run: python evaluate.py --config config.yaml --model outputs/checkpoints/best_model.pth")
    else:
        print("❌ Setup verification FAILED")
        print("\nPlease fix the issues above and run this script again.")
        sys.exit(1)
    print("="*60)


if __name__ == "__main__":
    main()
