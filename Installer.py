import subprocess
import sys
from pathlib import Path

def run() -> int:
    """Install project dependencies from requirements.txt."""

    # Get project root directory (where this script is located)
    project_root = Path(__file__).parent
    requirements_file = project_root / "requirements.txt"

    # Check if requirements.txt exists
    if not requirements_file.exists():
        print(f"Error: requirements.txt not found at {requirements_file}")
        return 1

    print(f"Installing dependencies from {requirements_file}...")

    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

    try:
        # Choose installation method based on environment
        if in_venv:
            # In virtual environment - install directly
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            print("Virtual environment detected - installing directly")
        else:
            # Not in virtual environment - try --user first
            cmd = [sys.executable, "-m", "pip", "install", "--user", "-r", str(requirements_file)]
            print("Installing with --user flag")

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ Dependencies installed successfully!")
            return 0
        else:
            # Check if it's the externally-managed-environment error
            if "externally-managed-environment" in result.stderr:
                print("✗ Installation failed: Externally managed environment detected")
                print("\n" + "="*60)
                print("SOLUTION OPTIONS:")
                print("="*60)
                print("1. Create and use a virtual environment:")
                print("   python3 -m venv venv")
                print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
                print("   python3 run install")
                print()
                print("2. Use the existing venv/ directory (if present):")
                venv_path = project_root / "venv"
                if venv_path.exists():
                    print(f"   source {venv_path}/bin/activate")
                    print("   python3 run install")
                else:
                    print("   No venv/ directory found in project root")
                print()
                print("3. Install system-wide (not recommended):")
                print("   python3 -m pip install --break-system-packages -r requirements.txt")
                print("="*60)
                return 1
            else:
                print(f"✗ Installation failed with exit code {result.returncode}")
                if result.stderr:
                    print("Error output:")
                    print(result.stderr)
                return result.returncode

    except FileNotFoundError:
        print("Error: pip not found. Please ensure Python and pip are installed.")
        return 1
    except Exception as e:
        print(f"Error during installation: {e}")
        return 1