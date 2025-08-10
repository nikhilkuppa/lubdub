#!/usr/bin/env python3
"""
Quick setup script to create the basic project structure
Run this first, then copy the code files from the artifacts
"""

import os
from pathlib import Path

def create_project_structure():
    """Create the basic directory structure"""
    
    directories = [
        "src",
        "src/models", 
        "src/preprocessing",
        "src/features",
        "models",
        "data",
        "logs", 
        "tests",
        "frontend",
        "reference"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create __init__.py files
    init_files = [
        "src/__init__.py",
        "src/models/__init__.py", 
        "src/preprocessing/__init__.py",
        "src/features/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✅ Created: {init_file}")

def create_gitignore():
    """Create a .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Models and data
models/*.pkl
models/*.joblib
data/
logs/
*.wav
*.mp3
*.flac

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Docker
.dockerignore

# Jupyter
.ipynb_checkpoints/

# Environment variables
.env
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    print("✅ Created .gitignore")

def main():
    print("🚀 Setting up PCG Analyzer project structure...")
    
    create_project_structure()
    create_gitignore()
    
    print("\n📋 Next steps:")
    print("1. Copy the code from the artifacts into these files:")
    print("   - main.py")
    print("   - src/config.py") 
    print("   - src/models/pcg_classifier.py")
    print("   - src/models/s1s2_detector.py")
    print("   - src/preprocessing/audio_processor.py")
    print("   - src/features/feature_extractor.py")
    print("   - requirements.txt")
    print("   - Dockerfile")
    print("   - docker-compose.yml")
    print("   - nginx.conf")
    print("   - frontend/index.html")
    print("   - tests/test_api.py")
    print("   - deploy.sh")
    print("")
    print("2. Then run the migration:")
    print("   python migrate_from_old.py /Users/nikhilkuppa/Documents/Research/neurosonic/Nikhil-Kuppa-Assignment/")
    print("")
    print("3. Make deploy script executable:")
    print("   chmod +x deploy.sh")

if __name__ == "__main__":
    main()