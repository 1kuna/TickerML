@echo off
setlocal

echo.
echo Setting up Crypto Time-Series Transformer Pipeline...
echo.

echo [INFO] Detected platform: Windows PC

echo [INFO] Looking for Python installation...

REM Try different Python commands in order of preference
set PYTHON_CMD=
echo [DEBUG] Testing Python installations...

"C:\Users\zcane\AppData\Local\Programs\Python\Python311\python.exe" --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD="C:\Users\zcane\AppData\Local\Programs\Python\Python311\python.exe"
    echo [SUCCESS] Found Python: C:\Users\zcane\AppData\Local\Programs\Python\Python311\python.exe
    goto found_python
)

python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    echo [SUCCESS] Found Python: python
    goto found_python
)

python3 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python3
    echo [SUCCESS] Found Python: python3
    goto found_python
)

py --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py
    echo [SUCCESS] Found Python: py
    goto found_python
)

echo [ERROR] Python not found. Please install Python 3.8+ and add it to PATH
pause
exit /b 1

:found_python
for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python version: %PYTHON_VERSION%

echo [INFO] Creating virtual environment...
if not exist "venv" (
    echo [INFO] Running: %PYTHON_CMD% -m venv venv
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created successfully
) else (
    echo [WARNING] Virtual environment already exists
)

echo [INFO] Activating virtual environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [SUCCESS] Virtual environment activated
) else (
    echo [ERROR] Virtual environment activation script not found
    pause
    exit /b 1
)

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip, continuing anyway...
)

REM Check if requirements files exist
if not exist "pc\requirements.txt" (
    echo [ERROR] pc\requirements.txt not found
    pause
    exit /b 1
)

if not exist "raspberry_pi\requirements.txt" (
    echo [ERROR] raspberry_pi\requirements.txt not found
    pause
    exit /b 1
)

echo [INFO] Installing PC requirements...
pip install -r pc\requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install PC requirements
    pause
    exit /b 1
)

echo [INFO] Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [INFO] NVIDIA GPU detected, installing CUDA-enabled PyTorch...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    echo [WARNING] No NVIDIA GPU detected, using CPU-only PyTorch
)

echo [INFO] Installing Raspberry Pi requirements for testing...
pip install -r raspberry_pi\requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install Raspberry Pi requirements
    pause
    exit /b 1
)

echo [INFO] Creating directory structure...
if not exist "data\dumps" mkdir data\dumps
if not exist "data\db" mkdir data\db
if not exist "data\features" mkdir data\features
if not exist "models\checkpoints" mkdir models\checkpoints
if not exist "models\onnx" mkdir models\onnx
if not exist "logs" mkdir logs
if not exist "raspberry_pi\templates" mkdir raspberry_pi\templates
if not exist "notebooks" mkdir notebooks

echo [SUCCESS] Directory structure created

echo [INFO] Setting up logging...
if not exist "logs\crypto_pipeline.log" (
    type nul > logs\crypto_pipeline.log
)

echo [INFO] Setting up Windows specific configurations...

echo [INFO] Setting up Ollama for sentiment analysis...
echo [WARNING] Please ensure Ollama is installed on your system:
echo [WARNING] Visit https://ollama.ai for installation instructions
echo [WARNING] After installation, run: ollama pull gemma3:4b

echo [INFO] Creating analysis notebook...
(
echo {
echo  "cells": [
echo   {
echo    "cell_type": "markdown",
echo    "metadata": {},
echo    "source": [
echo     "# Crypto Time-Series Analysis\n",
echo     "\n",
echo     "This notebook provides tools for analyzing the crypto prediction pipeline."
echo    ]
echo   },
echo   {
echo    "cell_type": "code",
echo    "execution_count": null,
echo    "metadata": {},
echo    "source": [
echo     "import pandas as pd\n",
echo     "import numpy as np\n",
echo     "import matplotlib.pyplot as plt\n",
echo     "import seaborn as sns\n",
echo     "from pathlib import Path\n",
echo     "\n",
echo     "# Load data\n",
echo     "data_path = Path('data/features'^)\n",
echo     "print(f\"Available feature files: {list(data_path.glob('*.csv'^)}\"^)"
echo    ]
echo   }
echo  ],
echo  "metadata": {
echo   "kernelspec": {
echo    "display_name": "Python 3",
echo    "language": "python",
echo    "name": "python3"
echo   }
echo  },
echo  "nbformat": 4,
echo  "nbformat_minor": 4
echo }
) > notebooks\analysis.ipynb

echo [INFO] Testing installations...

echo [INFO] Testing basic package imports...
python -c "import pandas as pd; import numpy as np; import requests; print('Basic packages imported successfully')"
if errorlevel 1 (
    echo [ERROR] Failed to import basic packages
    pause
    exit /b 1
)

echo [INFO] Testing ML package imports...
python -c "import torch; import onnx; import onnxruntime; print('ML packages imported successfully'); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
if errorlevel 1 (
    echo [ERROR] Failed to import ML packages
    pause
    exit /b 1
)

echo [INFO] Creating sample configuration...
if not exist "config\config.yaml" (
    if exist "config\config.yaml.sample" (
        copy "config\config.yaml.sample" "config\config.yaml" >nul
        echo [INFO] config\config.yaml created from sample. Please edit it with your API keys.
    ) else (
        echo [WARNING] config\config.yaml.sample not found. Please create config\config.yaml manually.
    )
) else (
    echo [WARNING] config\config.yaml already exists. Please ensure it's up to date with config\config.yaml.sample.
)

echo [SUCCESS] Setup completed successfully!

echo.
echo Next Steps:
echo 1. Update config\config.yaml with your API keys
echo 2. For NewsAPI: Get a free key from https://newsapi.org/
echo 3. Run feature engineering: python pc\features.py
echo 4. Train model: python pc\train.py
echo 5. Export to ONNX: python pc\export_quantize.py
echo 6. Test Raspberry Pi code: python raspberry_pi\harvest.py

echo.
echo Useful Commands:
echo - Activate environment: venv\Scripts\activate.bat
echo - View logs: type logs\crypto_pipeline.log
echo - Check database: sqlite3 data\db\crypto_data.db
echo - Start Jupyter: jupyter notebook
echo - Test Raspberry Pi dashboard: python raspberry_pi\dashboard.py

echo.
echo [SUCCESS] Happy trading!
echo.
pause 