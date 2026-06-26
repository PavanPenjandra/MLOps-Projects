# Virtual Environment Setup Guide

## Problem
When running Python scripts, the system uses Anaconda's Python installation instead of the project's virtual environment, causing import errors.

## Solution

### Option 1: Activate Virtual Environment (RECOMMENDED)

**On Windows PowerShell (EASIEST FIX - No Admin Needed):**
```powershell
# Navigate to project directory
cd "c:\Users\pavan\OneDrive\Documents\MLOps-Projects\NLP-Sentiment-Analysis-MLOps"

# Activate with bypass flag (single line, no admin needed)
powershell -ExecutionPolicy Bypass -Command "& .\venv\Scripts\Activate.ps1"

# Or use this shorter version:
.\venv\Scripts\Activate.ps1 -ExecutionPolicy Bypass

# You should see (venv) at the start of the prompt
# Now run your commands
python scripts/train_pipeline.py --config config.yaml
mlflow ui
```

**Alternative - One-Time Setup (Persistent, Requires Admin):**
```powershell
# Right-click PowerShell and select "Run as Administrator"
# Then run this ONCE:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Now you can activate normally without the bypass flag:
.\venv\Scripts\Activate.ps1
```

**Recommended - Use Command Prompt (cmd.exe) Instead (No Issues):**
```cmd
cd c:\Users\pavan\OneDrive\Documents\MLOps-Projects\NLP-Sentiment-Analysis-MLOps
venv\Scripts\activate.bat
python scripts/train_pipeline.py --config config.yaml
```

### Option 2: Use Full Path (If Activation Doesn't Work)

Run Python directly with the full venv path:
```powershell
c:\Users\pavan\OneDrive\Documents\MLOps-Projects\NLP-Sentiment-Analysis-MLOps\venv\Scripts\python.exe scripts/train_pipeline.py --config config.yaml
```

### Option 3: Deactivate Anaconda (NUCLEAR OPTION)

If Anaconda keeps interfering:
```powershell
# Deactivate Anaconda base environment
conda deactivate

# Then activate the venv
.\venv\Scripts\Activate.ps1
```

## Verification

After activating the venv, verify it's working:
```powershell
python --version
python -c "import sys; print(sys.executable)"
```

You should see:
- Python 3.9.12
- Path containing `venv\Scripts\python.exe` (NOT anaconda3)

## Why This Happens

1. **Anaconda modifies PATH** - It adds itself to Windows PATH, making its Python default
2. **Virtual environment modifies PATH** - When activated, it temporarily changes PATH to prioritize venv
3. **Without activation** - Windows finds Anaconda's Python first

## Recommended Workflow

```powershell
# 1. Open PowerShell and navigate to project
cd "c:\Users\pavan\OneDrive\Documents\MLOps-Projects\NLP-Sentiment-Analysis-MLOps"

# 2. ALWAYS activate venv first
.\venv\Scripts\Activate.ps1

# 3. Now run any commands
python scripts/train_pipeline.py --config config.yaml
python -m uvicorn src.app.main:app --reload
mlflow ui

# 4. When done, deactivate
deactivate
```

## For IDE Integration

**VS Code**: 
1. Open `.vscode/settings.json` (or create it)
2. Add: `"python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe"`
3. Restart VS Code

**PyCharm**:
1. File → Settings → Project → Python Interpreter
2. Click gear icon → Add → Existing Environment
3. Point to: `c:\Users\pavan\OneDrive\Documents\MLOps-Projects\NLP-Sentiment-Analysis-MLOps\venv\Scripts\python.exe`
4. Click OK

## Quick Fix for Current Terminal

**BEST SOLUTION - Copy and paste ONE of these:**

Option A (PowerShell - No Admin):
```powershell
cd "c:\Users\pavan\OneDrive\Documents\MLOps-Projects\NLP-Sentiment-Analysis-MLOps"; powershell -ExecutionPolicy Bypass -Command "& .\venv\Scripts\Activate.ps1"
```

Option B (Command Prompt - No Admin, EASIEST):
```cmd
cd c:\Users\pavan\OneDrive\Documents\MLOps-Projects\NLP-Sentiment-Analysis-MLOps && venv\Scripts\activate.bat
```

Option C (PowerShell - One-time admin setup, then works forever):
```powershell
# Run PowerShell as Administrator first, then:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# After this, you can use normal activation in future
```

After activation, your prompt should show `(venv)` and all imports will work correctly!
