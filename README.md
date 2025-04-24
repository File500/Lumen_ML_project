<div align="center">

# 🔍 Lumen ML Project
A machine learning project for image prediction and analysis.
</div>

---
<div align="center">

## 📋 Table of Contents

</div>

> [Environment Setup](#environment-setup)
>>[Automatic Setup](#automatic-setup)
> 
>>[Manual Setup](#manual-setup)

> [Activating Your Environment](#activating-your-environment)
 
> [Running the Predictor Script](#running-the-predictor-script)

> [Deactivating Your Environment](#deactivating-your-environment)

---
<div align="center">

## 🛠️

## Environment Setup

</div>

### Automatic Setup

**Linux**
```shell
bash setup.sh
```

**Powershell**
```shell
.\setup.sh
```

> ⚠️ After creation, don't forget to activate your environment (gambit_env)

### Manual Setup

**Create virtual environment:**
```shell
python -m venv gambit_env
```

**When inside virtual environment:**
```bash
pip install -r requirements.txt
```

**Install PyTorch with CUDA support:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> ⚠️ After creation, don't forget to activate your environment (gambit_env)

<div align="center">

## 🚀

## Activating Your Environment

</div>

**Powershell activation**
```shell
gambit_env\Scripts\Activate.ps1
```

**CMD activation**
```shell
gambit_env\Scripts\activate
```

**Linux terminal activation**
```shell
source gambit_env/bin/activate
```

<div align="center">

## 📊

## Running the Predictor Script

</div>

1. Inside the script folder, create a folder with your desired name where IMAGES and IMAGE METADATA csv file will be stored
2. Inside the script folder, there is a Python script named `predict.py` that you will run:
   ```shell
   python predict.py .\test_folder test.csv
   ```
   > 💡 Replace `test_folder` and `test.csv` with your desired names for input folder and output CSV file
3. Results will be stored in the specified CSV file (you do not need to create it)

<div align="center">

## 🔄

## Deactivating Your Environment

</div>

When you're done with your work:
```shell
deactivate
```