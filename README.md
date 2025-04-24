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
 
> [Running the Predictor Script - locally](#running-the-predictor-script---locally)

> [Running the Predictor Script - from Docker](#running-the-predictor-script---from-Docker)

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

## Running the Predictor Script - locally

</div>

1. To upload test data, run this code, drag and drop your folder with test images inside terminal and press enter
   ```shell
   python script/save_images.py
   ```
2. Run one of the following lines to start predicting using either model A or model B
   ```shell
   python script/predict.py
   ```
   ```shell
   python script/predict_combined.py
   ```
3. Both of these scripts will run the model on your input test images and save the results in 'Test_predictions.csv'

<div align="center">

## Running the Predictor Script - from Docker
</div>

1. First load docker image to your local environment
   ```shell
   docker load -i melanoma_predict.tar
   ```
2. From another terminal copy your folder with test images into docker container


   **Linux / macOS / WSL**:
   ```shell
   docker cp 'your/path/.' melanoma_predict:/app/data/uploaded_images/
   ```
   **PowerShell**:
   ```shell
   docker cp "C:\your\path\." melanoma_predict:/app/data/uploaded_images/
   ```
   **CMD**:
   ```shell
   docker cp C:\your\path\. melanoma_predict:/app/data/uploaded_images/
   ```
   > 💡 Make sure your path ends with .

3. Run one of the following lines to start predicting using either model A or model B
   ```shell
   python script/predict.py
   ```
   ```shell
   python script/predict_combined.py
   ```

4. When the script finishes copy the results file to your local computer
   ```shell
   docker cp melanoma_predict:/app/data/Test_predictions.csv .
   ```
<div align="center">

## 🔄

## Deactivating Your Environment

</div>

When you're done with your work:
```shell
deactivate
```