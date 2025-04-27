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

1. Run one of the following lines to start predicting using either model A or model B
   ```shell
   python script/predict.py <folder_path> <output_csv_filename>
   ```
   ```shell
   python script/predict_combined.py <folder_path> <output_csv_filename>
   ```
   > 💡 Example: python script/predict.py ~/Desktop/data_folder ~/Desktop/results.csv

2. Both of these scripts will run the model on your input test images and save the results in csv file where you specified
<div align="center">

## Running the Predictor Script - from Docker
</div>

<blockquote>
⚠️ Make sure you have <a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html">nvidia-container-toolkit</a> installed in your local environment to be able to use GPU inside the Docker container.
</blockquote>

1. First load docker image to your local environment
   ```shell
   docker load -i melanoma_predictor.tar
   ```

2. Run docker container
   ```shell
   docker run --gpus all --name melanoma_predictor -it melanoma_predictor bash
   ```
  
3. From another terminal on local computer copy your folder with test images into docker container


   **Linux / macOS / WSL**:
   ```shell
   docker cp 'your/path' melanoma_predictor:/app/data
   ```
   **PowerShell**:
   ```shell
   docker cp "C:\your\path" melanoma_predictor:/app/data
   ```
   **CMD**:
   ```shell
   docker cp C:\your\path melanoma_predictor:/app/data
   ```

4. Run one of the following lines to start predicting using either model A or model B
   ```shell
   python script/predict.py <folder_path> <output_csv_filename>
   ```
   ```shell
   python script/predict_combined.py <folder_path> <output_csv_filename>
   ```
   > 💡 'folder_path' is now in docker container where you defined in step 3
   
   > 💡 Example: python script/predict.py ./data/data_folder ./results.csv
   
   > 💡 Run script with -h flag for help with lowering batch size if you run out of memory


5. When the script finishes copy the results file to your local computer
   ```shell
   docker cp melanoma_predictor:/app/<output_csv_filename> .
   ```
   



<div align="center">

## 🔄

## Deactivating Your Environment

</div>

When you're done with your work:
```shell
deactivate
```
