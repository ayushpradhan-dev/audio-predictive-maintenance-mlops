# Audio-Based Predictive Maintenance for Industrial Machinery (MLOps)

This project implements a full end-to-end MLOps pipeline for classifying the health status of industrial machinery based on audio recordings. The goal is to build a system that can listen to a short audio clip of a fan and determine if it is in a "normal" or "abnormal" condition, demonstrating a realistic workflow from data analysis to model deployment.

## The Core Challenge: Generalization to Unseen Machines

A common pitfall in machine learning is creating a model that performs well in the lab but fails in the real world. This project directly confronts this challenge. The MIMII dataset contains audio from four different physical fans (`id_00`, `id_02`, `id_04`, `id_06`). An initial investigation revealed a significant **data distribution shift** between different fan units.

As shown below, the acoustic signature of an "abnormal" sound from one fan (`id_00`) is visually distinct from the signature of an "abnormal" sound from another (`id_06`). A model trained only on fans like `id_00` would fail to generalize to a new, unseen fan like `id_06`. This insight is central to the project's methodology.

<p align="center">
  <img src="docs/images/spectrogram_comparison.png" alt="Spectrogram comparison showing data distribution shift" width="800"/>
</p>

## Methodology & Key Findings

To build a robust model, this project followed a multi-stage, iterative approach.

### 1. Leave-One-Group-Out (LOGO) Cross-Validation
To get an honest estimate of real-world performance, a **Leave-One-Group-Out (LOGO)** cross-validation experiment was conducted. Four separate models were trained, each holding out a different fan unit for testing. This provided a reliable performance baseline and confirmed that generalization was a significant challenge.

### 2. Data Augmentation
The key to improving generalization was identified as a lack of variety in the training data. The final production model was trained with **data augmentation** (`SpecAugment`), which randomly masks time and frequency bands in the spectrograms. This forces the model to learn more robust, generalizable features instead of overfitting to the specific acoustic properties of the training fans.

### 3. Production Model Performance
A final production model was trained on all available fan data with data augmentation enabled. This model demonstrated excellent performance on its validation set, achieving an **AUC of 0.99** and a near-perfect precision, indicating an extremely low false-alarm rate. This is the model that will be used for deployment.

## API Service & Serverless Architecture

The trained production model is served via a **FastAPI** application wrapped in a **Mangum** adapter, allowing it to run as a serverless function on AWS Lambda. The service exposes a single `/predict` endpoint that accepts a `.wav` audio file and returns a JSON response containing the predicted label ("normal" or "abnormal") and a confidence score.

The image below shows a demo of the app, as it takes an uploaded wav file of a normally functioning fan of type `id_02` from the dataset, `(00000102.wav)`, as input and returns a prediction that correctly classifies the example as "normal".

<p align="center">
  <img src="docs/images/api_demo.png" alt="Screenshot of the FastAPI /docs page showing a successful prediction" width="800"/>
</p>

## Current Status: AWS Migration & Infrastructure

The project has pivoted to a cloud-native AWS Serverless architecture to optimize for cost and scalability. The application container is being prepared for deployment to AWS Lambda via Amazon ECR.

**Completed Milestones:**
*   **LOGO Cross-Validation:** Completed a full 4-fold cross-validation to establish a robust, unbiased performance baseline.
*   **Data Augmentation:** Implemented `SpecAugment` to solve the generalization challenge.
*   **Production Model:** Trained a final, high-performance model (AUC 0.99).
*   **Containerization:** Successfully containerized the application using **Docker**.
*   **Serverless Adaptation:** Integrated `Mangum` adapter to enable FastAPI execution within AWS Lambda.
*   **Infrastructure as Code (AWS):** Initiated Terraform code for AWS resources, starting with Amazon ECR and Lifecycle Policies.

**Next Steps: Cloud Deployment & Automation**
*   Provision AWS Lambda and API Gateway using **Terraform**.
*   Configure **GitHub Actions** to automate the build-and-push workflow to Amazon ECR.
*   Finalize the end-to-end CI/CD pipeline for automated model deployment.

## Tech Stack

-   **Data Analysis & Modeling:** PyTorch, Librosa, Scikit-learn, Pandas
-   **API Development:** FastAPI, Uvicorn, Mangum (AWS Adapter)
-   **Containerization:** Docker
-   **Infrastructure as Code:** Terraform
-   **Cloud Provider:** AWS (Lambda, ECR, API Gateway)
-   **CI/CD & Automation:** GitHub Actions

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

-   Git
-   Git LFS ([Install from here](https://git-lfs.github.com/))
-   Conda for environment management
-   Docker Desktop ([Install from here](https://www.docker.com/products/docker-desktop/))
-   AWS CLI (Configured with credentials)
-   Terraform

### 1. Clone the Repository
```bash
git clone https://github.com/ayushpradhan-dev/audio-predictive-maintenance-mlops.git
cd audio-predictive-maintenance-mlops
```

### 2. Set Up Git LFS
Initialize Git LFS to download the model files.
```bash
git lfs install
git lfs pull
```

### 3. Create the Conda Environment
This step is required for local development and to run the data processing/training notebooks. The project uses `conda-lock` for reproducibility.
```bash
# Create the environment from the lock file
conda create --name audio-mlops --file conda-lock.yml

# Activate the named environment
conda activate audio-mlops
```

### 4. Data Setup
This project uses the **MIMII Dataset (6dB Fan subset)**, which is not tracked by Git.

1.  **Download the data:** Download the `6_dB_fan.zip` file from the official dataset repository: [https://zenodo.org/records/3384388](https://zenodo.org/records/3384388).
2.  **Place and extract the data:** Create a `data/` directory in the project root and place the `6_dB_fan.zip` file inside. Extract the contents so your final structure is:
    ```
    audio-predictive-maintenance-mlops/
    └── data/
        └── fan/
            ├── id_00/, id_02/, etc.
    ```

## Usage

There are two ways to run this application: using the pre-built Docker image (recommended) or running the source code locally for development.

### 1. Run with Docker (Recommended)
This is the fastest and easiest way to get the application running. It downloads the pre-built, ready-to-run container image from Docker Hub.

**A) Pull the Image from Docker Hub:**
```bash
docker pull ayushpradhan24/audio-predictive-maintenance:latest
```

**B) Run the Docker Container:**
This command starts the container and maps your local port 8000 to the container's port 8000.
```bash
docker run -p 8000:8000 ayushpradhan24/audio-predictive-maintenance:latest
```
Once running, you can access the interactive API documentation at `http://127.0.0.1:8000/docs`.

---

### 2. Local Development & Experiments
This workflow is for users who want to modify the source code, run the data processing, or retrain the models.

**A) Process Raw Audio Data:**
This script converts all raw `.wav` files into spectrogram images. Run this once after setting up the data.
```bash
python src/audio_processor.py
```

**B) Run Experiments:**
The project contains notebooks to reproduce the experiments in the `notebooks/` directory. You will need to create the Conda environment first (see Getting Started).

**C) Run the API Server Locally:**
This command starts the FastAPI server directly on your local machine.
```bash
python src/main.py
```

## License
This project is licensed under the MIT License.

## Acknowledgements
This project uses the MIMII Dataset. Please cite the original authors if you use this data.
