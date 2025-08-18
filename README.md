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

## API Service

The trained production model is served via a **FastAPI** application. The service exposes a single `/predict` endpoint that accepts a `.wav` audio file and returns a JSON response containing the predicted label ("normal" or "abnormal") and a confidence score.

The image below shows a demo of the app, as it takes an uploaded wav file of a normally functioning fan of type `id_02` from the dataset, `(00000102.wav)`,  as input and returns a prediction that correctly classifies the example as "normal".

<p align="center">
  <img src="docs/images/api_demo.png" alt="Screenshot of the FastAPI /docs page showing a successful prediction" width="800"/>
</p>

## Current Status: API Development Complete

The machine learning and local API development phases of this project are complete. The next steps focus on containerization and cloud deployment.

**Completed Milestones:**
*   **LOGO Cross-Validation:** Completed a full 4-fold cross-validation to establish a robust, unbiased performance baseline.
*   **Data Augmentation:** Implemented `SpecAugment` to solve the generalization challenge.
*   **Production Model:** Trained a final, high-performance model (AUC 0.99) for deployment.
*   **API Development:** Developed a robust FastAPI service to serve the production model.
*   **Artifact Versioning:** Successfully integrated Git LFS for versioning all large model and results files.

**Next Steps: Deployment Pipeline**
*   Containerize the service using **Docker**.
*   Define cloud infrastructure on **Azure** using **Terraform**.
*   Automate the entire build and deployment process with **GitHub Actions**.

## Tech Stack

-   **Data Analysis & Modeling:** PyTorch, Librosa, Scikit-learn, Pandas
-   **API Development:** FastAPI, Uvicorn
-   **Containerization:** Docker (Planned)
-   **Infrastructure as Code:** Terraform
-   **Cloud Deployment:** Microsoft Azure
-   **CI/CD & Automation:** GitHub Actions

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

-   Git
-   Git LFS ([Install from here](https://git-lfs.github.com/))
-   Conda for environment management

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
This project uses `conda-lock` to create a byte-for-byte identical environment from the `conda-lock.yml` file, ensuring reproducibility.
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

### 1. Process Raw Audio Data
This script converts all raw `.wav` files into spectrogram images, creating a unified `data/processed/` directory. Run this once.
```bash
python src/audio_processor.py
```

### 2. Train Models (Optional)
The project contains notebooks to reproduce the experiments:
*   `notebooks/03-model-training-and-evaluation.ipynb`: Contains the full Leave-One-Group-Out (LOGO) cross-validation experiment. Run this to reproduce the baseline performance analysis.
*   `notebooks/04-final-model-training.ipynb`: Trains the final, high-performance production model using data augmentation on the entire dataset. This produces the `production_model.pth` artifact used for deployment.

### 3. Run the API Server Locally
This command starts the FastAPI server using the pre-trained production model.
```bash
python src/main.py
```
Once running, you can access the interactive API documentation at `http://127.0.0.1:8000/docs`.

## License
This project is licensed under the MIT License.

## Acknowledgements
This project uses the MIMII Dataset. Please cite the original authors if you use this data:
> Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection," in Proceedings of the 4th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 2019.