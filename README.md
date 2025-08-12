# Audio-Based Predictive Maintenance for Industrial Machinery

This project is an end-to-end MLOps implementation for classifying the health status of industrial machinery based on audio recordings. The goal is to build a system that can listen to a short audio clip of a fan and determine if it is "Healthy" or "Abnormal" (indicating a fault), demonstrating a realistic workflow from data analysis to model deployment.

## The Core Challenge: Generalization to Unseen Machines

A common pitfall in machine learning is creating a model that performs well in the lab but fails in the real world. This project directly confronts this challenge. The MIMII dataset contains audio from multiple different physical fans, there are 4 types of fans (id_00, id_02, id_04, id_06) and each type has numerous 10 second recordings. An initial investigation revealed a significant **data distribution shift** between different fan units.

As shown below, the acoustic signature of an "abnormal" sound from one fan (`id_00`) is visually distinct from the signature of an "abnormal" sound from another (`id_06`). This real-world scenario proves that simply training a model on available data is not enough. The engineering challenge is to build a system that is robust to these variations. The next steps are explicitly designed to solve this problem.

<p align="center">
  <img src="docs/images/spectrogram_comparison.png" alt="Spectrogram comparison showing data distribution shift" width="800"/>
</p>

A model trained only on fans like `id_00` would fail to generalize to a new, unseen fan like `id_06`. This insight is central to the project's methodology.

## Methodology: Leave-One-Group-Out (LOGO) Validation

To build a robust model, we cannot mix all the data and split it randomly. This would give a deceptively high score but would not reflect real-world performance.

Instead, this project uses the **Leave-One-Group-Out (LOGO)** cross-validation strategy:
1.  **Hold Out:** An entire fan unit (e.g., `id_06`) is completely excluded from training and validation.
2.  **Train:** The model is trained on the remaining fan units (e.g., `id_00`, `id_02`, `id_04`).
3.  **Test:** The model's true performance is measured on the held-out fan it has never encountered.

This approach provides an honest and reliable estimate of the model's ability to generalize to new, unseen machinery.

## Current Status: Work in Progress

This project is under active development. The current focus is on building a robust, generalizable model that can handle real-world challenges like data distribution shifts between different machines.

**Completed Milestones:**
*   Established a baseline model using a ResNet-50 architecture with Attention Pooling.
*   Implemented a Leave-One-Group-Out (LOGO) validation strategy, training on three fan types and testing on a completely unseen fan type (`id_06`).
*   Discovered and confirmed a significant data distribution shift between the training and test sets.
*   Successfully integrated Git LFS for versioning large model artifacts.

**Next Steps:**
1.  **Implement Data Augmentation:** Introduce audio-specific augmentations (e.g., Time and Frequency Masking) during training to improve the model's ability to generalize.
2.  **Complete LOGO Cross-Validation:** Train and evaluate models for all four fan combinations to get a highly reliable estimate of real-world performance.
3.  **Deployment Pipeline:**
    *   Develop a FastAPI service to serve the trained model.
    *   Containerize the service using Docker.
    *   Define cloud infrastructure on Azure using Terraform.
    *   Automate the entire build and deploy process with GitHub Actions.

## Tech Stack

-   **Data Analysis & Modeling:** PyTorch, Librosa, Scikit-learn, Pandas, Matplotlib
-   **API Development:** FastAPI
-   **Containerization:** Docker
-   **Infrastructure as Code:** Terraform
-   **Cloud Deployment:** Microsoft Azure (ACR, ACI)
-   **CI/CD & Automation:** GitHub Actions

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

-   Git
-   Git LFS (see instructions [here](https://git-lfs.github.com/))
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
This project uses conda-lock to create a byte-for-byte identical environment from the conda-lock.yml file, ensuring reproducibility.
```bash
# Create the environment from the lock file
conda create --name audio-mlops --file conda-lock.yml

# Activate the named environment
conda activate audio-mlops
```
This creates a standard, named conda environment called audio-mlops.

### 4. Data Setup
This project uses the **MIMII Dataset (6dB Fan subset)**, which is not tracked by Git.

1.  **Download the data:** Download the `6_dB_fan.zip` file from the official dataset repository: [https://zenodo.org/records/3384388](https://zenodo.org/records/3384388).
2.  **Place and extract the data:**
    *   Create a directory `data/` in the project root.
    *   Place the `6_dB_fan.zip` file inside `data/`.
    *   Extract the contents. Your final directory structure must be:
    ```
    audio-predictive-maintenance-mlops/
    └── data/
        └── fan/
            ├── id_00/
            ├── id_02/
            ├── id_04/
            └── id_06/
    ```

## Usage

1.  **Process the Data:**
    *   To create the processed training images (`id_00`, `id_02`, `id_04`): 
    ```bash
    python src/audio_processor.py
    ```
    *   To create the processed test images (`id_06`):
    ```bash
    python src/create_test_data.py
    ```

2.  **Train the Model:**
    Open the `notebooks/03-model-training-and-evaluation.ipynb` notebook and run the cells. This will train the model and save the best version to `models/`.

3.  **Evaluate the Model:**
    The final cells in the training notebook load the best model and evaluate it against the hold-out test set, generating the confusion matrix and ROC curve.

## License
This project is licensed under the MIT License.

## Acknowledgements
This project uses the MIMII Dataset. Please cite the original authors if you use this data:
> Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection," in Proceedings of the 4th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 2019.