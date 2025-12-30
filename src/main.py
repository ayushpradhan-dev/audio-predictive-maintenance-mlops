# src/main.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import librosa
import numpy as np
import pandas as pd
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from torchvision import transforms
import torchaudio.transforms as T
import uvicorn
from pathlib import Path

from mangum import Mangum

# Setup
# Create the FastAPI app
app = FastAPI(title="Audio-Based Predictive Maintenance API")

# Define device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define the model path and load the production model,
# assume script is run from project root.
MODEL_PATH = "models/production_model.pth"

# Import model's class from src/model.py
from model import SpectrogramResNet

# Load model and set to evaluation mode
model = SpectrogramResNet(num_classes=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Define the image transforms
IMAGE_SIZE = (224, 224)

transforms_pipeline = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class names
class_names = ['abnormal', 'normal']

# Pydantic models for input/output
# Defines structure of the JSON response
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

# Prediction Helper Function
def predict_audio(audio_bytes: bytes) -> tuple[str, float]:
    """
    Takes raw audio bytes, preprocesses them, and returns a prediction with the correct label and confidence.
    """
    # Load audio from in-memory bytes
    waveform, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # Convert to Mel Spectrogram
    spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
    db_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Convert spectrogram (numpy array) to a PIL Image for the transform pipeline
    norm_spectrogram = (db_spectrogram - db_spectrogram.min()) / (db_spectrogram.max() - db_spectrogram.min())
    image = Image.fromarray((norm_spectrogram * 255).astype(np.uint8)).convert("RGB")

    # Apply the same transforms as the validation set
    image_tensor = transforms_pipeline(image).unsqueeze(0).to(DEVICE)

    # Make a prediction
    with torch.no_grad():
        logits = model(image_tensor)
        # Probabilities represents the model's confidence in the "normal" class (class 1)
        probabilities = torch.sigmoid(logits.squeeze())
        
        # If the probability of being 'normal' is > 0.5, the prediction is 'normal'.
        if probabilities.item() > 0.5:
            prediction_label = "normal"
            confidence_score = probabilities.item()
        # Otherwise, the prediction is 'abnormal'.
        else:
            prediction_label = "abnormal"
            # The confidence is how sure it is of being 'abnormal' (1 - P(normal))
            confidence_score = 1 - probabilities.item()

    return prediction_label, confidence_score

# API Endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accepts a .wav file, performs inference, and returns the prediction.
    """
    # Read the audio file from the upload
    audio_bytes = await file.read()

    # Get the prediction from helper function
    prediction, confidence = predict_audio(audio_bytes)

    return {"prediction": prediction, "confidence": confidence}


# Execution block for running the server
# Run the app directly using `python src/main.py`
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

handler = Mangum(app)