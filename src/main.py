# src/main.py
import os
from pathlib import Path

# Fix for OMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Fix for Librosa/Numba on Lambda
# AWS Lambda filesystem is read-only except for /tmp. 
# tell Numba to use /tmp for caching
os.environ["NUMBA_CACHE_DIR"] = "/tmp"

# redirect pytorch cached pretrained weights to /tmp
os.environ['TORCH_HOME'] = '/tmp/torch'

import torch
import librosa
import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from torchvision import transforms
import uvicorn

from mangum import Mangum

# SETUP
app = FastAPI(title="Audio-Based Predictive Maintenance API")

# Define device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL PATHING
# absolute paths to ensure the model is found regardless of CWD.
# __file__ is src/main.py -> parent is src/ -> parent is project root /app
BASE_DIR = Path(__file__).resolve().parent.parent 
MODEL_PATH = BASE_DIR / "models" / "production_model.pth"

# Import model's class
from src.model import SpectrogramResNet

# Load model
print(f"Loading model from: {MODEL_PATH}")
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

# Pydantic models
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

# Helper Function
def predict_audio(audio_bytes: bytes) -> tuple[str, float]:
    """
    Takes raw audio bytes, preprocesses them, and returns a prediction.
    """
    # Load audio from in-memory bytes (Perfect for Lambda)
    waveform, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # Convert to Mel Spectrogram
    spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
    db_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Convert to PIL Image
    norm_spectrogram = (db_spectrogram - db_spectrogram.min()) / (db_spectrogram.max() - db_spectrogram.min())
    image = Image.fromarray((norm_spectrogram * 255).astype(np.uint8)).convert("RGB")

    # Apply transforms
    image_tensor = transforms_pipeline(image).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits.squeeze())
        
        if probabilities.item() > 0.5:
            prediction_label = "normal"
            confidence_score = probabilities.item()
        else:
            prediction_label = "abnormal"
            confidence_score = 1 - probabilities.item()

    return prediction_label, confidence_score

# API Endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    prediction, confidence = predict_audio(audio_bytes)
    return {"prediction": prediction, "confidence": confidence}

# Local Execution
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# AWS Lambda Handler
handler = Mangum(app)