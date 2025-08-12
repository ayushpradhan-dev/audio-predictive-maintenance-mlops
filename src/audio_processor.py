# src/audio_processor.py
import librosa
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import pandas as pd


def save_spectrogram(spectrogram_data, file_path):
    """Saves a spectrogram numpy array as a PNG image file."""
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    librosa.display.specshow(spectrogram_data, ax=ax)

    fig.savefig(file_path)
    plt.close(fig)


def process_and_save_spectrograms(fan_df: pd.DataFrame, output_dir: str):
    """
    Processes audio files from a DataFrame to create and save their spectrogram images.
    
    This function iterates through the DataFrame, performs the following for each file:
    1. Excludes files belonging to the hold-out test set ('id_06').
    2. Loads the audio .wav file.
    3. Converts it to a Mel spectrogram.
    4. Converts the power spectrogram to the decibel (dB) scale.
    5. Saves the result as a .png image in the appropriate 'normal' or 'abnormal' subfolder.

    Args:
        fan_df: DataFrame containing the file paths and labels (from data_loader).
        output_dir: The root directory to save the processed spectrogram images.
    """

    # Exclude the hold-out test set ('id_06')
    processing_df = fan_df[fan_df['fan_id'] != 'id_06'].copy()

    # Convert the output directory string to a Path object for better path handling
    output_path = Path(output_dir)

    print(f"Starting processing. A total of {len(processing_df)} files will be processed.")

    # Loop through the filitered DataFrame with progress bar (tqdm)
    for index, row in tqdm(processing_df.iterrows(), total=len(processing_df), desc="Processing audio files"):

        # Define output path and create the directory if it doesn't exist
        label = row['label']
        # e.g., 'data/processed/fan/abnormal'
        target_dir = output_path / label
        target_dir.mkdir(parents=True, exist_ok=True)

        # Get the original filename and create a new filename for the .png
        original_wav_path = Path(row['file_path'])

        # e.g., 'abnormal_id_00_00000000.png'
        new_filename = original_wav_path.stem + '.png'
        final_file_path = target_dir / new_filename

        # Load the audio file
        y, sr = librosa.load(original_wav_path)

        # Create Mel spectrogram
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

        # Convert to decibels (dB)
        db_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        # Save the spectrogram image using helper function save_spectrogram
        save_spectrogram(db_spectrogram, final_file_path)
