# src/create_test_data.py
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

# Import the necessary functions
from data_loader import load_fan_data
from audio_processor import save_spectrogram 

import librosa
import numpy as np

def create_test_spectrograms(fan_df: pd.DataFrame, output_dir: str):
    """
    Processes audio files for the held-out test set ('id_06') only.
    """
    # Filter the DataFrame to include ONLY the test fan
    test_df = fan_df[fan_df['fan_id'] == 'id_06'].copy()
    
    output_path = Path(output_dir)
    
    print(f"Starting test set processing. Found {len(test_df)} test files.")
    
    # Loop through the filtered test DataFrame
    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing test files"):
        
        label = row['label']
        target_dir = output_path / label 
        target_dir.mkdir(parents=True, exist_ok=True)
        
        original_wav_path = Path(row['file_path'])
        new_filename = original_wav_path.stem + '.png' 
        final_file_path = target_dir / new_filename
        
        # Perform the conversion from wav to spectrogram image
        y, sr = librosa.load(original_wav_path)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        db_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
        # Save the image using the helper function
        save_spectrogram(db_spectrogram, final_file_path)


if __name__ == '__main__':
    # Define the path relative to the script's location
    # __file__ is the path to the current script (create_test_data.py)
    # .parent gives the directory (src)
    # .parent again gives the project root
    project_root = Path(__file__).parent.parent
    DATA_DIR = project_root / 'data' / 'fan'
    
    print(f"Attempting to load data from: {DATA_DIR}")
    
    # Load all file paths
    full_fan_df = load_fan_data(DATA_DIR)
    
    if full_fan_df.empty:
        print("\n[ERROR] The DataFrame is empty. No .wav files were found.")
        print("Please check the following:")
        print(f"1. Does the directory '{DATA_DIR}' exist?")
        print(f"2. Does it contain the 'id_00', 'id_02', etc. subfolders with .wav files?")
    elif 'fan_id' not in full_fan_df.columns:
        print("\n[ERROR] 'fan_id' column is missing from the DataFrame.")
        print(f"Columns found: {full_fan_df.columns}")
    else:
        # If checks pass, proceed with processing
        TEST_OUTPUT_DIR = project_root / 'data' / 'processed_test' / 'fan'
        create_test_spectrograms(fan_df=full_fan_df, output_dir=TEST_OUTPUT_DIR)
        print(f"\nTest data created successfully in '{TEST_OUTPUT_DIR}'")