# src/audio_processor.py
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_loader import load_fan_data

# helper function to store wav as spectrogram png
def save_spectrogram(spectrogram_data, file_path):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    librosa.display.specshow(spectrogram_data, ax=ax)
    fig.savefig(file_path)
    plt.close(fig)


def process_all_spectrograms(fan_df: pd.DataFrame, output_dir: str):
    """
    Processes all audio files and saves them into a nested directory structure
    that mirrors the source: output_dir/fan_id/label/filename.png
    """
    output_path = Path(output_dir)
    print(f"Processing all {len(fan_df)} files...")

    for index, row in tqdm(fan_df.iterrows(), total=len(fan_df), desc="Processing files"):
        fan_id = row['fan_id']
        label = row['label']

        # Create the nested target directory, e.g., ../processed/fan/id_00/abnormal
        target_dir = output_path / fan_id / label
        target_dir.mkdir(parents=True, exist_ok=True)

        original_wav_path = Path(row['file_path'])
        new_filename = original_wav_path.stem + '.png'
        final_file_path = target_dir / new_filename

        y, sr = librosa.load(original_wav_path)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        db_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        save_spectrogram(db_spectrogram, final_file_path)


if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    DATA_DIR = project_root / 'data' / 'fan'
    OUTPUT_DIR = project_root / 'data' / 'processed' / 'fan'

    full_fan_df = load_fan_data(DATA_DIR)

    if not full_fan_df.empty:
        process_all_spectrograms(fan_df=full_fan_df, output_dir=OUTPUT_DIR)
        print(f"\nAll spectrograms created successfully in '{OUTPUT_DIR}'")




