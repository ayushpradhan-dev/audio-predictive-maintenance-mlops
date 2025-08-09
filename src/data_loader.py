# src/data_loader.py
import pandas as pd
from pathlib import Path
from typing import Union

def load_fan_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """
    Scans the MIMII (Malfunctioning Industrial Machine Investigation) dataset directory
    to create a DataFrame of file paths and their labels.
    
    As filenames are resued, this function relies on the parent directory names ('normal', 'abnormal')
    and grandparent directory names('id_00', 'id_02', etc.) to uniquely identify and label each audio clip.
    
    Args:
        data_path: The file path to the root 'fan' data directory (e.g.m '../data/fan')
        
    Returns:
        A pandas DataFrame with columns: 'file_path', 'fan_id', and 'label'.
    """

    # Convert the input string path into a Path object
    data_root = Path(data_path)

    # Use glob to recursively find all files ending in .wav.
    # ** pattern searches the current directory and all subdirectories.
    all_wav_paths = list(data_root.glob('**/*.wav'))

    # Create a list to store dictionaries, where each dictionary holds one row of data.
    metadata = []
    for path in all_wav_paths:
        # Extract the 'label' (e.g., 'normal', 'abnormal') from the immediate parent fodler's name.
        label = path.parent.name

        # Extract the 'fan_id' (e.g., 'id_00') from grandparent folder's name.
        fan_id = path.parent.parent.name

        metadata.append({
            'file_path': str(path),
            'fan_id': fan_id,
            'label': label
        }) 
    
    # Create final DataFrame from the list of metadata.
    return pd.DataFrame(metadata)