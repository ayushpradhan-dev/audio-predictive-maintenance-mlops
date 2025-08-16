# src/data_setup.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

def create_logo_dataloaders(
        data_dir: str,
        test_fan_id: str,
        image_size: Tuple[int, int],
        batch_size: int,
        validation_split: float = 0.2,
        random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Creates DataLoaders for a Leave-One-Group-Out (LOGO) cross-validation setup.
    
    Args:
        data_dir: The root directory of the processed image data.
        test_fan_id: The ID of the fan type to hold out for the test set (e.g., 'id_06').
        image_size: Input image resolution typically 244x244 expected for ResNet.
        batch_size: Number of instances processed before weight update.
        validation_split: Percentage of training instances to be used to test model performence within epochs.
        random_seed: Set seed for reproducibility.

    Returns:
        A tuple containing: (train_loader, validation_loader, test_loader, class_to_idx)
    """

    # Prepare transformations for spectrograms images
    data_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load full dataset using ImageFolder with applied transformations
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

    # Create the correct class mapping and override the dataset's labels.
    correct_class_to_idx = {'abnormal': 0, 'normal': 1}
    corrected_samples = []
    for path, _ in full_dataset.samples: # Ignore the incorrect label
        if 'abnormal' in path:
            label = correct_class_to_idx['abnormal']
        else:
            label = correct_class_to_idx['normal']
        corrected_samples.append((path, label))
    
    # Manually override the dataset's internal state
    full_dataset.samples = corrected_samples
    full_dataset.targets = [s[1] for s in corrected_samples]
    full_dataset.class_to_idx = correct_class_to_idx

    # Find indices for the specified test fan
    # 'samples' is a list of (filepath, class_index)
    test_indices = [i for i, (path, _) in enumerate(full_dataset.samples) if test_fan_id in path]

    # All other indices are for the initial training pool
    train_pool_indices = [i for i in range(len(full_dataset)) if i not in test_indices]

    # Create test dataset
    test_dataset = Subset(full_dataset, test_indices)

    # Create a stratified split for the remaining data (train vs. validation)
    train_pool_labels = [full_dataset.targets[i] for i in train_pool_indices]
    train_indices, val_indices = train_test_split(
        train_pool_indices,
        test_size=validation_split,
        stratify=train_pool_labels,
        random_state=random_seed
    )

    train_dataset = Subset(full_dataset, train_indices)
    validation_dataset = Subset(full_dataset, val_indices)

    print(f"[INFO] Hold-out Test Fan: {test_fan_id}")
    print(f"[INFO] Training images: {len(train_dataset)}")
    print(f"[INFO] Validation images: {len(validation_dataset)}")
    print(f"[INFO] Test images: {len(test_dataset)}")

    # Create the final DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader, correct_class_to_idx