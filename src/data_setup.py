from pathlib import Path
from typing import Tuple, Dict, List

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


def create_dataloaders(
        data_dir: str,
        image_size: Tuple[int, int],
        batch_size:int,
        validation_split: float = 0.2,
        random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Creates training and validation DataLoaders from an image folder.
    
    Applies necessary transformations for pre-trained ResNet models and performs
    a stratified split to maintain class proportions in both train and validation sets.
    
    Args:
        data_dir: The root directory of the image data, with subfolders for each class.
        image_size: The target size (height, width) for the images.
        batch_size: The number of samples per batch in the DataLoaders.
        validation_split: The fraction of the data to use for the validation set.
        random_seed: Seed for the random split for reproducibility.
        
    Returns:
        A tuple containing: (train_dataloader, validation__dataloader, class_to_idx)
    """

    # Define image transformations to match the pre-trained model's requirements
    data_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create the initial dataset with ImageFolder
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    class_to_idx = full_dataset.class_to_idx

    # Create a stratified train/validation split
    labels = full_dataset.targets
    indices = list(range(len(labels)))
    train_indices, val_indices = train_test_split(
        indices, test_size=validation_split, stratify=labels, random_state=random_seed 
    )

    # Create subset objects for train and validation
    train_dataset = Subset(full_dataset, train_indices)
    validation_dataset = Subset(full_dataset, val_indices)

    print(f"[INFO] Total images: {len(full_dataset)}")
    print(f"[INFO] Training images: {len(train_dataset)}")
    print(f"[INFO] Validation images: {len(validation_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, validation_loader, class_to_idx


# Block to test the script
if __name__ == '__main__':
    # Define test parameters
    DATA_DIR = '../data/processed/fan'
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32

    # Create the dataloaders
    train_loader, validation_loader, class_to_idx = create_dataloaders(
        data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # Verify the output
    print(f"\n[INFO] Class to index mapping: {class_to_idx}")
    train_features, train_labels = next(iter(train_loader))
    print(f"\n[INFO] Feature batch shape: {train_features.size()}")
    print(f"[INFO] Labels batch shape: {train_labels.size()}")
