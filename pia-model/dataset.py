import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Optional, Callable, Dict, Any

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-sized pages.
    Instead of stacking tensors, we'll pad them to the maximum size in the batch.
    """
    # Find max number of elements in the batch
    max_elements = max(item['css_features'].size(0) for item in batch)
    feature_dim = batch[0]['css_features'].size(1)
    bbox_dim = batch[0]['bbox_features'].size(1)
    
    # Initialize padded tensors
    batch_size = len(batch)
    css_features = torch.zeros(batch_size, max_elements, feature_dim)
    bbox_features = torch.zeros(batch_size, max_elements, bbox_dim)
    labels = torch.zeros(batch_size, max_elements, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_elements, dtype=torch.bool)
    
    # Fill in the actual values and create attention mask
    for i, item in enumerate(batch):
        num_elements = item['css_features'].size(0)
        css_features[i, :num_elements] = item['css_features']
        bbox_features[i, :num_elements] = item['bbox_features']
        labels[i, :num_elements] = item['label']
        attention_mask[i, :num_elements] = True
    
    return {
        'css_features': css_features,
        'bbox_features': bbox_features,
        'label': labels,
        'attention_mask': attention_mask
    }

class CSSWebdataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        class_names: List[str],
        split: str = 'train',  # Added split parameter
        transform: Optional[Callable] = None,
    ):
        self.root_dir = root_dir
        self.class_names = class_names
        self.transform = transform
        self.split = split
    
        print(f"Initializing CSSWebdataset for {split} split with {len(class_names)} classes")

        self.csv_files = []
        
        # Only process the specified split directory
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist")
                
        print(f"Processing {split} directory...")
        
        for country in os.listdir(split_dir):
            country_path = os.path.join(split_dir, country)
            if not os.path.isdir(country_path):
                continue
            
            for website in os.listdir(country_path):
                website_path = os.path.join(country_path, website)
                if not os.path.isdir(website_path):
                    continue
                
                for page in os.listdir(website_path):
                    page_path = os.path.join(website_path, page)
                    if not os.path.isdir(page_path):
                        continue
                    
                    csv_path = os.path.join(page_path, 'elements.csv')
                    if os.path.exists(csv_path):
                        self.csv_files.append(csv_path)

        print(f"\nFound {len(self.csv_files)} CSV files in {split} split")

        if len(self.csv_files) == 0:
            print(f"Warning: No CSV files found in {split_dir}")
            print("Directory structure should be:")
            print("root_dir/")
            print("  train/")
            print("    country/")
            print("      website/")
            print("        page/")
            print("          elements.csv")

        self.data = []
        total_elements = 0

        print("Loading data from CSV files...")

        for i, csv_file in enumerate(self.csv_files):
            if i % 1000 == 0 and i > 0:
                print(f"Processed {i}/{len(self.csv_files)} CSV files")

            try:
                df = pd.read_csv(csv_file)
                
                # Extract pre-encoded features and convert to tensors
                css_features = torch.tensor(df.iloc[:, 4:-1].values, dtype=torch.float32)  # CSS features
                bbox_features = torch.tensor(df[['x', 'y', 'width', 'height']].values, dtype=torch.float32)  # Bounding box features
                labels = torch.tensor(df['label'].values, dtype=torch.long)  # Labels as long integers

                # Validate labels
                if not all(0 <= label < len(self.class_names) for label in labels):
                    raise ValueError(f"Found labels outside the expected range [0, {len(self.class_names)}]")
                
                self.data.append({
                    'css_features': css_features,
                    'bbox_features': bbox_features,
                    'label': labels
                })
                total_elements += len(labels)

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

        print(f"\nLoaded {len(self.data)} pages with total of {total_elements} elements")
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.transform:
            item = self.transform(item)
            
        return item

def create_data_loaders(
    root_dir: str,
    class_names: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test data loaders using pre-split data.
    
    Args:
        root_dir: Root directory containing train/val/test splits
        class_names: List of class names
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        
    Returns:
        Dictionary containing train, val, and test data loaders
    """
    # Create datasets for each split
    train_dataset = CSSWebdataset(root_dir, class_names, split='train')
    val_dataset = CSSWebdataset(root_dir, class_names, split='val')
    test_dataset = CSSWebdataset(root_dir, class_names, split='test')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }