import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple
import numpy as np
from pathlib import Path

class WebDataset(Dataset):
    def __init__(self, tensor_data: torch.Tensor):
        """
        Args:
            tensor_data: En 2D tensor hvor hver rad representerer et webelement.
                        Kolonner er [x, y, w, h, css_1, ..., css_102, label].
        """
        self.data = tensor_data

    # Gir antall rader i tensoren = antall web-elementer
    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        row = self.data[idx]
        bbox_features = row[:4]  # Første 4 kolonner: [x, y, w, h]
        css_features = row[4:-1]  # CSS-attributter
        label = row[-1]  # Siste kolonne: Label
        
        return {
            'bbox_features': bbox_features.float(),
            'css_features': css_features.float(),
            'label': label.long()
        }

def load_csv_files(directory: str) -> torch.Tensor:
    """
    Laster alle CSV-filer fra en gitt mappe og konverterer til en tensor.
    """
    all_data = []
    root_dir = Path(directory)
    
    print(f"Laster data fra {directory}")
    
    # Gå gjennom alle landkoder
    for country_dir in root_dir.iterdir():
        if not country_dir.is_dir():
                continue
            
        print(f"Prosesserer land: {country_dir.name}")
        
        # Gå gjennom alle domener
        for domain_dir in country_dir.iterdir():
            if not domain_dir.is_dir():
                continue
                
            # Gå gjennom alle sidenumre
            for page_dir in domain_dir.iterdir():
                if not page_dir.is_dir():
                    continue
                
                csv_path = page_dir / 'elements.csv'
                if not csv_path.exists():
                        continue

                try:
                    df = pd.read_csv(csv_path)
                    
                    # Ekstraher features
                    bbox_features = df[['x', 'y', 'width', 'height']].values
                    css_features = df.iloc[:, 4:-1].values
                    labels = df['label'].values.reshape(-1, 1)
                    
                    # Kombiner alle features
                    combined = np.hstack([bbox_features, css_features, labels])
                    all_data.append(combined)
                    
                except Exception as e:
                    print(f"Feil ved lasting av {csv_path}: {e}")
    
    if not all_data:
        raise ValueError(f"Ingen data funnet i {directory}")
        
    # Konverter til tensor
    return torch.tensor(np.vstack(all_data))

def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Oppretter DataLoader-objekter for trening, validering og testing.
    """
    # Last data fra hver mappe
    train_tensor = load_csv_files(os.path.join(data_dir, 'train'))
    val_tensor = load_csv_files(os.path.join(data_dir, 'val'))
    test_tensor = load_csv_files(os.path.join(data_dir, 'test'))
    
    print(f"Datasett størrelser:")
    print(f"Train: {train_tensor.size(0)} eksempler")
    print(f"Val: {val_tensor.size(0)} eksempler")
    print(f"Test: {test_tensor.size(0)} eksempler")
    
    # Opprett datasett
    train_dataset = WebDataset(train_tensor)
    val_dataset = WebDataset(val_tensor)
    test_dataset = WebDataset(test_tensor)
    
    # Opprett dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

# Eksempel på bruk:
if __name__ == "__main__":
    # Anta at dataset-mappen ligger ved siden av pia-model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'dataset')
    
    try:
        loaders = create_data_loaders(
            data_dir=data_dir,
            batch_size=32,
            num_workers=4
        )
        print("Dataloaders opprettet vellykket!")
        
        # Test første batch
        train_batch = next(iter(loaders['train']))
        print("\nEksempel på batch-struktur:")
        print(f"Bbox features shape: {train_batch['bbox_features'].shape}")
        print(f"CSS features shape: {train_batch['css_features'].shape}")
        print(f"Labels shape: {train_batch['label'].shape}")
        
    except Exception as e:
        print(f"Feil ved lasting av data: {e}")
