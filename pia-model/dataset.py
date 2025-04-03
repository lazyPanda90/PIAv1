import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import List, Optional, Callable

class CSSWebdataset(Dataset):
    
    def __init__(
        self,
        root_dir: str,
        class_names: List[str],
        transform: Optional[Callable] = None,
    ):
        self.root_dir = root_dir
        self.class_names = class_names
        self.transform = transform
    
        print(f"Initializing CSSWebdataset with {len(class_names)} classes")

        self.csv_files = []
        countries_processed = set()
        total_countries = len([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        processed_countries = 0
        
        for country_dir in os.listdir(root_dir):
            country_path = os.path.join(root_dir, country_dir)
            if not os.path.isdir(country_path):
                continue
            
            processed_countries += 1

            if country_dir not in countries_processed:
                print(f"Processing country {country_dir} ({processed_countries}/{total_countries})")
                countries_processed.add(country_dir)

            for website_dir in os.listdir(country_path):
                website_path = os.path.join(country_path, website_dir)
                if not os.path.isdir(website_path):
                    continue
                
                for page_dir in os.listdir(website_path):
                    page_path = os.path.join(website_path, page_dir)
                    if not os.path.isdir(page_path):
                        continue

                    csv_path = os.path.join(page_path, 'elements.csv')
                    if os.path.exists(csv_path):
                        self.csv_files.append(csv_path)

        print(f"\nFound {len(self.csv_files)} CSV files")

        if len(self.csv_files) == 0:
            print(f"Warning: No CSV files found in {root_dir}")
            print("Directory structure should be:")
            print("root_dir/")
            print("  country/")
            print("    website/")
            print("      page/")
            print("        elements.csv")

        self.data = []
        total_elements = 0

        print("Loading data from CSV files...")

        for i, csv_file in enumerate(self.csv_files):
            if i % 10000 == 0 and i > 0:
                print(f"Processed {i}/{len(self.csv_files)} CSV files")

            try:
                df = pd.read_csv(csv_file)
                
                # Extract pre-encoded features directly from CSV
                css_features = df.iloc[:, 4:-1].values  # Skip x,y,width,height and label columns
                bbox_features = df[['x', 'y', 'width', 'height']].values
                labels = df['label'].values

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