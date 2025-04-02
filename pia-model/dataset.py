import os
from torch.utils.data import Dataset

class CSSWebdataset(Dataset):
    
    def __init__(
        self,
        root_dir,
        class_names: List[str],
        transform: Optional[Callable] = None,
    ):
        self.root_dir = root_dir
        self.class_names = class_names
        self.transform = transform
    
        print(f"Initializing CSSWebdataset with {len(class_names)} classes")

        self.csv:files = []
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
                
                for page_dir in os.listdir(website_dir):
                    page_path = os.path.join(website_path, page_dir)
                    if not os.path.isdir(page_path):
                        continue

                    csv_path = os.oath.join(page_path, 'elements.csv')
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
        self.labels = []
        total_countries = 0

        print("Loading data from CSV files...")

        for i, csv_file in enumerate(self.csv_files):
            if i % 10000 == 0 and i > 0:
                print(f"Processed {i}/{len(self.csv_files)} CSV files")

            try:
                df = pd.read_csv(csv_file)
                
                css_features = self._process_css_features(df)

                bbox_features = = df[['x', 'y', 'width', 'height']].values
                
                label = self._process_label(df)

                if not all(0 <= label < len(self.class_names) for label in labels):
                    raise ValueError(f"Found labels outsite the expected range [0, {len(self.class_names)}]")
                
                self.data.append({
                    'css_features': css_features,
                    'bbox_features': bbox_features,
                    'label': label
                })
                total_elements += len(labels)

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

        print(f"\nLoaded {len(self.data)} pages with total of {total_elements} elements")
            
    def _process_css_features(self, df: pd.DataFrame) -> np.ndarray:

        display_map = {
            'none': 0,
            'block': 1,
            'inline': 2,
            'inline-block': 3}
        visibility_map = {
            'visible': 0,
            'hidden': 1,
        }
        text_align_map = {
            'left': 0,
            'center': 1,
            'right': 2,
            'justify': 3,
        }

        features = []

        
            
        }