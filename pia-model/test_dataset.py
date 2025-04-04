import torch
from dataset import create_data_loaders

def test_dataset():
    print("Initializing dataset...")
    
    # Initialize data loaders with pre-split data
    class_names = ['button', 'text_field', 'checkbox', 'radio_button']
    dataloaders = create_data_loaders(
        root_dir="../dataset/dataset-klarna-new",
        class_names=class_names,
        batch_size=32,
        num_workers=4,
        shuffle=True
    )
    
    # Test each loader
    for split, loader in dataloaders.items():
        print(f"\nTesting {split} loader...")
        try:
            # Get first batch
            batch = next(iter(loader))
            
            # Print batch statistics
            print(f"Batch size: {batch['css_features'].size(0)}")
            print(f"Max elements in batch: {batch['css_features'].size(1)}")
            print(f"Feature dimensions: {batch['css_features'].size(2)}")
            print(f"Bbox dimensions: {batch['bbox_features'].size(2)}")
            print(f"Number of valid elements (attention mask): {batch['attention_mask'].sum().item()}")
            
            # Verify data types
            print("\nData types:")
            for key, tensor in batch.items():
                print(f"{key}: {tensor.dtype}")
            
            # Print detailed bbox information
            print("\nBbox feature details:")
            print(f"Shape: {batch['bbox_features'].shape}")  # Should be [batch_size, max_elements, 4]
            print("\nFirst element's bbox features:")
            first_element = batch['bbox_features'][0][0]  # First element of first page
            print(f"x: {first_element[0]:.2f}")
            print(f"y: {first_element[1]:.2f}")
            print(f"width: {first_element[2]:.2f}")
            print(f"height: {first_element[3]:.2f}")
            
            # Print value ranges for each bbox component
            print("\nBbox value ranges:")
            print(f"x range: [{batch['bbox_features'][..., 0].min():.2f}, {batch['bbox_features'][..., 0].max():.2f}]")
            print(f"y range: [{batch['bbox_features'][..., 1].min():.2f}, {batch['bbox_features'][..., 1].max():.2f}]")
            print(f"width range: [{batch['bbox_features'][..., 2].min():.2f}, {batch['bbox_features'][..., 2].max():.2f}]")
            print(f"height range: [{batch['bbox_features'][..., 3].min():.2f}, {batch['bbox_features'][..., 3].max():.2f}]")
            
            # Print CSS feature ranges
            print("\nCSS feature ranges:")
            print(f"CSS features: [{batch['css_features'].min():.3f}, {batch['css_features'].max():.3f}]")
            print(f"Labels: [{batch['label'].min()}, {batch['label'].max()}]")
            
        except Exception as e:
            print(f"Error testing {split} loader: {e}")

if __name__ == "__main__":
    test_dataset() 