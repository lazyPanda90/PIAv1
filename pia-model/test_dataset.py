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
            
            # Verify value ranges
            print("\nValue ranges:")
            print(f"CSS features: [{batch['css_features'].min():.3f}, {batch['css_features'].max():.3f}]")
            print(f"Bbox features: [{batch['bbox_features'].min():.3f}, {batch['bbox_features'].max():.3f}]")
            print(f"Labels: [{batch['label'].min()}, {batch['label'].max()}]")
            
        except Exception as e:
            print(f"Error testing {split} loader: {e}")

if __name__ == "__main__":
    test_dataset() 