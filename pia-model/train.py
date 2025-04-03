import torch
import torch.nn as nn
from typing import List, Optional, Callable
from torch.utils.data import DataLoader
from dataset import CSSWebdataset

def create_data_loaders(
    train_root: str,
    test_root: str,
    class_names: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    transform: Optional[Callable] = None
) -> tuple[DataLoader, DataLoader]:
    """Create data loaders for training and testing.
    
    Args:
        train_root: Root directory for training data
        test_root: Root directory for test data
        class_names: List of class names
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        transform: Optional transform to apply to the data
    """
    train_dataset = CSSWebdataset(
        root_dir=train_root,
        class_names=class_names,
        transform=transform
    )
    
    test_dataset = CSSWebdataset(
        root_dir=test_root,
        class_names=class_names,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device
) -> nn.Module:
    """Train the PIA model.
    
    Args:
        model: The PIA model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on
    """
    best_test_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        print("Training...")

        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            css_features = batch['css_features'].to(device)
            bbox_features = batch['bbox_features'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(css_features, bbox_features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update training metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Print training progress
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {train_loss/(batch_idx+1):.4f}, "
                      f"Accuracy: {100.*train_correct/train_total:.2f}%")
                
        # Validation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        print("Validating...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                css_features = batch['css_features'].to(device)
                bbox_features = batch['bbox_features'].to(device)
                labels = batch['label'].to(device)

                outputs = model(css_features, bbox_features)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                if batch_idx % 100 == 0:
                    print(f"Batch {batch_idx}/{len(test_loader)} - Loss: {test_loss/(batch_idx+1):.4f}, "
                          f"Accuracy: {100.*test_correct/test_total:.2f}%")
                    
        # Calculate validation accuracy
        test_acc = 100.*test_correct/test_total
        print(f"Validation Accuracy: {test_acc:.2f}%")

        # Save model if it has the highest accuracy so far
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved")

        # Update learning rate
        scheduler.step()

    print("Training complete")
    return model

    
    #Testing phase

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    print("Testing...")

    with.torch.no_grad()
        for batch_idx, (css_features, bbox_features, labels) in enumerate(test_loader):
             #move data to device
             css_features = css_features.to(device)
             bbox_features = bbox_features.to(device)
             labels = labels.to(device)

             #Forward pass
             outputs = model(css_features, bbox_features)
             loss = criterion(outputs, labels)

             #Update testing metrics
             test_loss += loss.item()
             _, predicted = torch.max(outputs, 1)
             test_total += labels.size(0)
             test_correct += (predicted == labels).sum().item()

             if batch_idx % 100 == 0:
                  print(f"Batch {batch_idx}/{len(test_loader)} - Loss: {test_loss/{batch_idx+1}:.4f}, Accuracy: {100.*test_correct/test_total:.2f}%")
                
        #Calculate epoch statistics