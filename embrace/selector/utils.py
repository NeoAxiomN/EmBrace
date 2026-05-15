import torch
from torch.utils.data import Subset, DataLoader, ConcatDataset
import numpy as np


def stratified_sampling(
    dataset, 
    num_classes, 
    sample_ratio=0.
) -> list:
        """
        Stratified sampling to maintain class distribution.
        
        Args:
            dataset: PyTorch dataset
            num_classes: Number of classes
            sample_ratio: Ratio of data to sample
        
        Returns:
            List of sampled indices
        """
        class_indices = {i: [] for i in range(num_classes)}
        
        # Collect indices for each class
        for idx in range(len(dataset)):
            label = dataset[idx][1]
            
            # Handle different return formats
            if isinstance(label, torch.Tensor):
                label = label.view(-1)[0].item()
            elif isinstance(label, np.ndarray):
                label = label.ravel()[0]
            
            label = int(label)
            if label in class_indices:
                class_indices[label].append(idx)

        
        # Sample from each class proportionally
        sampled_indices = []
        for class_id, indices in class_indices.items():
            if not indices: continue
            
            n_samples = max(1, int(len(indices) * sample_ratio))
            sampled = np.random.choice(indices, n_samples, replace=False)
            sampled_indices.extend(sampled.tolist())
    
        return sampled_indices





def load_stratified_sampling_dataset(
    num_classes,
    batch_size,
    train_data_loader, 
    val_data_loader, 
    sample_ratio=0.2
):
    train_dataset, val_dataset = train_data_loader.dataset, val_data_loader.dataset
    combined_dataset = ConcatDataset([train_dataset, val_dataset])

    sampled_indices = stratified_sampling(
        dataset=combined_dataset, 
        num_classes=num_classes, 
        sample_ratio=sample_ratio
    )
        
    sampled_dataset = Subset(combined_dataset, sampled_indices)

    dataloader = DataLoader(
        sampled_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=train_data_loader.collate_fn if hasattr(train_data_loader, 'collate_fn') else None
    )
    
    print(f"Stratified sampling: {sample_ratio}")
    print(f">> Total samples: {len(combined_dataset)}")
    print(f">> Sampled: {len(sampled_indices)} ({len(sampled_indices)/len(combined_dataset)*100:.1f}%)")
    print(f">> Num classes: {num_classes}")
    
    return dataloader



