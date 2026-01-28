import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

class MNISTM_Synthetic(Dataset):
    """
    Synthetic MNIST-M dataset.
    Generates data on-the-fly by taking MNIST digits and blending them 
    with random colored noise backgrounds to simulate domain shift.
    """
    def __init__(self, train=True, transform=None):
        # Load standard MNIST
        self.mnist = datasets.MNIST(root='./data', train=train, download=True)
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        
        # Convert grayscale to RGB to allow coloring
        image = image.convert('RGB')
        np_img = np.array(image)
        
        # --- Domain Shift Simulation ---
        # 1. Generate random colored background noise
        background = np.random.randint(0, 255, size=np_img.shape, dtype=np.uint8)
        
        # 2. Create a mask of the digit (where pixel value > threshold)
        mask = np_img > 30 
        
        # 3. Blend: Invert colors where the digit is, use noise background elsewhere
        img_colored = np.where(mask, 255 - np_img, background).astype(np.uint8)
        
        final_image = Image.fromarray(img_colored)

        if self.transform:
            final_image = self.transform(final_image)

        return final_image, label

def get_dataloaders(batch_size=64):
    """
    Returns DataLoaders for Source (MNIST) and Target (MNIST-M).
    Note: Source MNIST is converted to 3 channels to match Target architecture.
    """
    # Transform for Target (MNIST-M is already RGB)
    tf = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    # Transform for Source (MNIST is Grayscale -> needs to be 3 channels)
    tf_source = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Source: Standard MNIST (Labeled)
    source_data = datasets.MNIST(root='./data', train=True, transform=tf_source, download=True)
    source_loader = DataLoader(source_data, batch_size=batch_size, shuffle=True, drop_last=True)

    # Target: Synthetic MNIST-M (Unlabeled for training purposes)
    target_data = MNISTM_Synthetic(train=True, transform=tf)
    target_loader = DataLoader(target_data, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Test: Target domain for evaluation
    test_target_data = MNISTM_Synthetic(train=False, transform=tf)
    test_loader = DataLoader(test_target_data, batch_size=batch_size, shuffle=False)
    
    return source_loader, target_loader, test_loader