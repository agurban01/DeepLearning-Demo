import torch
import numpy as np

def collect_features(model, dataloader, device, max_samples=500):
    """
    Passes data through the model to extract features from the bottleneck layer
    without calculating gradients. Used for t-SNE visualization.
    
    Returns:
        features: Numpy array of extracted features.
        domain_labels: Array indicating if sample is Source (0) or Target (1).
    """
    model.eval()
    features_list = []
    
    # We only need a subset of data for visualization (e.g., 500 samples)
    count = 0
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            
            # Extract features (output of G_f)
            feats = model.feature_extractor(imgs)
            feats = feats.view(feats.size(0), -1) # Flatten
            
            features_list.append(feats.cpu().numpy())
            count += imgs.size(0)
            if count > max_samples:
                break
                
    return np.concatenate(features_list)[:max_samples]