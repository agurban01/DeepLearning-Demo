import torch

def compute_accuracy(model, dataloader, device):
    """Calculates classification accuracy on a given dataset."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            # We only care about class output [0] here
            outputs, _ = model(images) 
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total