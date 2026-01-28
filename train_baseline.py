import torch
import torch.optim as optim
import torch.nn as nn
from models.dann import DANN_Model
from data.mnist_m import get_dataloaders
from utils.training_utils import compute_accuracy

def train_baseline():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    EPOCHS = 10
    BATCH_SIZE = 64
    LR = 0.01

    # Load Data
    source_loader, _, test_loader = get_dataloaders(BATCH_SIZE)
    
    # Initialize Model
    model = DANN_Model().to(device)
    
    # Optimizer & Loss
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    print("--- Starting Baseline Training (Source Only) ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for images, labels in source_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (alpha=0 disables the gradient reversal effect)
            class_output, _ = model(images, alpha=0)
            
            loss = criterion(class_output, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Evaluation
        acc = compute_accuracy(model, test_loader, device)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(source_loader):.4f} | Target Acc: {acc:.2f}%")

    print("Baseline training complete.")
    torch.save(model.state_dict(), "baseline_model.pth")

if __name__ == "__main__":
    train_baseline()