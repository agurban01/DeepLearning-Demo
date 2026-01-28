import torch
import torch.optim as optim
import torch.nn as nn
from models.dann import DANN_Model
from data.mnist_m import get_dataloaders
from utils.lambda_schedule import get_lambda
from utils.training_utils import compute_accuracy

def train_dann():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    EPOCHS = 20
    BATCH_SIZE = 64
    LR = 0.01
    
    # Get DataLoaders (Source: MNIST, Target: Synthetic MNIST-M)
    source_loader, target_loader, test_loader = get_dataloaders(BATCH_SIZE)
    
    # Initialize Model
    model = DANN_Model().to(device)
    
    # Optimizer (updates both Feature Extractor and Classifiers)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    
    # Loss Functions
    criterion_class = nn.CrossEntropyLoss() # For digit classification
    criterion_domain = nn.BCEWithLogitsLoss() # For domain classification (Binary)

    print("--- Starting Domain-Adversarial Training (DANN) ---")
    
    for epoch in range(EPOCHS):
        model.train()
        
        # Calculate dynamic lambda (alpha)
        alpha = get_lambda(epoch, EPOCHS)
        
        len_dataloader = min(len(source_loader), len(target_loader))
        total_loss = 0
        total_domain_loss = 0
        
        # Zip source and target batches to train simultaneously
        for i, ((data_s, label_s), (data_t, _)) in enumerate(zip(source_loader, target_loader)):
            
            data_s, label_s = data_s.to(device), label_s.to(device)
            data_t = data_t.to(device)
            
            optimizer.zero_grad()
            
            # ---------------------------
            # 1. Train on Source Domain
            # ---------------------------
            # We want to classify the digit AND recognize it's from Source (0)
            class_out_s, domain_out_s = model(data_s, alpha=alpha)
            
            loss_s_label = criterion_class(class_out_s, label_s)
            
            # Domain label for Source is 0
            domain_label_s = torch.zeros(data_s.size(0), 1).to(device)
            loss_s_domain = criterion_domain(domain_out_s, domain_label_s)
            
            # ---------------------------
            # 2. Train on Target Domain
            # ---------------------------
            # We don't have digit labels, only domain labels (1)
            _, domain_out_t = model(data_t, alpha=alpha)
            
            # Domain label for Target is 1
            domain_label_t = torch.ones(data_t.size(0), 1).to(device)
            loss_t_domain = criterion_domain(domain_out_t, domain_label_t)
            
            # ---------------------------
            # 3. Optimization Step
            # ---------------------------
            # Total Loss = Classification Loss + Domain Loss
            # Note: The GRL inside the model handles the sign reversal for the backbone.
            # We simply sum the losses here.
            loss = loss_s_label + loss_s_domain + loss_t_domain
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_domain_loss += (loss_s_domain.item() + loss_t_domain.item())
            
        # Evaluation on Target Data
        acc = compute_accuracy(model, test_loader, device)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len_dataloader:.4f} | "
              f"Dom Loss: {total_domain_loss/len_dataloader:.4f} | "
              f"Alpha: {alpha:.2f} | Target Acc: {acc:.2f}%")

    print("DANN training complete.")
    torch.save(model.state_dict(), "dann_model.pth")

if __name__ == "__main__":
    train_dann()