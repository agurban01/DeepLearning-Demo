import torch.nn as nn
from models.grl import GradientReversalLayer

class DANN_Model(nn.Module):
    def __init__(self):
        super(DANN_Model, self).__init__()
        
        # 1. Feature Extractor (G_f)
        # Standard CNN. We use 3 input channels because MNIST-M is RGB.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, kernel_size=5),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
        )
        
        # Calculate flattened size: 48 channels * 4 * 4 spatial dimension
        self.flat_features = 48 * 4 * 4

        # 2. Label Predictor (G_y)
        # Classifies digits (0-9) based on features.
        self.class_classifier = nn.Sequential(
            nn.Linear(self.flat_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 10) # Output: 10 digit classes
        )

        # 3. Domain Classifier (G_d)
        # Classifies the domain (Source vs. Target).
        # This branch uses the Gradient Reversal Layer.
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.flat_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 1) # Output: Binary classification (Logits)
        )
        
        self.grl = GradientReversalLayer()

    def forward(self, x, alpha=1.0):
        # Extract features (common representation)
        features = self.feature_extractor(x)
        features = features.view(-1, self.flat_features)
        
        # Branch 1: Label Prediction (Standard training)
        class_output = self.class_classifier(features)
        
        # Branch 2: Domain Prediction (Adversarial training)
        # We pass features through GRL first
        reverse_features = self.grl(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        
        return class_output, domain_output