import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from hyperbolic import poincare_distance, expmap0

class FeatureExtractor(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        modules = list(base_model.children())[:-1]
        self.feature = nn.Sequential(*modules)
        # Dropout added between layers to prevent co-adaptation of neurons
        self.projector = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),     # <-- drops 30% of neurons randomly during training
            nn.Linear(128, 64)
        )
        self.out_dim = 64

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.projector(x)
        return expmap0(x)

def compute_prototypes(features, labels, n_way):
    prototypes = []
    for i in range(n_way):
        class_feats = features[labels == i]
        prototype = class_feats.mean(dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes)

def classify_query(query_feats, prototypes):
    dists = poincare_distance(query_feats, prototypes)
    logits = -dists
    return logits