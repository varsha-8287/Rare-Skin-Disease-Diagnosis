import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from hyperbolic import poincare_distance, expmap0

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        modules = list(base_model.children())[:-1]  # remove final FC
        self.feature = nn.Sequential(*modules)
        # Projection head: 512-dim Euclidean → 64-dim → Poincaré ball
        self.projector = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.out_dim = 64

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)   # [B, 512]
        x = self.projector(x)        # [B, 64] Euclidean
        x = expmap0(x)               # [B, 64] on Poincaré ball
        return x

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