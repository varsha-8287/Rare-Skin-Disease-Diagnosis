# predict.py
import torch
from torchvision import transforms
from PIL import Image
from model import FeatureExtractor, compute_prototypes, classify_query
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = FeatureExtractor().to(device)
model.eval()

# Load saved model
model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))

# Class names in the same order as during training
class_names = ["melanoma", "nevus"]  # Adjust if you have more classes

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    query_image = transform(image).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

    # Dummy support set (replace with better samples if available)
    support_images = []
    support_labels = []

    for idx, cls in enumerate(class_names):
        # Load one representative image per class
        sample_path = os.path.join("..", "dataset", "train", cls)
        sample_image = os.listdir(sample_path)[0]  # first image
        img = Image.open(os.path.join(sample_path, sample_image)).convert("RGB")
        img = transform(img).to(device)
        support_images.append(img)
        support_labels.append(idx)

    support_images = torch.stack(support_images)
    support_labels = torch.tensor(support_labels).to(device)

    # Extract features
    support_feats = model(support_images)
    query_feat = model(query_image)

    # Compute prototypes & classify
    prototypes = compute_prototypes(support_feats, support_labels, n_way=len(class_names))
    logits = classify_query(query_feat, prototypes)

    # Prediction & confidence
    probs = torch.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = torch.max(probs).item()
    predicted_class = class_names[pred_idx]

    return predicted_class, confidence

