import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from model import FeatureExtractor
from utils import FewShotDataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = FeatureExtractor().to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Load test dataset (or train)
dataset = FewShotDataset("../dataset/test", n_way=2, k_shot=5, q_query=2)
loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

features = []
labels = []

# Collect embeddings
with torch.no_grad():
    for batch in loader:
        support_images, support_labels, query_images, query_labels = batch[0]

        all_images = support_images + query_images
        all_labels = support_labels + query_labels

        all_images = torch.stack(all_images).to(device)
        embeddings = model(all_images).cpu().numpy()

        features.extend(embeddings)
        labels.extend(all_labels)

        if len(features) > 100:  # limit points for clarity
            break

# t-SNE projection
tsne = TSNE(n_components=2, perplexity=5, init='random', random_state=42)
tsne_results = tsne.fit_transform(np.array(features))


# Plot
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i, label in enumerate(set(labels)):
    idxs = [j for j, l in enumerate(labels) if l == label]
    plt.scatter(tsne_results[idxs, 0], tsne_results[idxs, 1], label=f'Class {label}', alpha=0.7, color=colors[i])

plt.title("t-SNE of Support + Query Feature Embeddings")
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_plot.png")
print("t-SNE plot saved as tsne_plot.png")
plt.show()
