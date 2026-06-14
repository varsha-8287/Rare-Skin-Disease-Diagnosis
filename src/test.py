import torch
import numpy as np
from torch.utils.data import DataLoader
from model import FeatureExtractor, compute_prototypes, classify_query
from utils import FewShotDataset
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_WAY    = 3
K_SHOT   = 10   # match train k_shot for fair evaluation
Q_QUERY  = 5
EPISODES = 200

class_list = torch.load("class_list.pth")
print(f"Classes: {class_list}")

print("\nLoading test dataset...")
dataset = FewShotDataset("../dataset/test", n_way=N_WAY, k_shot=K_SHOT,
                          q_query=Q_QUERY, class_list=class_list, augment=False)
loader  = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

model = FeatureExtractor().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
print("Model loaded successfully.\n")

accs = []
class_correct = defaultdict(int)
class_total   = defaultdict(int)

with torch.no_grad():
    for i, (support_images, support_labels, query_images, query_labels) in enumerate(loader):
        if len(support_images) == 0 or len(query_images) == 0:
            continue

        support_images = torch.stack(support_images).to(device)
        query_images   = torch.stack(query_images).to(device)
        support_labels = torch.tensor(support_labels).to(device)
        query_labels   = torch.tensor(query_labels).to(device)

        support_feats = model(support_images)
        query_feats   = model(query_images)

        prototypes = compute_prototypes(support_feats, support_labels, n_way=N_WAY)
        logits     = classify_query(query_feats, prototypes)

        preds = torch.argmax(logits, dim=1)
        acc   = (preds == query_labels).float().mean().item()
        accs.append(acc)

        for pred, true in zip(preds.cpu(), query_labels.cpu()):
            class_correct[true.item()] += (pred == true).item()
            class_total[true.item()]   += 1

        if i + 1 >= EPISODES:
            break

accs = np.array(accs)
mean = accs.mean() * 100
ci   = 1.96 * accs.std() / np.sqrt(len(accs)) * 100

print(f"Overall Test Accuracy ({len(accs)} episodes): {mean:.2f}% ± {ci:.2f}% (95% CI)")
print(f"Random baseline for {N_WAY}-way: {100/N_WAY:.1f}%")
gap = mean - (100/N_WAY)
print(f"Improvement over random: +{gap:.1f}%\n")

print("Per-class accuracy:")
for idx, cls in enumerate(class_list):
    if class_total[idx] > 0:
        cls_acc = class_correct[idx] / class_total[idx] * 100
        bar = "█" * int(cls_acc / 5)
        print(f"  {cls:<25} {cls_acc:5.1f}%  {bar}")