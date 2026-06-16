import torch
from torch.utils.data import DataLoader
from model import FeatureExtractor, compute_prototypes, classify_query
from utils import FewShotDataset
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

N_WAY       = 3
K_SHOT      = 10
Q_QUERY     = 5
EPISODES    = 2000
UNFREEZE_AT = 400

print("\nLoading training dataset...")
dataset = FewShotDataset("../dataset/train", n_way=N_WAY, k_shot=K_SHOT,
                          q_query=Q_QUERY, augment=True)
loader  = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])

model = FeatureExtractor(dropout=0.3).to(device)

# Phase 1: freeze backbone
for param in model.feature.parameters():
    param.requires_grad = False
print("Phase 1: Backbone FROZEN\n")

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3, weight_decay=5e-4   # stronger weight decay vs last run (1e-4 → 5e-4)
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPISODES, eta_min=1e-5
)

torch.save(dataset.classes, "class_list.pth")
print(f"Classes: {dataset.classes}")
print(f"Starting: {N_WAY}-way {K_SHOT}-shot | {EPISODES} episodes\n")

best_avg_acc = 0.0
window_acc   = []
window_loss  = []

for episode, (support_images, support_labels, query_images, query_labels) in enumerate(loader):
    if len(support_images) == 0 or len(query_images) == 0:
        continue

    if episode == UNFREEZE_AT:
        for param in model.feature.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam([
            {"params": model.feature.parameters(),   "lr": 5e-5, "weight_decay": 5e-4},
            {"params": model.projector.parameters(), "lr": 2e-4, "weight_decay": 5e-4},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPISODES - UNFREEZE_AT, eta_min=1e-6
        )
        print(f"\nPhase 2 (ep {episode+1}): Backbone UNFROZEN\n")

    support_images = torch.stack(support_images).to(device)
    query_images   = torch.stack(query_images).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    query_labels   = torch.tensor(query_labels).to(device)

    model.train()
    support_feats = model(support_images)
    query_feats   = model(query_images)

    prototypes = compute_prototypes(support_feats, support_labels, n_way=N_WAY)
    logits     = classify_query(query_feats, prototypes)
    loss       = F.cross_entropy(logits, query_labels)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    preds = torch.argmax(logits, dim=1)
    acc   = (preds == query_labels).float().mean().item()

    window_acc.append(acc)
    window_loss.append(loss.item())
    if len(window_acc) > 50:
        window_acc.pop(0)
        window_loss.pop(0)

    if (episode + 1) % 100 == 0:
        avg_acc  = sum(window_acc)  / len(window_acc)  * 100
        avg_loss = sum(window_loss) / len(window_loss)
        lr_now   = optimizer.param_groups[0]['lr']
        print(f"Episode {episode+1:>5}/{EPISODES} | "
              f"Loss: {avg_loss:.4f} | "
              f"Avg Acc: {avg_acc:.1f}% | "
              f"LR: {lr_now:.6f}")

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ New best: {best_avg_acc:.1f}% — saved best_model.pth")

    if episode + 1 >= EPISODES:
        break

torch.save(model.state_dict(), "saved_model.pth")
print(f"\nTraining done! Best avg acc: {best_avg_acc:.1f}%")
print("Saved: best_model.pth, saved_model.pth, class_list.pth")