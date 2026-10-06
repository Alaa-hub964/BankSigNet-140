"""
Baseline 2: Two-Stage Siamese Network with Spatial Transformation + Focal Loss
================================================================================
Based on: "Two-stage Siamese network framework for offline handwritten
signature verification with spatial transformation module and focal loss"

Run on Google Colab with GPU runtime.
Same train/val/test split as MSFV-Net for fair comparison.
"""



import os, random, pickle, numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from itertools import combinations
from PIL import Image

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = Path("extracted")
SPLITS_PATH  = Path("splits_v12_final.pkl")
SEED         = 42
EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 1e-4
PATIENCE     = 10
FOCAL_GAMMA  = 2.0   # Focal loss gamma
FOCAL_ALPHA  = 0.75  # Focal loss alpha (weight for positive class)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
print(f"✅ Device: {DEVICE}")

#  CELL 2: Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        prob = torch.sigmoid(logits)
        p_t  = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()

#  CELL 3: Spatial Transformation Module ─
class SpatialTransformModule(nn.Module):
    """Lightweight STN that localises and aligns signature regions."""
    def __init__(self, in_channels=1024):
        super().__init__()
        self.localisation = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(in_channels * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )
        self.localisation[-1].weight.data.zero_()
        self.localisation[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x, feature_map):
        theta = self.localisation(feature_map)
        theta = theta.view(-1, 2, 3)
        grid  = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

#  CELL 4: Two-Stage Siamese Model ─
class TwoStageSiamese(nn.Module):
    def __init__(self):
        super().__init__()
        # Stage 1: ResNet50 backbone (same as MSFV-Net for fair comparison)
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Stage 1: up to layer3 for STM insertion
        self.stage1  = nn.Sequential(*list(backbone.children())[:8])   # up to layer3
        self.stage2  = nn.Sequential(*list(backbone.children())[8:9])  # layer4
        self.pool    = nn.AdaptiveAvgPool2d(1)

        # Spatial transform applied after stage1 (layer3 output = 1024ch)
        self.stm = SpatialTransformModule(in_channels=1024)

        # Embedding head
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512)
        )

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward_one(self, x):
        # Stage 1 — up to layer3
        f1 = self.stage1(x)              # [B, 1024, 14, 14]
        # Spatial transform on input guided by stage1 features
        x_transformed = self.stm(x, f1)
        # Re-run stage1 on transformed input
        f1t = self.stage1(x_transformed)
        # Stage 2 — layer4
        f3  = self.stage2(f1t)
        out = self.pool(f3)
        return F.normalize(self.embed(out), dim=1)

    def forward(self, x1, x2):
        e1 = self.forward_one(x1)
        e2 = self.forward_one(x2)
        return self.classifier(torch.cat([e1, e2], dim=1)).squeeze(1)

#  CELL 5: Dataset (reuse from baseline 1) ─
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def get_script(pid): return pid[0].upper()

def load_all_images(root):
    data = {}
    for split in ['genuine', 'forged']:
        for person_dir in sorted((root / split).iterdir()):
            if not person_dir.is_dir(): continue
            pid  = person_dir.name
            imgs = sorted(list(person_dir.glob("*.png")) +
                         list(person_dir.glob("*.jpg")))
            if pid not in data:
                data[pid] = {'genuine': [], 'forged': []}
            data[pid][split] = imgs
    return data

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        return (self.transform(Image.open(p1).convert("RGB")),
                self.transform(Image.open(p2).convert("RGB")),
                torch.tensor(label, dtype=torch.float32))

def make_pairs(data, pids, n_pairs=8000):
    pairs = []
    pids  = [p for p in pids if data[p]['genuine'] and data[p]['forged']]
    for pid in pids:
        imgs = data[pid]['genuine']
        for i,j in combinations(range(len(imgs)), 2):
            pairs.append((imgs[i], imgs[j], 1))
    for pid in pids:
        for g in data[pid]['genuine']:
            for f in data[pid]['forged']:
                pairs.append((g, f, 0))
    random.shuffle(pairs)
    return pairs[:n_pairs]

all_data = load_all_images(DATASET_ROOT)
all_pids = sorted(all_data.keys())

if SPLITS_PATH.exists():
    with open(SPLITS_PATH, 'rb') as f:
        splits = pickle.load(f)
    train_ids = splits.get('train', all_pids[:98])
    val_ids   = splits.get('val',   all_pids[98:119])
    test_ids  = splits.get('test',  all_pids[119:])
else:
    random.shuffle(all_pids)
    n = len(all_pids)
    train_ids = all_pids[:int(0.70*n)]
    val_ids   = all_pids[int(0.70*n):int(0.85*n)]
    test_ids  = all_pids[int(0.85*n):]

print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

train_loader = DataLoader(PairDataset(make_pairs(all_data, train_ids), transform),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(PairDataset(make_pairs(all_data, val_ids, 1500), transform),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

#  CELL 6: Training
model     = TwoStageSiamese().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

best_val_acc = 0
patience_cnt = 0
best_state   = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for x1, x2, labels in train_loader:
        x1, x2, labels = x1.to(DEVICE), x2.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(x1, x2)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_preds, val_labels, val_loss = [], [], 0
    with torch.no_grad():
        for x1, x2, labels in val_loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            out  = model(x1, x2)
            loss = criterion(out, labels.to(DEVICE))
            val_loss += loss.item()
            preds = (torch.sigmoid(out) > 0.5).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.numpy())

    val_acc = accuracy_score(val_labels, val_preds) * 100
    scheduler.step(val_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train={train_loss/len(train_loader):.4f} | "
              f"val={val_loss/len(val_loader):.4f} | "
              f"val_acc={val_acc:.1f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"Early stop at epoch {epoch}")
            break

#  CELL 7: Evaluation
model.load_state_dict(best_state)
model.eval()

def evaluate(model, data, person_ids, transform, device):
    all_preds, all_labels, all_scores = [], [], []
    script_results = {'A': [], 'E': [], 'H': []}

    for pid in person_ids:
        if not data[pid]['genuine'] or not data[pid]['forged']: continue
        script   = get_script(pid)
        genuines = data[pid]['genuine']
        forgeds  = data[pid]['forged']

        gallery_embs = []
        with torch.no_grad():
            for gp in genuines:
                img = transform(Image.open(gp).convert("RGB")).unsqueeze(0).to(device)
                gallery_embs.append(model.forward_one(img))
        gallery = torch.stack(gallery_embs).mean(0)

        for img_path, label in [(g,1) for g in genuines] + [(f,0) for f in forgeds]:
            img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                emb   = model.forward_one(img)
                score = torch.sigmoid(
                    model.classifier(torch.cat([emb, gallery], dim=1))
                ).item()
            pred = 1 if score > 0.5 else 0
            all_preds.append(pred); all_labels.append(label); all_scores.append(score)
            script_results[script].append((pred, label, score))

    return all_preds, all_labels, all_scores, script_results

preds, labels, scores, script_res = evaluate(
    model, all_data, test_ids, transform, DEVICE)

labels_list = list(labels)
acc = accuracy_score(labels_list, preds) * 100
f1  = f1_score(labels_list, preds)
auc = roc_auc_score(labels_list, scores)
gar = sum(1 for p,l in zip(preds,labels_list) if p==1 and l==1) / max(sum(labels_list),1) * 100
far = sum(1 for p,l in zip(preds,labels_list) if p==1 and l==0) / max(labels_list.count(0),1) * 100

print(f"\n{'='*60}")
print(f"  Two-Stage Siamese + STM + Focal Loss — BankSigNet-140")
print(f"{'='*60}")
print(f"  Overall ACC={acc:.1f}% | GAR={gar:.1f}% | FAR={far:.1f}%")
print(f"  F1={f1:.3f} | AUC={auc:.3f}")
print(f"{'='*60}")
for s, name in {'A':'Arabic','E':'English','H':'Hindi'}.items():
    if not script_res[s]: continue
    sp=[x[0] for x in script_res[s]]; sl=[x[1] for x in script_res[s]]
    ss=[x[2] for x in script_res[s]]
    sacc=accuracy_score(sl,sp)*100
    sgar=sum(1 for p,l in zip(sp,sl) if p==1 and l==1)/max(sum(sl),1)*100
    sfar=sum(1 for p,l in zip(sp,sl) if p==1 and l==0)/max(sl.count(0),1)*100
    sauc=roc_auc_score(sl,ss) if len(set(sl))>1 else 0
    print(f"  {name:8s}: ACC={sacc:.1f}% GAR={sgar:.1f}% FAR={sfar:.1f}% AUC={sauc:.3f}")
print(f"{'='*60}")
