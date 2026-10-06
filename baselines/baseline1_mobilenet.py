"""
Baseline 1: MobileNetV2 Siamese Network
========================================
Run on Google Colab with GPU runtime.
Uses same train/val/test split as MSFV-Net for fair comparison.

Output: prints ACC, GAR, FAR, F1, AUC per script and overall
"""



import os, random, pickle, numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from itertools import combinations
from PIL import Image

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = Path("extracted")
SPLITS_PATH  = Path("splits_v12_final.pkl")
SEED         = 42
EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 1e-4
PATIENCE     = 10

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
print(f"✅ Device: {DEVICE}")

#  CELL 2: Dataset 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def get_script(person_id):
    return person_id[0].upper()  # A, E, or H

def load_all_images(root):
    """Returns dict: person_id -> {genuine: [paths], forged: [paths]}"""
    data = {}
    for split in ['genuine', 'forged']:
        split_dir = root / split
        for person_dir in sorted(split_dir.iterdir()):
            if not person_dir.is_dir(): continue
            pid = person_dir.name
            imgs = sorted(list(person_dir.glob("*.png")) +
                         list(person_dir.glob("*.jpg")))
            if pid not in data:
                data[pid] = {'genuine': [], 'forged': []}
            data[pid][split] = imgs
    return data

class PairDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1 = self.transform(Image.open(p1).convert("RGB"))
        img2 = self.transform(Image.open(p2).convert("RGB"))
        return img1, img2, torch.tensor(label, dtype=torch.float32)

def make_pairs(data, person_ids, n_pairs=8000):
    pairs = []
    pids  = [p for p in person_ids if data[p]['genuine'] and data[p]['forged']]

    # Genuine pairs (same person, both genuine)
    for pid in pids:
        imgs = data[pid]['genuine']
        for i, j in combinations(range(len(imgs)), 2):
            pairs.append((imgs[i], imgs[j], 1))

    # Forged pairs (genuine vs forged, same person)
    for pid in pids:
        for g in data[pid]['genuine']:
            for f in data[pid]['forged']:
                pairs.append((g, f, 0))

    random.shuffle(pairs)
    return pairs[:n_pairs]

# Load data
all_data = load_all_images(DATASET_ROOT)
all_pids = sorted(all_data.keys())

# Use same 70/15/15 split
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

train_pairs = make_pairs(all_data, train_ids, n_pairs=8000)
val_pairs   = make_pairs(all_data, val_ids,   n_pairs=1500)

train_loader = DataLoader(PairDataset(train_pairs, transform),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(PairDataset(val_pairs,   transform),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

#  CELL 3: MobileNetV2 Siamese Model ─
class MobileNetSiamese(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.encoder = backbone.features
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.embed   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward_one(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        return nn.functional.normalize(self.embed(x), dim=1)

    def forward(self, x1, x2):
        e1 = self.forward_one(x1)
        e2 = self.forward_one(x2)
        return self.classifier(torch.cat([e1, e2], dim=1)).squeeze(1)

model     = MobileNetSiamese().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

#  CELL 4: Training ─
best_val_acc = 0
patience_cnt = 0
best_state   = None

for epoch in range(1, EPOCHS + 1):
    # Train
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

    # Validate
    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0
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

#  CELL 5: Evaluation ─
model.load_state_dict(best_state)
model.eval()

def evaluate(model, data, person_ids, transform, device):
    all_preds, all_labels, all_scores = [], [], []
    script_results = {'A': [], 'E': [], 'H': []}

    for pid in person_ids:
        if not data[pid]['genuine'] or not data[pid]['forged']:
            continue
        script = get_script(pid)
        genuines = data[pid]['genuine']
        forgeds  = data[pid]['forged']

        # Gallery: mean of all genuine embeddings
        gallery_embs = []
        with torch.no_grad():
            for gpath in genuines:
                img = transform(Image.open(gpath).convert("RGB")).unsqueeze(0).to(device)
                emb = model.forward_one(img)
                gallery_embs.append(emb)
        gallery = torch.stack(gallery_embs).mean(0)

        # Test: remaining genuine + all forged
        test_pairs = [(g, 1) for g in genuines] + [(f, 0) for f in forgeds]

        for img_path, label in test_pairs:
            img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                emb   = model.forward_one(img)
                score = torch.sigmoid(
                    model.classifier(torch.cat([emb, gallery], dim=1))
                ).item()
            pred = 1 if score > 0.5 else 0
            all_preds.append(pred)
            all_labels.append(label)
            all_scores.append(score)
            script_results[script].append((pred, label, score))

    return all_preds, all_labels, all_scores, script_results

preds, labels, scores, script_res = evaluate(
    model, all_data, test_ids, transform, DEVICE)

# Overall metrics
acc = accuracy_score(labels, preds) * 100
f1  = f1_score(labels, preds)
auc = roc_auc_score(labels, scores)
gar = sum(1 for p,l in zip(preds,labels) if p==1 and l==1) / max(sum(labels),1) * 100
far = sum(1 for p,l in zip(preds,labels) if p==1 and l==0) / max(labels.count(0),1) * 100

print(f"\n{'='*55}")
print(f"  MobileNetV2 Siamese — BankSigNet-140 Results")
print(f"{'='*55}")
print(f"  Overall ACC = {acc:.1f}% | GAR = {gar:.1f}% | FAR = {far:.1f}%")
print(f"  F1 = {f1:.3f} | AUC = {auc:.3f}")
print(f"{'='*55}")

script_names = {'A': 'Arabic', 'E': 'English', 'H': 'Hindi'}
for s, name in script_names.items():
    if not script_res[s]: continue
    sp = [x[0] for x in script_res[s]]
    sl = [x[1] for x in script_res[s]]
    ss = [x[2] for x in script_res[s]]
    sacc = accuracy_score(sl, sp) * 100
    sgar = sum(1 for p,l in zip(sp,sl) if p==1 and l==1) / max(sum(sl),1) * 100
    sfar = sum(1 for p,l in zip(sp,sl) if p==1 and l==0) / max(sl.count(0),1) * 100
    sauc = roc_auc_score(sl, ss) if len(set(sl)) > 1 else 0
    print(f"  {name:8s}: ACC={sacc:.1f}% GAR={sgar:.1f}% FAR={sfar:.1f}% AUC={sauc:.3f}")

print(f"{'='*55}")
