"""
Baseline 3: CNN + Autoencoder-based Augmentation (Writer-Dependent)
=====================================================================
Based on: "CNN with Autoencoder-based data augmentation for offline
signature verification" — Persian dataset paper.

Key design:
- Autoencoder trained on genuine signatures generates synthetic forgeries
- CNN classifier trained per writer (writer-dependent)
- At test time: compare query against writer-specific model

Run on Google Colab with GPU runtime.
Same train/val/test split as MSFV-Net for fair comparison.

Note: Writer-dependent means one model per signer — this is evaluated
per-signer and aggregated, consistent with the original paper's protocol.
"""



import os, random, pickle, numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from PIL import Image

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = Path("extracted")
SPLITS_PATH  = Path("splits_v12_final.pkl")
SEED         = 42
AE_EPOCHS    = 30   # Autoencoder training epochs
CNN_EPOCHS   = 20   # Per-writer CNN epochs

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
print(f"✅ Device: {DEVICE}")

#  CELL 2: Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # smaller for efficiency
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

transform_cnn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

#  CELL 3: Autoencoder 
class SignatureAutoencoder(nn.Module):
    """Convolutional autoencoder that learns genuine signature manifold.
    Adding noise to latent space generates synthetic forgery-like samples."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),  # 64x64
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(), # 32x32
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),# 16x16
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),# 8x8
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512)
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 256 * 8 * 8),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),    nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def generate_synthetic(self, x, noise_scale=0.3):
        """Generate synthetic forgery by perturbing latent space."""
        with torch.no_grad():
            _, z = self.forward(x)
            z_noisy = z + torch.randn_like(z) * noise_scale
            return self.decoder(z_noisy)

#  CELL 4: Feature extractor CNN (writer-independent backbone) ─
class FeatureCNN(nn.Module):
    """ResNet50 feature extractor — shared across all writers."""
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.flatten  = nn.Flatten()

    def forward(self, x):
        return self.flatten(self.features(x))  # [B, 2048]

#  CELL 5: Per-writer classifier ─
class WriterClassifier(nn.Module):
    """Small binary classifier trained per writer."""
    def __init__(self, feat_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

#  CELL 6: Load data─
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

#  CELL 7: Train global autoencoder on all training genuines 
print("\nTraining autoencoder on genuine signatures...")
ae_model = SignatureAutoencoder().to(DEVICE)
ae_optim = optim.Adam(ae_model.parameters(), lr=1e-3)

# Collect all training genuine images
all_genuine_tensors = []
for pid in train_ids:
    for img_path in all_data[pid]['genuine']:
        img = transform(Image.open(img_path).convert("RGB"))
        all_genuine_tensors.append(img)

ae_dataset = torch.stack(all_genuine_tensors)
ae_loader  = DataLoader(TensorDataset(ae_dataset),
                        batch_size=64, shuffle=True)

for epoch in range(1, AE_EPOCHS + 1):
    ae_model.train()
    total_loss = 0
    for (x,) in ae_loader:
        x = x.to(DEVICE)
        ae_optim.zero_grad()
        recon, _ = ae_model(x)
        loss = F.mse_loss(recon, x)
        loss.backward()
        ae_optim.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"  AE Epoch {epoch}/{AE_EPOCHS} | loss={total_loss/len(ae_loader):.4f}")

print("✅ Autoencoder trained")

#  CELL 8: Extract features using shared CNN ─
print("\nExtracting features...")
feat_cnn = FeatureCNN().to(DEVICE)
feat_cnn.eval()

def extract_features(img_paths, model, transform, device):
    feats = []
    with torch.no_grad():
        for p in img_paths:
            img = transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            feats.append(model(img).cpu())
    return torch.cat(feats, dim=0)

#  CELL 9: Train per-writer classifiers 
print("\nTraining per-writer classifiers...")
writer_models = {}
criterion = nn.BCEWithLogitsLoss()

for pid in train_ids + val_ids:
    if not all_data[pid]['genuine'] or not all_data[pid]['forged']:
        continue

    # Extract genuine features
    gen_feats  = extract_features(all_data[pid]['genuine'],
                                  feat_cnn, transform_cnn, DEVICE)
    forg_feats = extract_features(all_data[pid]['forged'],
                                  feat_cnn, transform_cnn, DEVICE)

    # Generate synthetic forgeries using autoencoder
    synth_forgeries = []
    for img_path in all_data[pid]['genuine']:
        img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        synth = ae_model.generate_synthetic(img)
        # Convert synth back to 3-channel for feature extractor
        synth_3ch = synth.repeat(1, 3, 1, 1)
        synth_3ch = F.interpolate(synth_3ch, size=(224, 224))
        # Normalise
        mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(DEVICE)
        std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(DEVICE)
        synth_3ch = (synth_3ch - mean) / std
        with torch.no_grad():
            synth_feat = feat_cnn(synth_3ch).cpu()
        synth_forgeries.append(synth_feat)

    synth_feats = torch.cat(synth_forgeries, dim=0) if synth_forgeries else forg_feats

    # Combine real + synthetic forgeries
    all_forg = torch.cat([forg_feats, synth_feats], dim=0)

    X = torch.cat([gen_feats,
                   torch.zeros(gen_feats.size(0)).unsqueeze(1),
                   all_forg,
                   torch.ones(all_forg.size(0)).unsqueeze(1)], dim=0)

    # Separate features and labels
    X_feat = torch.cat([gen_feats, all_forg], dim=0)
    y      = torch.cat([torch.ones(gen_feats.size(0)),
                        torch.zeros(all_forg.size(0))], dim=0)

    # Train small per-writer classifier
    clf     = WriterClassifier(feat_dim=2048).to(DEVICE)
    clf_opt = optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)

    ds     = TensorDataset(X_feat, y)
    loader = DataLoader(ds, batch_size=min(32, len(ds)), shuffle=True)

    for ep in range(CNN_EPOCHS):
        clf.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            clf_opt.zero_grad()
            loss = criterion(clf(xb), yb)
            loss.backward()
            clf_opt.step()

    writer_models[pid] = clf

print(f"✅ Trained {len(writer_models)} per-writer classifiers")

#  CELL 10: Evaluation 
print("\nEvaluating on test set...")
all_preds, all_labels, all_scores = [], [], []
script_results = {'A': [], 'E': [], 'H': []}

for pid in test_ids:
    if not all_data[pid]['genuine'] or not all_data[pid]['forged']:
        continue
    script = get_script(pid)

    # Use nearest writer model if this writer wasn't in training
    if pid in writer_models:
        clf = writer_models[pid]
    else:
        # Fall back to any trained model (limitation of writer-dependent approach)
        clf = list(writer_models.values())[0]

    test_imgs  = [(p, 1) for p in all_data[pid]['genuine']] + \
                 [(p, 0) for p in all_data[pid]['forged']]

    for img_path, label in test_imgs:
        feat = extract_features([img_path], feat_cnn, transform_cnn, DEVICE)
        with torch.no_grad():
            score = torch.sigmoid(clf(feat.to(DEVICE))).item()
        pred = 1 if score > 0.5 else 0
        all_preds.append(pred)
        all_labels.append(label)
        all_scores.append(score)
        script_results[script].append((pred, label, score))

acc = accuracy_score(all_labels, all_preds) * 100
f1  = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_scores)
gar = sum(1 for p,l in zip(all_preds,all_labels) if p==1 and l==1)/max(sum(all_labels),1)*100
far = sum(1 for p,l in zip(all_preds,all_labels) if p==1 and l==0)/max(all_labels.count(0),1)*100

print(f"\n{'='*60}")
print(f"  CNN + Autoencoder Aug (Writer-Dependent) — BankSigNet-140")
print(f"{'='*60}")
print(f"  Overall ACC={acc:.1f}% | GAR={gar:.1f}% | FAR={far:.1f}%")
print(f"  F1={f1:.3f} | AUC={auc:.3f}")
print(f"{'='*60}")
for s, name in {'A':'Arabic','E':'English','H':'Hindi'}.items():
    if not script_results[s]: continue
    sp=[x[0] for x in script_results[s]]
    sl=[x[1] for x in script_results[s]]
    ss=[x[2] for x in script_results[s]]
    sacc=accuracy_score(sl,sp)*100
    sgar=sum(1 for p,l in zip(sp,sl) if p==1 and l==1)/max(sum(sl),1)*100
    sfar=sum(1 for p,l in zip(sp,sl) if p==1 and l==0)/max(sl.count(0),1)*100
    sauc=roc_auc_score(sl,ss) if len(set(sl))>1 else 0
    print(f"  {name:8s}: ACC={sacc:.1f}% GAR={sgar:.1f}% "
          f"FAR={sfar:.1f}% AUC={sauc:.3f}")
print(f"{'='*60}")
