"""
MSFV-Net Training Script
========================
Trains MSFV-Net on BankSigNet-140 with forged-aware dual loss,
script-balanced triplet sampling, and synthetic augmentation.

Usage (Google Colab):
    python train.py --data_root /content/extracted \
                    --splits    splits_v12_final.pkl \
                    --save_path verifier_msfv.pth \
                    --seed      42
"""

import os
import pickle
import random
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score

from model.msfvnet import MSFVNet


#  Transforms
def get_train_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomAffine(
                degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.RandomApply(
                [transforms.Lambda(lambda x: x + 0.02 * torch.randn_like(x))], p=0.3
            ),
        ]
    )


def get_val_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


#  Dataset
def scan_dataset(root: str) -> dict:
    """Scan extracted/ folder and return per-signer image lists."""
    root = Path(root)
    data = {}
    for split in ["genuine", "forged"]:
        for person_dir in sorted((root / split).iterdir()):
            if not person_dir.is_dir():
                continue
            pid = person_dir.name
            imgs = sorted(
                list(person_dir.glob("*.png")) + list(person_dir.glob("*.jpg"))
            )
            data.setdefault(pid, {"genuine": [], "forged": [], "script": pid[0]})
            data[pid][split] = imgs
    return data


class SignaturePairDataset(Dataset):
    """
    Builds genuine-genuine (label=0) and genuine-forged (label=1) pairs.
    Script-balanced: equal sampling from A / E / H scripts.
    """

    def __init__(
        self, data: dict, person_ids: list, transform=None, aug_factor: int = 3
    ):
        self.data = data
        self.transform = transform
        self.pairs = self._build_pairs(person_ids, aug_factor)

    def _build_pairs(self, pids, aug_factor):
        pairs = []
        for pid in pids:
            d = self.data.get(pid)
            if not d or not d["genuine"] or not d["forged"]:
                continue
            g = d["genuine"]
            f = d["forged"]
            s = d["script"]
            # Genuine-genuine pairs
            for i in range(len(g)):
                for j in range(i + 1, min(i + 1 + aug_factor, len(g))):
                    pairs.append((g[i], g[j], 0, s))
            # Genuine-forged pairs
            for gi in g:
                for fi in f:
                    pairs.append((gi, fi, 1, s))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label, script = self.pairs[idx]
        img1 = self._load(p1)
        img2 = self._load(p2)
        return img1, img2, torch.tensor(label, dtype=torch.float), script

    def _load(self, path):
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img


def collate_fn(batch):
    imgs1, imgs2, labels, scripts = zip(*batch)
    return (torch.stack(imgs1), torch.stack(imgs2), torch.stack(labels), list(scripts))


#  Loss
class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.5):
        super().__init__()
        self.margin = margin

    def forward(self, e1, e2, labels):
        d = F.pairwise_distance(e1, e2)
        pos = (1 - labels) * d.pow(2)
        neg = labels * F.relu(self.margin - d).pow(2)
        return (pos + neg).mean()


def dual_loss(logits, e1, e2, labels, lambda_t: float = 0.7, lambda_b: float = 1.3):
    """Forged-aware dual loss: λt·triplet + λb·BCE."""
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    triplet = TripletLoss()(e1, e2, labels)
    return lambda_t * triplet + lambda_b * bce


#  Evaluation
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x1, x2, labels, _ in loader:
        x1, x2 = x1.to(device), x2.to(device)
        e1 = model.forward_one(x1)
        e2 = model.forward_one(x2)
        logits = model.forward_pair(e1, e2)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.extend(probs)
        targets.extend(labels.numpy())
    preds = np.array(preds)
    targets = np.array(targets)
    acc = np.mean((preds >= 0.5) == targets) * 100
    auc = roc_auc_score(targets, preds) if len(set(targets)) > 1 else 0.0
    return acc, auc


# Training loop
def train(args):
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {args.seed}")

    # Data
    all_data = scan_dataset(args.data_root)
    with open(args.splits, "rb") as f:
        splits = pickle.load(f)

    valid = set(all_data.keys())
    all_pids = sorted(set(splits["train_gen"].keys()) & valid)
    random.shuffle(all_pids)
    n = len(all_pids)
    train_pids = all_pids[: int(0.70 * n)]
    val_pids = all_pids[int(0.70 * n) : int(0.85 * n)]

    print(f"Train: {len(train_pids)} | Val: {len(val_pids)} persons")

    train_ds = SignaturePairDataset(
        all_data, train_pids, get_train_transform(), aug_factor=3
    )
    val_ds = SignaturePairDataset(all_data, val_pids, get_val_transform(), aug_factor=1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    model = MSFVNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_auc, patience_count = 0.0, 0

    print(f"\n{'='*60}")
    print(f"  Training MSFV-Net on BankSigNet-140")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0

        for x1, x2, labels, _ in train_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            optimizer.zero_grad()
            e1 = model.forward_one(x1)
            e2 = model.forward_one(x2)
            logits = model.forward_pair(e1, e2)
            loss = dual_loss(logits, e1, e2, labels, args.lambda_t, args.lambda_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        val_acc, val_auc = evaluate(model, val_loader, device)
        scheduler.step(val_auc)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Ep {epoch:3d}/{args.epochs} | "
                f"Loss {total_loss/len(train_loader):.4f} | "
                f"Val ACC {val_acc:.2f}% | Val AUC {val_auc:.4f} | "
                f"({time.time()-t0:.1f}s)"
            )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_count = 0
            torch.save(model.state_dict(), args.save_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    print(f"\n  Best Val AUC: {best_auc:.4f}")
    print(f"  Model saved : {args.save_path}")


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MSFV-Net")
    parser.add_argument("--data_root", default="extracted")
    parser.add_argument("--splits", default="splits_v12_final.pkl")
    parser.add_argument("--save_path", default="verifier_msfv.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--lambda_t", type=float, default=0.7)
    parser.add_argument("--lambda_b", type=float, default=1.3)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    train(args)
