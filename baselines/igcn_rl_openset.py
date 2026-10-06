# ================================================================
#  Incremental GCN + RL Attention — Signature Verification
#  Dataset  : BankSigNet-140 (140 persons: EN-46, HI-45, AR-49)
#  Protocol : Open-set person-level split (70/15/15)
#             Train persons are NEVER seen during test — same
#             protocol as Priya et al. TBIOM 2025 and MSFV-Net
#  Platform : Google Colab + Google Drive
# ================================================================

# ── CELL 1: Mount ─────────────────────────────────────────────

# ── CELL 2: Dependencies ──────────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

install("torch-geometric")
install("scikit-image")
print("Dependencies ready ✓")

# ── CELL 3: Imports ───────────────────────────────────────────
import os, cv2, random, time, warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (roc_auc_score, roc_curve,
                              confusion_matrix, accuracy_score)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv

from skimage.morphology import skeletonize
from skimage.feature import corner_harris, corner_peaks

warnings.filterwarnings("ignore")
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")

# ── CELL 4: Config ────────────────────────────────────────────
class Config:
    DATA_ROOT   = "extracted"
    SPLITS_PATH = "splits_v12_final.pkl"

    # Image
    IMG_H, IMG_W = 128, 256

    # Graph
    MAX_NODES   = 64
    EDGE_THRESH = 22
    NODE_DIM    = 7

    # Model
    GCN_HIDDEN  = 128
    GCN_LAYERS  = 3
    EMBED_DIM   = 256
    ATTN_DIM    = 64
    DROPOUT     = 0.3

    # Training
    BATCH_SIZE  = 16
    LR          = 1e-3
    WEIGHT_DECAY= 1e-4
    EPOCHS      = 60
    MARGIN      = 1.0
    SEED        = 42

    # Open-set split ratios (person-level)
    TRAIN_RATIO = 0.70
    VAL_RATIO   = 0.15
    # TEST_RATIO  = 0.15 (remainder)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
random.seed(cfg.SEED)
print(f"Device   : {cfg.DEVICE}")
print(f"Data root: {cfg.DATA_ROOT}")

# ── CELL 5: Dataset Scanner ───────────────────────────────────
def scan_dataset(root: str) -> dict:
    root        = Path(root)
    genuine_dir = root / "genuine"
    forged_dir  = root / "forged"
    sigs = {}

    for person_folder in sorted(genuine_dir.iterdir()):
        if not person_folder.is_dir(): continue
        key    = person_folder.name
        script = key[0]
        imgs   = sorted(person_folder.glob("*.png"))
        if not imgs: continue
        sigs.setdefault(key, {'script': script, 'genuine': [], 'forged': []})
        sigs[key]['genuine'] = imgs

    for person_folder in sorted(forged_dir.iterdir()):
        if not person_folder.is_dir(): continue
        key  = person_folder.name
        imgs = sorted(person_folder.glob("*.png"))
        if not imgs: continue
        if key not in sigs:
            sigs[key] = {'script': key[0], 'genuine': [], 'forged': []}
        sigs[key]['forged'] = imgs

    return sigs


def print_split_summary(label, pids, sigs):
    by_script = {'A': 0, 'E': 0, 'H': 0}
    for pid in pids:
        s = sigs[pid]['script']
        if s in by_script:
            by_script[s] += 1
    total = sum(by_script.values())
    print(f"  {label:6s}: {total:3d} persons  "
          f"(AR={by_script['A']} EN={by_script['E']} HI={by_script['H']})")


# ── CELL 6: Open-set Person-level Split ───────────────────────
import pickle

sigs     = scan_dataset(cfg.DATA_ROOT)
all_pids = sorted(sigs.keys())

# Use same splits as MSFV-Net if available
splits_path = Path(cfg.SPLITS_PATH)
if splits_path.exists():
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    train_pids = splits.get('train', [])
    val_pids   = splits.get('val',   [])
    test_pids  = splits.get('test',  [])
    print("✅ Loaded existing splits from splits_v10_cleaned.pkl")
else:
    random.seed(cfg.SEED)
    pids_shuffled = all_pids.copy()
    random.shuffle(pids_shuffled)
    n        = len(pids_shuffled)
    n_train  = int(cfg.TRAIN_RATIO * n)
    n_val    = int(cfg.VAL_RATIO   * n)
    train_pids = pids_shuffled[:n_train]
    val_pids   = pids_shuffled[n_train:n_train + n_val]
    test_pids  = pids_shuffled[n_train + n_val:]
    print("⚠️  splits_v10_cleaned.pkl not found — using random 70/15/15 split")

print(f"\nOpen-set person-level split:")
print_split_summary("Train", train_pids, sigs)
print_split_summary("Val",   val_pids,   sigs)
print_split_summary("Test",  test_pids,  sigs)
print(f"  Total : {len(train_pids)+len(val_pids)+len(test_pids)} persons\n")

# Sanity check — no overlap
assert not set(train_pids) & set(test_pids),  "Train/test overlap!"
assert not set(val_pids)   & set(test_pids),  "Val/test overlap!"
assert not set(train_pids) & set(val_pids),   "Train/val overlap!"
print("✅ No person overlap between splits")

# ── CELL 7: Graph Construction ────────────────────────────────
def preprocess(img):
    _, binary = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def extract_keypoints(binary, max_nodes):
    skel    = skeletonize(binary > 0).astype(np.uint8)
    corners = corner_peaks(
        corner_harris(skel), min_distance=8,
        threshold_rel=0.02, num_peaks=max_nodes // 2)
    skel_pts = np.column_stack(np.where(skel > 0))
    if len(skel_pts) > max_nodes // 2:
        idx      = np.linspace(0, len(skel_pts)-1,
                               max_nodes//2, dtype=int)
        skel_pts = skel_pts[idx]
    pts = np.vstack([corners, skel_pts]) if len(corners) > 0 else skel_pts
    if len(pts) > max_nodes:
        idx = np.random.choice(len(pts), max_nodes, replace=False)
        pts = pts[idx]
    if len(pts) < 4:
        pad = np.zeros((4-len(pts), 2), dtype=np.int64)
        pts = np.vstack([pts, pad])
    return pts.astype(np.float32)


def node_features(pts, binary, gray):
    h, w  = gray.shape
    gx    = cv2.Sobel(gray,   cv2.CV_32F, 1, 0, ksize=3)
    gy    = cv2.Sobel(gray,   cv2.CV_32F, 0, 1, ksize=3)
    dist  = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    feats = []
    for r, c in pts:
        r, c    = int(np.clip(r, 0, h-1)), int(np.clip(c, 0, w-1))
        patch   = binary[max(0,r-4):r+5, max(0,c-4):c+5]
        density = patch.sum() / (81*255 + 1e-8)
        gxv     = gx[r, c] / 255.0
        gyv     = gy[r, c] / 255.0
        angle   = np.arctan2(gyv, gxv) / np.pi
        sw      = dist[r, c] / (max(h, w)/2 + 1e-8)
        feats.append([c/w, r/h, gxv, gyv, density, angle, sw])
    return np.array(feats, dtype=np.float32)


def build_edges(pts, thresh):
    n = len(pts)
    rows, cols = [], []
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i]-pts[j]) < thresh:
                rows += [i, j]; cols += [j, i]
    if not rows:
        for i in range(n-1):
            rows += [i, i+1]; cols += [i+1, i]
    return np.array([rows, cols], dtype=np.int64)


def image_to_graph(path):
    img    = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((cfg.IMG_H, cfg.IMG_W), np.uint8)
    gray   = cv2.resize(img, (cfg.IMG_W, cfg.IMG_H))
    binary = preprocess(gray)
    pts    = extract_keypoints(binary, cfg.MAX_NODES)
    x      = node_features(pts, binary, gray)
    ei     = build_edges(pts, cfg.EDGE_THRESH)
    return Data(x=torch.tensor(x,  dtype=torch.float),
                edge_index=torch.tensor(ei, dtype=torch.long))

# ── CELL 8: Dataset — person-aware pair builder ───────────────
class BankSigNetOpenSet(Dataset):
    """
    Open-set pair dataset.
    Each instance receives a fixed list of person IDs.
    Test persons are never in train or val.

    Label convention:
      0 = genuine-genuine (same person)
      1 = genuine-forged  (impostor)
    """

    def __init__(self, sigs: dict, person_ids: list,
                 tag: str = 'train'):
        self.sigs  = sigs
        self.pairs = self._build_pairs(person_ids)
        sc = {'A': 0, 'E': 0, 'H': 0}
        for p in self.pairs:
            sc[p[3]] = sc.get(p[3], 0) + 1
        print(f"  [{tag:5s}] {len(self.pairs):5d} pairs  "
              f"AR={sc['A']} EN={sc['E']} HI={sc['H']}")

    def _build_pairs(self, pids):
        pairs = []
        for pid in pids:
            d = self.sigs.get(pid)
            if not d: continue
            g = d['genuine']
            f = d['forged']
            s = d['script']
            # GG pairs — up to 5 per person
            for i in range(len(g)):
                for j in range(i+1, min(i+6, len(g))):
                    pairs.append((g[i], g[j], 0, s))
            # GF pairs — first 8 genuine × 8 forged
            for gi in g[:8]:
                for fi in f[:8]:
                    pairs.append((gi, fi, 1, s))
        return pairs

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label, script = self.pairs[idx]
        return (image_to_graph(p1),
                image_to_graph(p2),
                torch.tensor(label, dtype=torch.float),
                script)


def collate_fn(batch):
    g1s, g2s, labels, scripts = zip(*batch)
    return (Batch.from_data_list(list(g1s)),
            Batch.from_data_list(list(g2s)),
            torch.stack(labels),
            list(scripts))


print("\nBuilding open-set pairs …")
train_ds = BankSigNetOpenSet(sigs, train_pids, tag='train')
val_ds   = BankSigNetOpenSet(sigs, val_pids,   tag='val  ')
test_ds  = BankSigNetOpenSet(sigs, test_pids,  tag='test ')

train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                          shuffle=True,  collate_fn=collate_fn,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                          shuffle=False, collate_fn=collate_fn,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                          shuffle=False, collate_fn=collate_fn,
                          num_workers=2, pin_memory=True)

# ── CELL 9: Model ─────────────────────────────────────────────
class RLAttention(nn.Module):
    def __init__(self, feat_dim, attn_dim):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(feat_dim, attn_dim), nn.Tanh(),
            nn.Linear(attn_dim, 1))
        self.critic = nn.Sequential(
            nn.Linear(feat_dim, attn_dim), nn.ReLU(),
            nn.Linear(attn_dim, 1))

    def forward(self, x, batch):
        logits = self.policy(x).squeeze(-1)
        scores = self._scatter_softmax(logits, batch)
        weighted = x * scores.unsqueeze(-1)
        B   = batch.max().item() + 1
        out = torch.zeros(B, x.size(1), device=x.device)
        out.scatter_add_(0,
                         batch.unsqueeze(-1).expand_as(weighted),
                         weighted)
        values = self.critic(x).squeeze(-1)
        return out, scores, values

    @staticmethod
    def _scatter_softmax(logits, batch):
        B     = batch.max().item() + 1
        max_v = torch.zeros(B, device=logits.device)
        max_v.scatter_reduce_(0, batch, logits,
                              reduce='amax', include_self=True)
        exp   = torch.exp(logits - max_v[batch])
        denom = torch.zeros(B, device=logits.device)
        denom.scatter_add_(0, batch, exp)
        return exp / (denom[batch] + 1e-8)


class GCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.bn   = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout)
        self.res  = (nn.Linear(in_dim, out_dim, bias=False)
                     if in_dim != out_dim else nn.Identity())

    def forward(self, x, edge_index):
        h = F.relu(self.bn(self.conv(x, edge_index)))
        return self.drop(h) + self.res(x)


class IncrementalGCN(nn.Module):
    def __init__(self, in_dim, hidden, embed, L=3, dropout=0.3):
        super().__init__()
        self.proj_in   = nn.Linear(in_dim, hidden)
        self.layers    = nn.ModuleList(
            [GCNBlock(hidden, hidden, dropout) for _ in range(L)])
        self.out_projs = nn.ModuleList(
            [nn.Linear(hidden, embed) for _ in range(L)])
        self.embed = embed
        self.L     = L

    def forward(self, x, edge_index):
        h   = F.relu(self.proj_in(x))
        acc = torch.zeros(h.size(0), self.embed, device=x.device)
        for i, (blk, proj) in enumerate(zip(self.layers, self.out_projs)):
            h   = blk(h, edge_index)
            acc = acc + (i+1)/self.L * proj(h)
        return acc


class SignatureGCNVerifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gcn  = IncrementalGCN(
            cfg.NODE_DIM, cfg.GCN_HIDDEN,
            cfg.EMBED_DIM, cfg.GCN_LAYERS, cfg.DROPOUT)
        self.attn = RLAttention(cfg.EMBED_DIM, cfg.ATTN_DIM)
        self.head = nn.Sequential(
            nn.Linear(cfg.EMBED_DIM * 4, 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def encode(self, g):
        nf            = self.gcn(g.x, g.edge_index)
        emb, sc, vals = self.attn(nf, g.batch)
        return emb, sc, vals

    def forward(self, g1, g2):
        e1, s1, v1 = self.encode(g1)
        e2, s2, v2 = self.encode(g2)
        fused  = torch.cat([e1, e2, (e1-e2).abs(), e1*e2], dim=-1)
        logits = self.head(fused).squeeze(-1)
        return logits, {'e1': e1, 'e2': e2, 's1': s1, 's2': s2}

    @torch.no_grad()
    def get_embedding(self, g):
        e, _, _ = self.encode(g)
        return e


model    = SignatureGCNVerifier(cfg).to(cfg.DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {n_params:,}")

# ── CELL 10: Loss ─────────────────────────────────────────────
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, e1, e2, labels):
        d   = F.pairwise_distance(e1, e2)
        pos = (1-labels) * d.pow(2)
        neg = labels * F.relu(self.margin - d).pow(2)
        return (pos + neg).mean()


def entropy_bonus(scores, coef=0.01):
    return -coef * (-(scores * (scores+1e-8).log())).mean()


def compute_loss(logits, info, labels):
    bce    = F.binary_cross_entropy_with_logits(logits, labels)
    contra = ContrastiveLoss(cfg.MARGIN)(info['e1'], info['e2'], labels)
    ent    = entropy_bonus(info['s1']) + entropy_bonus(info['s2'])
    return bce + 0.3*contra + ent, bce.item(), contra.item()

# ── CELL 11: Train/Eval helpers ───────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, preds, targets = 0.0, [], []
    for g1, g2, labels, _ in tqdm(loader, desc="  train", leave=False):
        g1, g2, labels = (g1.to(cfg.DEVICE), g2.to(cfg.DEVICE),
                          labels.to(cfg.DEVICE))
        optimizer.zero_grad()
        logits, info = model(g1, g2)
        loss, _, _   = compute_loss(logits, info, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        targets.extend(labels.cpu().numpy())
    auc = roc_auc_score(targets, preds) if len(set(targets)) > 1 else 0.0
    return total_loss / len(loader), auc


@torch.no_grad()
def evaluate(model, loader, split_name='val'):
    model.eval()
    preds, targets = [], []
    per_script = {'A': ([], []), 'E': ([], []), 'H': ([], [])}

    for g1, g2, labels, scripts in loader:
        g1, g2    = g1.to(cfg.DEVICE), g2.to(cfg.DEVICE)
        logits, _ = model(g1, g2)
        probs     = torch.sigmoid(logits).cpu().numpy()
        lbls      = labels.numpy()
        preds.extend(probs)
        targets.extend(lbls)
        for p, l, s in zip(probs, lbls, scripts):
            if s in per_script:
                per_script[s][0].append(p)
                per_script[s][1].append(l)

    preds   = np.array(preds)
    targets = np.array(targets, dtype=int)

    auc          = roc_auc_score(targets, preds)
    fpr, tpr, th = roc_curve(targets, preds)
    fnr          = 1 - tpr
    eer_idx      = np.argmin(np.abs(fnr - fpr))
    eer          = float((fpr[eer_idx] + fnr[eer_idx]) / 2) * 100
    best_acc     = max(np.mean((preds >= t) == targets) for t in th) * 100
    pred_bin     = (preds >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, pred_bin, labels=[0,1]).ravel()
    far  = fp / (fp+tn+1e-8) * 100
    frr  = fn / (fn+tp+1e-8) * 100

    script_auc = {}
    for s, (sp, sl) in per_script.items():
        if len(set(sl)) > 1:
            script_auc[s] = roc_auc_score(sl, sp)

    return {
        'auc': auc, 'eer': eer, 'accuracy': best_acc,
        'far': far, 'frr': frr,
        'fpr': fpr, 'tpr': tpr,
        'script_auc': script_auc,
    }

# ── CELL 12: Training Loop ────────────────────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.EPOCHS, eta_min=1e-5)

history          = {'train_loss': [], 'train_auc': [],
                    'val_auc': [],    'val_eer': []}
best_val_auc     = 0.0
best_val_metrics = {}
SAVE_PATH        = "best_gcn_rl_openset.pth"

print("\n" + "="*65)
print("  IGCN+RL — BankSigNet-140 — Open-set Person-level Split")
print("="*65)

for epoch in range(1, cfg.EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_auc = train_epoch(model, train_loader, optimizer)
    val_m           = evaluate(model, val_loader, 'val')
    scheduler.step()

    history['train_loss'].append(tr_loss)
    history['train_auc'].append(tr_auc)
    history['val_auc'].append(val_m['auc'])
    history['val_eer'].append(val_m['eer'])

    if val_m['auc'] > best_val_auc:
        best_val_auc     = val_m['auc']
        best_val_metrics = {k: v for k, v in val_m.items()
                            if k not in ('fpr', 'tpr')}
        torch.save(model.state_dict(), SAVE_PATH)

    if epoch % 5 == 0 or epoch == 1:
        sc_str = "  ".join(
            f"{s}:{v:.3f}" for s, v in val_m['script_auc'].items())
        print(f"  Ep {epoch:3d}/{cfg.EPOCHS} │ "
              f"Loss {tr_loss:.4f} │ "
              f"Tr-AUC {tr_auc:.4f} │ "
              f"Val-AUC {val_m['auc']:.4f} │ "
              f"EER {val_m['eer']:.2f}% │ "
              f"FAR {val_m['far']:.2f}% │ "
              f"FRR {val_m['frr']:.2f}%  "
              f"[{sc_str}]  ({time.time()-t0:.1f}s)")

print(f"\n  Best Val AUC : {best_val_auc:.4f}")
print(f"  Model saved  : {SAVE_PATH}")

# ── CELL 13: Final Test Evaluation (unseen persons) ───────────
print("\n" + "="*65)
print("  FINAL TEST EVALUATION — Unseen Persons")
print("="*65)

# Load best model
model.load_state_dict(torch.load(SAVE_PATH, map_location=cfg.DEVICE))
test_m = evaluate(model, test_loader, 'test')

print(f"\n  Test ACC  : {test_m['accuracy']:.2f}%")
print(f"  Test AUC  : {test_m['auc']:.4f}")
print(f"  Test EER  : {test_m['eer']:.2f}%")
print(f"  Test FAR  : {test_m['far']:.2f}%")
print(f"  Test FRR  : {test_m['frr']:.2f}%")
print()
label_map = {'A': 'Arabic', 'E': 'English', 'H': 'Hindi'}
for s, name in label_map.items():
    auc_s = test_m['script_auc'].get(s, 0)
    print(f"  {name:8s} AUC : {auc_s:.4f}")

# ── CELL 14: Comparison Table ─────────────────────────────────
# MSFV-Net results for reference (update if you have new numbers)
msfvnet = {
    'accuracy': 80.8, 'auc': None,
    'eer':  None,     'far': None, 'frr': None,
    'script_auc': {'A': 0.721, 'E': 0.724, 'H': 0.922}
}

SEP = "─" * 68
print(f"\n{SEP}")
print(f"  {'Metric':<22}  {'IGCN+RL [Priya2025]':>22}  {'MSFV-Net (proposed)':>20}")
print(f"  {'Protocol':<22}  {'Open-set person-split':>22}  {'Open-set person-split':>20}")
print(SEP)

rows = [
    ('Accuracy (%)',  'accuracy', '{:.2f}', True),
    ('AUC-ROC',       'auc',      '{:.4f}', True),
    ('EER (%)',        'eer',      '{:.2f}', False),
    ('FAR (%)',        'far',      '{:.2f}', False),
    ('FRR (%)',        'frr',      '{:.2f}', False),
]
for name, key, fmt, higher in rows:
    gv = test_m.get(key, 0) or 0
    mv = msfvnet.get(key, None)
    gs = fmt.format(gv) if gv else '   —'
    ms = fmt.format(mv) if mv else '   —'
    print(f"  {name:<22}  {gs:>22}  {ms:>20}")

print(SEP)
print(f"  {'Script AUC':<22}")
for s, name in label_map.items():
    gv = test_m['script_auc'].get(s, 0)
    mv = msfvnet['script_auc'].get(s, 0)
    print(f"  {name:8s} AUC{' ':>11}  {gv:.4f}{' ':>18}  {mv:.4f}")
print(SEP)
print("  Note: Both protocols use open-set person-level split.")
print("        Test persons are entirely unseen during training.")
print(SEP)

# ── CELL 15: Training Curves ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(history['train_loss'], color='steelblue', lw=1.8)
axes[0].set_title('Training Loss'); axes[0].set_xlabel('Epoch')
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_auc'], color='steelblue', lw=1.8, label='Train')
axes[1].plot(history['val_auc'],   color='darkorange', lw=1.8, label='Val')
axes[1].set_title('AUC-ROC'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(history['val_eer'], color='crimson', lw=1.8)
axes[2].set_title('Validation EER (%)'); axes[2].set_xlabel('Epoch')
axes[2].grid(True, alpha=0.3)

plt.suptitle('IGCN+RL — BankSigNet-140 — Open-set Protocol', y=1.02)
plt.tight_layout()
plt.savefig('gcn_rl_openset_curves.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Curves saved")
