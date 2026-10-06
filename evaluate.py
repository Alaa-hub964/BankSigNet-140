"""
MSFV-Net Evaluation Script
===========================
Evaluates a trained MSFV-Net checkpoint on BankSigNet-140
under open-set signer-disjoint protocol.

Reports: ACC, EER, FAR, FRR, AUC (overall and per-script).

Usage:
    python evaluate.py --checkpoint verifier_msfv.pth \
                       --data_root  extracted \
                       --splits     splits_v12_final.pkl \
                       --seeds      42 123 456
"""

import os
import pickle
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix

from model.msfvnet import MSFVNet


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


def scan_dataset(root: str) -> dict:
    root = Path(root)
    data = {}
    for split in ['genuine', 'forged']:
        for person_dir in sorted((root / split).iterdir()):
            if not person_dir.is_dir():
                continue
            pid  = person_dir.name
            imgs = sorted(list(person_dir.glob('*.png')) +
                          list(person_dir.glob('*.jpg')))
            data.setdefault(pid, {'genuine': [], 'forged': [], 'script': pid[0]})
            data[pid][split] = imgs
    return data


@torch.no_grad()
def evaluate_persons(model, person_ids, all_data, device, transform):
    """Evaluate model on a list of test person IDs."""
    scores, labels = [], []
    script_res = {'A': ([], []), 'E': ([], []), 'H': ([], [])}

    for pid in person_ids:
        d = all_data.get(pid)
        if not d or not d['genuine'] or not d['forged']:
            continue
        script = d['script']

        # Build gallery from all genuine references
        gallery = torch.stack([
            model.forward_one(
                transform(Image.open(g).convert('L')).unsqueeze(0).to(device))
            for g in d['genuine']
        ]).mean(0)

        # Score all images
        for path, lbl in ([(g, 1) for g in d['genuine']] +
                          [(f, 0) for f in d['forged']]):
            img   = transform(Image.open(path).convert('L')).unsqueeze(0).to(device)
            emb   = model.forward_one(img)
            score = torch.sigmoid(model.forward_pair(emb, gallery)).item()
            scores.append(score)
            labels.append(lbl)
            if script in script_res:
                script_res[script][0].append(score)
                script_res[script][1].append(lbl)

    return np.array(scores), np.array(labels), script_res


def compute_metrics(scores, labels):
    preds = (scores >= 0.5).astype(int)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    fnr     = 1 - tpr
    eer_idx = np.argmin(np.abs(fnr - fpr))
    eer     = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        'acc': np.mean(preds == labels) * 100,
        'eer': eer,
        'far': fp / (fp + tn) * 100,
        'frr': fn / (fn + tp) * 100,
        'auc': roc_auc,
        'eer_threshold': thresholds[eer_idx],
    }


def print_results(tag, metrics):
    print(f"  {tag:10s}: "
          f"ACC={metrics['acc']:.2f}%  "
          f"EER={metrics['eer']:.2f}%  "
          f"FAR={metrics['far']:.2f}%  "
          f"FRR={metrics['frr']:.2f}%  "
          f"AUC={metrics['auc']:.4f}")


def run_evaluation(args):
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transform()

    # Load model
    model = MSFVNet().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"✅ Model loaded from {args.checkpoint}")

    # Load data
    all_data = scan_dataset(args.data_root)
    all_pids = sorted(all_data.keys())
    print(f"✅ Dataset: {len(all_pids)} persons\n")

    seed_results = []

    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        shuffled   = all_pids.copy()
        random.shuffle(shuffled)
        n          = len(shuffled)
        test_pids  = shuffled[int(0.85 * n):]

        scores, labels, script_res = evaluate_persons(
            model, test_pids, all_data, device, transform)

        metrics = compute_metrics(scores, labels)
        seed_results.append(metrics)

        print(f"Seed {seed} — Test: {len(test_pids)} persons")
        print(f"{'='*65}")
        print_results('Overall', metrics)

        script_names = {'A': 'Arabic', 'E': 'English', 'H': 'Hindi'}
        for s, name in script_names.items():
            sc, sl = np.array(script_res[s][0]), np.array(script_res[s][1])
            if len(set(sl)) < 2:
                continue
            sm = compute_metrics(sc, sl)
            print_results(name, sm)
        print(f"{'='*65}\n")

    # Summary across seeds
    if len(seed_results) > 1:
        print("SUMMARY (mean ± std across seeds)")
        print(f"{'='*65}")
        for key in ['acc', 'eer', 'far', 'frr', 'auc']:
            vals = [r[key] for r in seed_results]
            fmt  = '.4f' if key == 'auc' else '.2f'
            unit = '' if key == 'auc' else '%'
            print(f"  {key.upper():6s}: {np.mean(vals):{fmt}}{unit} "
                  f"± {np.std(vals):{fmt}}")
        print(f"{'='*65}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate MSFV-Net')
    parser.add_argument('--checkpoint', default='verifier_msfv.pth')
    parser.add_argument('--data_root',  default='extracted')
    parser.add_argument('--splits',     default='splits_v12_final.pkl')
    parser.add_argument('--seeds',      type=int, nargs='+', default=[42, 123, 456])
    args = parser.parse_args()
    run_evaluation(args)
