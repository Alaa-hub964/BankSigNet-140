# BankSigNet-140

**The first offline signature dataset collected directly from real Indian bank cheques, covering Arabic, English, and Hindi scripts.**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Paper](https://img.shields.io/badge/Paper-IJDAR-blue)](https://link.springer.com/journal/10032)

---

## Overview

| Property | Value |
|---|---|
| Total signers | 140 |
| Arabic signers | 49 |
| English signers | 46 |
| Hindi signers | 45 |
| Genuine per signer | 9 |
| Forged per signer | 6 |
| Total images | 2,100 |
| Resolution | 300 DPI |
| Background | Real Indian bank cheques |
| Forgery type | Skilled (imitation from memory) |

---

## What Makes BankSigNet-140 Unique

- **Real bank cheque backgrounds** — signatures appear on printed cheque paper with pre-printed fields, borders, bank logos, and alphanumeric text. No existing public dataset provides this.
- **Three scripts simultaneously** — Arabic, English (Latin), and Hindi (Devanagari) in a single unified benchmark.
- **Skilled forgeries** — forgers viewed the target signature for 60 seconds before producing imitations from memory. Tracing was not permitted.
- **Reproducible splits** — fixed signer-disjoint train/val/test partition included (`splits_v12_final.pkl`).

---

## Dataset

The full dataset is included in this repository as `extracted.zip`.

```bash
unzip extracted.zip
```

### Structure

```
extracted/
├── genuine/
│   ├── A001/          # Arabic signer 001
│   │   ├── 001_A_G_01.png
│   │   ├── 001_A_G_02.png
│   │   └── ...        # 9 genuine images
│   ├── E001/          # English signer 001
│   ├── H001/          # Hindi signer 001
│   └── ...
└── forged/
    ├── A001/
    │   ├── 001_A_F_01.png
    │   └── ...        # 6 forged images
    ├── E001/
    ├── H001/
    └── ...
```

### File Naming Convention

```
{signer_id}_{script}_{type}_{index}.png
```

| Field | Values | Meaning |
|---|---|---|
| `signer_id` | 001–140 | Unique signer number |
| `script` | A / E / H | Arabic / English / Hindi |
| `type` | G / F | Genuine / Forged |
| `index` | 01–09 (G), 01–06 (F) | Sample index |

---

## Train / Validation / Test Split

The fixed signer-disjoint split used in all experiments is provided in `splits_v12_final.pkl`.

```python
import pickle

with open('splits_v12_final.pkl', 'rb') as f:
    splits = pickle.load(f)

# Keys: train_gen, train_frg, test_gen, test_frg
# Each value is a dict: {person_id: [list of image paths]}
train_ids = list(splits['train_gen'].keys())  # 98 signers
test_ids  = list(splits['test_gen'].keys())   # 21 signers (held out)
```

- **Training:** 98 signers
- **Validation:** 21 signers
- **Test:** 21 signers
- **Protocol:** Open-set signer-disjoint — test signers never appear in training or validation

---

## Collection Protocol

- **Scanner:** Canon DR-C230 flatbed document scanner, 300 DPI, A4 format
- **Annotation:** Signature bounding boxes marked manually by two independent annotators; disagreements resolved by consensus
- **Genuine collection:** Single session, standard banking signature on provided cheque forms, no style instructions
- **Forgery collection:** Forgers recruited separately, shown target signature for 60 seconds, produced imitations from memory (no tracing)
- **Ethics:** All participants provided written informed consent. No personal banking information, account numbers, or financial data was recorded or retained.

---

## MSFV-Net Model

MSFV-Net is a multi-script forgery verification network built on a Siamese ResNet50 backbone. Key components:

- **Forged-aware dual loss** — combines triplet loss with binary cross-entropy
- **Multi-reference signature gallery** — query compared against multiple genuine references per signer
- **Script-balanced batch sampling** — equal representation of Arabic, English, and Hindi in each batch
- **Synthetic background augmentation** — simulates cheque background variation during training

Model definition: [`model/msfvnet.py`](model/msfvnet.py)

---

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python train.py \
    --data_root extracted \
    --splits    splits_v12_final.pkl \
    --save_path verifier_msfv.pth \
    --seed      42
```

### Evaluation

```bash
python evaluate.py \
    --checkpoint verifier_msfv.pth \
    --data_root  extracted \
    --splits     splits_v12_final.pkl \
    --seeds      42 123 456
```

Reports ACC, EER, FAR, FRR, and AUC overall and per script.

### Explainability (Difference GradCAM)

```bash
python xai/difference_gradcam.py \
    --checkpoint verifier_msfv.pth \
    --data_root  extracted \
    --splits     splits_v12_final.pkl
```

Computes Difference GradCAM heatmaps and Deletion/Insertion AUC faithfulness metrics.

---

## Benchmark Results

Results from our paper (open-set signer-disjoint evaluation, mean ± std over 3 seeds):

| Method | ACC (%) | EER (%) | AUC |
|---|---|---|---|
| MobileNetV2 Siamese (Reddy 2025) | 60.9 | — | — |
| ResNet50 + Focal Loss (Xiao 2024) | 41.0 | — | — |
| CNN + Autoencoder (Harinadh 2025) | 48.4 | — | — |
| IGCN+RL (Priya 2025) | 57.1† | — | — |
| **MSFV-Net (ours)** | **73.56 ± 2.16** | **26.07 ± 1.45** | **0.810 ± 0.017** |

† Evaluated under open-set person-level protocol; original paper reports 94.11% on BHSig260-Hindi (plain paper).

Per-script accuracy on BankSigNet-140:

| Script | ACC (%) |
|---|---|
| Hindi (n=45) | 88.6 |
| English (n=46) | 69.2 |
| Arabic (n=49) | 66.5 |
| **Mean** | **73.56** |

Cross-dataset result on BHSig260-Hindi (standard benchmark):

| Method | ACC (%) |
|---|---|
| SigNet | 84.64 |
| IDN | 93.04 |
| **MSFV-Net (ours)** | **93.15** |

---

## Citation

If you use BankSigNet-140 or MSFV-Net in your research, please cite:

```bibtex
@article{alowaidi2026msfvnet,
  title   = { MSFV-Net for Multi-Script Bank Cheque Signature Verification
             },
  author  = { Alaa Alowaidi, and Pushpendra Kumar Pateriya },
  journal = {Al-Khwarizmi Engineering Journal},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

The dataset is released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
You are free to share and adapt the dataset for any purpose, provided appropriate credit is given.

---

## Contact

For questions about the dataset or model, please open a GitHub issue or contact:
**abd94057@gmail.com**
