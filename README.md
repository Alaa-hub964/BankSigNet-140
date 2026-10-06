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
- **Reproducible splits** — fixed signer-disjoint train/val/test partition included.

---

## Dataset Structure

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

## Benchmark Results

Results from our paper (open-set signer-disjoint evaluation, mean ± std over 3 seeds):

| Method | ACC (%) | EER (%) | AUC |
|---|---|---|---|
| MobileNetV2 Siamese (Reddy 2025) | 60.9 | — | — |
| ResNet50 + Focal Loss (Xiao 2024) | 41.0 | — | — |
| CNN + Autoencoder (Harinadh 2025) | 48.4 | — | — |
| IGCN+RL (Priya 2025) | 57.1† | — | — |
| **MSFV-Net (ours)** | **75.8 ± 3.2** | **24.25 ± 3.11** | **0.879 ± 0.039** |

† Evaluated under open-set person-level protocol; original paper reports 94.11% on BHSig260-Hindi (plain paper).

Per-script accuracy:

| Script | ACC (%) | EER (%) | AUC |
|---|---|---|---|
| Hindi (n=45) | 90.9 | 3.9 | 0.993 |
| English (n=46) | 74.2 | 18.2 | 0.924 |
| Arabic (n=49) | 72.8 | 30.6 | 0.803 |

---



## Usage

```python
from pathlib import Path
from PIL import Image

ROOT = Path('extracted')

def load_signer(person_id):
    genuine = sorted((ROOT / 'genuine' / person_id).glob('*.png'))
    forged  = sorted((ROOT / 'forged'  / person_id).glob('*.png'))
    return genuine, forged

# Example
genuine, forged = load_signer('H001')
img = Image.open(genuine[0])
```

---

## Citation

If you use BankSigNet-140 in your research, please cite:

```bibtex
@article{alowaidi2026msfvnet,
  title   = {MSFV-Net: Explainable Multi-Script Offline Signature Verification
             on Real Bank Cheque Data},
  author  = {Alowaidi, Alaa and Kumar Pateriya, Pushpendra and Mahajan, Divya},
  journal = {International Journal on Document Analysis and Recognition},
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

For questions about the dataset, please open a GitHub issue or contact:
**abd94057@gmail.com**
