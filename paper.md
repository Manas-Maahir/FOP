# Paper Summary — *Revisiting Computer-Aided Tuberculosis Diagnosis*

**Authors:** Yun Liu, Yu-Huan Wu, Shi-Chen Zhang, Li Liu, Min Wu, Ming-Ming Cheng
**Venue:** IEEE TPAMI (journal extension of the CVPR 2020 **oral** "Rethinking Computer-Aided Tuberculosis Diagnosis")
**Preprint:** arXiv:2307.02848v2 (5 Dec 2023)
**Code & data:** https://github.com/yun-liu/Tuberculosis

---

## TL;DR
Tuberculosis (TB) kills ~1.4M people/year, and early diagnosis from chest X-rays (CXR) is hard — even experienced radiologists in this study reach only **68.7%** accuracy. Deep learning could help, but the public TB datasets were tiny (hundreds of images, image-level labels only). This paper makes three contributions: (1) **TBX11K**, a large-scale dataset of **11,200 CXR images** with **bounding-box** annotations of TB infection areas across **4 classes**; (2) **SymFormer**, a detector that exploits the **bilateral (left/right) symmetry** of CXRs via **Symmetric Search Attention (SymAttention)** and **Symmetric Positional Encoding (SPE)** to jointly classify the image and detect TB regions; and (3) a **benchmark** (metrics, baselines, and an online challenge). SymFormer reaches state-of-the-art on TBX11K.

---

## 1. Problem & Motivation
- TB is the 2nd most lethal infectious disease (after COVID-19 at the time of writing). The diagnostic gold standard (sputum culture in a BSL-3 lab) can take **months** and is unavailable in many resource-constrained settings.
- **CXR** is the WHO-recommended first screening step, but reading it is error-prone: the paper's human study shows an experienced radiologist scores only **68.7%** accuracy vs the gold standard (**84.8%** if active vs latent TB is not distinguished).
- Deep learning needs lots of data. Before TBX11K, public TB datasets were small and **image-level only**:

  | Dataset | Year | #Classes | Annotations | #Samples |
  |---|---|---|---|---|
  | MC | 2014 | 2 | Image-level | 138 |
  | Shenzhen | 2014 | 2 | Image-level | 662 |
  | DA | 2014 | 2 | Image-level | 156 |
  | DB | 2014 | 2 | Image-level | 150 |
  | **TBX11K (this paper)** | — | **4** | **Bounding box** | **11,200** |

  TBX11K is ~**17× larger** than the previous largest (Shenzhen) and is the **first** dataset with TB-area bounding boxes.

---

## 2. Contribution 1 — The TBX11K Dataset

### Classes (4)
- **Healthy**
- **Sick but non-TB** (a new, clinically important class — many chest diseases mimic TB and cause false positives if "non-TB" only means "healthy")
- **Active TB**
- **Latent TB**

Distinguishing **active vs latent** matters for treatment (active TB is contagious/sick; latent is neither).

### Composition
- **11,200** CXR images, each a unique patient, resolution ≈ **3000 × 3000**, with age & gender metadata.
- **5,000 healthy**, **5,000 sick-but-non-TB**, **1,200 TB**.
- Of the 1,200 TB images: **924 active**, **212 latent**, **54 both active & latent**, **10 uncertain** (type not currently recognizable).
- Realistic clinical imbalance: 44.6% healthy, 44.6% sick-non-TB, 10.7% TB.

### Split (Table 2) — TB cases kept at a 3:1:2 train:val:test ratio
| Group | Class | Train | Val | Test | Total |
|---|---|---|---|---|---|
| Non-TB | Healthy | 3,000 | 800 | 1,200 | 5,000 |
| Non-TB | Sick & Non-TB | 3,000 | 800 | 1,200 | 5,000 |
| TB | Active TB | 473 | 157 | 294 | 924 |
| TB | Latent TB | 104 | 36 | 72 | 212 |
| TB | Active & Latent TB | 23 | 7 | 24 | 54 |
| TB | Uncertain TB | 0 | 0 | 10 | 10 |
| **Total** | | **6,600** | **1,800** | **2,800** | **11,200** |

- The **10 uncertain** TB images go to the test set to enable **class-agnostic** TB detection evaluation.
- **Test ground truth is private** — scoring on test is done via an **online challenge**.
- **Recommended protocol:** train on *train*, tune hyper-parameters on *val*, then **retrain on trainval (train+val)** and report on *test*.

### Annotation protocol
- Every CXR is first labelled by the **gold standard** (diagnostic microbiology) for its image-level class.
- TB bounding boxes are drawn by a radiologist with **5–10 years** of experience, then reviewed by one with **>10 years**; each box also gets a TB **type** (active/latent).
- Box types are double-checked against the gold-standard image label; mismatches are re-annotated blind. Data is de-identified and approved for public release.

---

## 3. Contribution 2 — The SymFormer Framework

SymFormer performs **simultaneous CXR image classification and TB infection-area detection**. Core idea: a normal chest is roughly **bilaterally symmetric** — comparing the left and right lungs is how radiologists spot abnormalities — so the network is built to **search the mirror-image location** of each position.

### 3.1 Feature extraction
- Backbone: **ResNet-50** (or **P2T-Small**), producing 4 stages downsampled by 1/4, 1/8, 1/16, 1/32.
- **FPN** builds a feature pyramid `F = {F1, F2, F3, F4}` with **C = 256** channels.

### 3.2 SAS — Symmetric Abnormity Search (the novel module)
An SAS module is inserted after **each** FPN level (weights **shared** across levels) and contains three parts: **SPE → SymAttention → FFN**.

**(a) SPE — Symmetric Positional Encoding**
- Start from standard **absolute** sine/cosine positional encoding `P` (the paper found relative PE inferior):
  - `P[pos, 2j]   = sin(pos / 10000^{2j/C})`
  - `P[pos, 2j+1] = cos(pos / 10000^{2j/C})`
- Real CXRs aren't perfectly symmetric (pose/rotation), so SPE **recalibrates** the encoding:
  1. Split `P` into left/right halves at the vertical centerline.
  2. Transfer the **right** side to the left using a **Spatial Transformer Network (STN)** (predicts an affine transform) + **horizontal flip**.
  3. Concatenate → `P_sym`.
  4. Recalibrate the feature: `F_recalib = F + P_sym`.
- **STN micro-design:** 2× alternating [max-pool + Conv-ReLU] → flatten → MLP → affine matrix, **initialised to the identity** transform.
- Default direction: transfer **right → left** (slightly better than left → right in ablations).

**(b) SymAttention — Symmetric Search Attention**
- Built on **Deformable DETR**-style sparse sampling, but each query attends **around its bilaterally symmetric location** (mirror across the vertical centerline) instead of globally.
- `M = 8` attention heads, `K = 4` sampled key locations per head; sampling offsets are **learned** (observed to stay within ~10% of feature-map width).
- Learns coordinate shifts `Δp`, computes attention `A` and values, samples at the **mirrored** coordinates, then a residual connection + MLP.
- Complexity is **O(N·C²)** (N = H×W), i.e. efficient and pyramid-friendly.
- Rationale: global self-attention is wasteful for CXRs (single scene, tiny abnormal regions); the authors note plain **DETR fails to converge** on this task. Restricting attention to the symmetric neighbourhood injects the right inductive bias.

### 3.3 The two heads
- **Detection head** = **RetinaNet** (default, one-stage) — or **Deformable DETR** — with box-classification + box-regression branches; detects two categories: **active TB** and **latent TB**.
- **Classification head** = take the top pyramid level `F̂4` → **5× (Conv 512 + ReLU)** → global average pooling → **FC with 3 outputs**: healthy / sick-non-TB / TB.
- The classifier is used to **filter detection false positives**: if an image is classified non-TB, its detected boxes are discarded. This matters because most clinical CXRs have no TB.

### 3.4 Two-stage training
1. **Stage 1 — detection.** Train the backbone + detection head using **TB images only** (avoids drowning the detector in pure-background non-TB images). 24 epochs (50 for Deformable DETR).
2. **Stage 2 — classification.** **Freeze** the backbone + detection head; train only the classification head using **all** images. 12 epochs.
- Training first on detection gives fine-grained, transferable features and avoids overfitting the backbone to global classification cues.

### 3.5 Implementation details (from the paper)
- **PyTorch** + **mmdetection**; FPN channels **C = 256** (as in RetinaNet).
- Optimizers: **AdamW** for Deformable-DETR-based models, **SGD** for the rest.
- **Batch size 8**; **resize to 512×512**; augmentation = **random flipping**; hardware = **2× TITAN XP**.
- SymAttention hyper-parameters: **M = 8, K = 4**.

---

## 4. Contribution 3 — The CTD Benchmark

### Baselines
Five popular detectors, each **reformed** with the same classification head + two-stage training: **SSD** (VGGNet-16), **RetinaNet**, **Faster R-CNN**, **FCOS**, **Deformable DETR** (latter four on ResNet-50 + FPN).

### Metrics
- **Classification:** Accuracy, **AUC** (TB), **Sensitivity** (recall for TB), **Specificity** (recall for non-TB), Average Precision (AP), Average Recall (AR), **F1**, and the confusion matrix.
- **Detection:** COCO **AP^bb** (averaged over IoU 0.5:0.05:0.95) and **AP^bb_50** (IoU 0.5), reported for **active / latent / category-agnostic** TB, under **two evaluation modes** — using **all** test images vs **TB-only** images.

### Headline results

**Image classification on TBX11K test (Table 3):**
| Method | Backbone | Acc | AUC(TB) | Sens | Spec | AP | AR |
|---|---|---|---|---|---|---|---|
| SSD | VGGNet-16 | 84.7 | 93.0 | 78.1 | 89.4 | 82.1 | 83.8 |
| RetinaNet | ResNet-50+FPN | 87.4 | 91.8 | 81.6 | 89.8 | 84.8 | 86.8 |
| Faster R-CNN | ResNet-50+FPN | 89.7 | 93.6 | 91.2 | 89.9 | 87.7 | 90.5 |
| FCOS | ResNet-50+FPN | 88.9 | 92.4 | 87.3 | 89.9 | 86.6 | 89.2 |
| Deformable DETR | ResNet-50+FPN | 91.3 | 97.6 | 89.2 | 95.3 | 89.8 | 91.0 |
| **SymFormer w/ Def. DETR** | ResNet-50+FPN | 94.3 | 98.5 | 87.3 | 97.3 | 93.2 | 93.2 |
| **SymFormer w/ RetinaNet** | ResNet-50+FPN | 94.5 | 98.9 | 91.0 | 96.8 | 93.3 | 94.0 |
| **SymFormer w/ RetinaNet** | P2T-Small+FPN | 94.6 | 99.1 | 92.1 | 96.7 | 93.4 | 94.2 |

All deep models beat the radiologist (84.8% comparable accuracy). Baselines tend to over-predict "TB" (high sensitivity, poor specificity), so **F1** (SymFormer w/ RetinaNet = **89.0** vs RetinaNet 73.1) is the fairer summary. SymFormer w/ RetinaNet runs at **24.3 fps** (ResNet-50) / 17.9 fps (P2T-Small).

**TB detection on TBX11K test (Table 7), TB-only mode, AP50 / AP:**
| Method | Backbone | Cat-agnostic | Active TB | Latent TB |
|---|---|---|---|---|
| SSD | VGGNet-16 | 68.3 / 28.7 | 63.7 / 28.0 | 10.7 / 4.0 |
| RetinaNet | ResNet-50+FPN | 69.4 / 28.3 | 61.5 / 25.3 | 10.2 / 4.1 |
| Faster R-CNN | ResNet-50+FPN | 63.4 / 24.6 | 58.7 / 23.7 | 9.6 / 2.8 |
| FCOS | ResNet-50+FPN | 56.3 / 22.5 | 47.9 / 19.8 | 7.4 / 2.4 |
| Deformable DETR | ResNet-50+FPN | 57.4 / 24.2 | 54.5 / 23.5 | 7.6 / 2.3 |
| **SymFormer w/ Def. DETR** | ResNet-50+FPN | 60.8 / 24.5 | 55.2 / 23.8 | 9.2 / 2.6 |
| **SymFormer w/ RetinaNet** | ResNet-50+FPN | **73.4 / 31.5** | **67.1 / 29.2** | **14.7 / 4.8** |
| **SymFormer w/ RetinaNet** | P2T-Small+FPN | **75.7 / 32.1** | 68.9 / 28.9 | 13.0 / 4.7 |

SymFormer w/ RetinaNet is the authors' **default** model (best detection; better suited to RetinaNet than to Deformable DETR).

**Ablation on TBX11K val (Table 8), category-agnostic, TB-only, AP50 / AP:**
| Attention | Positional Encoding | Symmetry | AP50 | AP |
|---|---|---|---|---|
| None | None | — | 72.7 | 31.0 |
| Vanilla (deformable) | APE | — | 73.4 | 30.6 |
| Vanilla | RPE | — | 72.7 | 29.7 |
| Vanilla | SPE w/o STN | R→L | 74.3 | 30.8 |
| Vanilla | SPE | R→L | 75.7 | 29.6 |
| SymAttention | APE | — | 74.9 | 30.0 |
| SymAttention | RPE | — | 73.6 | 29.1 |
| SymAttention | SPE w/o STN | R→L | 75.5 | 30.7 |
| **SymAttention** | **SPE** | **R→L** | **76.6** | **31.7** |

Takeaways from the ablation: **SymAttention > vanilla attention** at every PE setting; **SPE > APE > RPE**; the **STN** adds a further gain; **right→left** transfer is slightly better than left→right. A **4-fold cross-validation** (Table 9) confirms the results are stable.

---

## 5. Why it matters
- TBX11K removes the data bottleneck and is the first dataset enabling TB **detection** (not just classification), with a realistic 4-class clinical distribution.
- SymFormer shows a **domain-specific inductive bias** (bilateral symmetry) can beat strong generic detectors and even DETR, which fails to converge here.
- The benchmark + online challenge provide a fair, standard yardstick for future computer-aided TB diagnosis (CTD) research.

---

*This file summarizes the paper only. For how we plan to replicate it on Google Colab, see [CLAUDE.md](CLAUDE.md); for caveats, see [limitations.md](limitations.md).*
