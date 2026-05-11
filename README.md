# 🧠 Graph-Enhanced Computer Vision Framework with Quantum-Inspired Classification for Alzheimer's Disease Prediction from MRI

[![Paper](https://img.shields.io/badge/Springer%20LNNS-CVR%202026-blue?style=flat-square&logo=springer)](https://springer.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green?style=flat-square&logo=python)](https://python.org)
[![PennyLane](https://img.shields.io/badge/Quantum-PennyLane-blueviolet?style=flat-square)](https://pennylane.ai)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Amit2004k/alzheimers-quantum-gat-mri?style=flat-square)](https://github.com/Amit2004k/alzheimers-quantum-gat-mri/stargazers)

> **Accepted and Presented at CVR 2026 — Springer Lecture Notes in Networks and Systems (LNNS)**
> *Graph-Enhanced Computer Vision Framework with Quantum-Inspired Classification for Alzheimer's Disease Prediction from MRI*

---

## 🧠 Overview

Alzheimer's Disease (AD) affects **55 million people worldwide** — yet early diagnosis from MRI remains a major clinical bottleneck. This work proposes a novel hybrid framework combining:

- **Graph Attention Networks (GAT)** to model spatial relationships between brain regions
- **Quantum-Inspired Variational Circuits** (via PennyLane) as the final classifier
- **Explainable AI** to highlight which MRI brain regions drive predictions

---

## 🔥 Key Contributions

- ✅ **Graph-enhanced MRI representation** — brain regions modeled as graph nodes with structural connectivity edges
- ✅ **Graph Attention Networks (GAT)** — learns which brain region relationships matter most
- ✅ **Quantum-inspired classification** — variational quantum circuit (VQC) using PennyLane as a hybrid classifier
- ✅ **Multi-class AD staging** — CN (Cognitively Normal), MCI (Mild Cognitive Impairment), AD
- ✅ **GradCAM + attention visualization** — brain region-level explanations for clinicians
- ✅ Evaluated on **ADNI dataset** (Alzheimer's Disease Neuroimaging Initiative)

---

## 📊 Results at a Glance

| Model | CN vs AD | MCI vs AD | 3-Class Acc. | AUC |
|-------|----------|-----------|-------------|-----|
| ResNet-50 baseline | 91.2% | 78.3% | 82.1% | 0.941 |
| GAT only | 93.7% | 81.6% | 85.4% | 0.961 |
| VQC only | 89.4% | 79.1% | 81.8% | 0.938 |
| **GAT + VQC (Ours)** | **95.8%** | **84.2%** | **88.7%** | **0.974** |

> The quantum-inspired hybrid classifier shows **+3.3%** accuracy improvement over the classical GAT-only baseline on 3-class staging.

---

## 🏗️ Framework Architecture

```
MRI Scans (ADNI)
      │
      ▼
┌─────────────────────────┐
│  Preprocessing          │  ← Skull stripping, normalization, ROI extraction
│  (FSL / Nibabel)        │  ← 90 brain regions (AAL atlas)
└──────────┬──────────────┘
           │
      ▼
┌─────────────────────────┐
│  Graph Construction     │  ← Brain regions = nodes
│                         │  ← Structural connectivity = edges (DTI/correlation)
│                         │  ← Node features = regional gray matter volume
└──────────┬──────────────┘
           │
      ▼
┌─────────────────────────┐
│  Graph Attention Network│  ← 3-layer GAT with multi-head attention
│  (GAT Encoder)          │  ← Learns importance of region-to-region connections
└──────────┬──────────────┘
           │
      ▼
┌─────────────────────────┐
│  Quantum Variational    │  ← Angle embedding → variational layers → measurement
│  Circuit (PennyLane)    │  ← 4-qubit circuit, 3 variational layers
│                         │  ← Hybrid classical-quantum optimization
└──────────┬──────────────┘
           │
      ▼
┌─────────────────────────┐
│  XAI: GradCAM +         │  ← Attention weight visualization per brain region
│  Attention Rollout      │  ← Clinical brain map overlays
└─────────────────────────┘
```

---

## 📁 Repository Structure

```
📦 alzheimers-quantum-gat-mri
├── 📂 src/
│   ├── preprocessing.py        # MRI preprocessing + ROI extraction
│   ├── graph_builder.py        # Brain connectivity graph construction
│   ├── gat_model.py            # Graph Attention Network encoder
│   ├── quantum_classifier.py   # PennyLane VQC classifier
│   ├── hybrid_model.py         # Full GAT + VQC pipeline
│   └── explainability.py       # GradCAM + attention visualization
├── 📂 notebooks/
│   ├── 01_mri_preprocessing.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_gat_training.ipynb
│   ├── 04_quantum_classifier.ipynb
│   ├── 05_hybrid_pipeline.ipynb
│   └── 06_xai_brain_maps.ipynb
├── 📂 data/
│   └── README.md               # ADNI access instructions
├── 📂 results/
│   ├── 📂 figures/             # Brain maps, attention overlays, ROC curves
│   └── 📂 tables/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/Amit2004k/alzheimers-quantum-gat-mri.git
cd alzheimers-quantum-gat-mri
pip install -r requirements.txt
```

### Run the hybrid GAT + VQC pipeline:

```python
from src.hybrid_model import AlzheimerHybridModel

model = AlzheimerHybridModel(
    n_regions=90,        # AAL atlas brain regions
    gat_heads=4,
    n_qubits=4,
    vqc_layers=3,
    n_classes=3          # CN, MCI, AD
)

# Forward pass
logits = model(node_features, edge_index, edge_attr)
```

### Quantum circuit (PennyLane):

```python
import pennylane as qml

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def variational_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(4))
    qml.StronglyEntanglingLayers(weights, wires=range(4))
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]
```

---

## 🗂️ Dataset

**ADNI (Alzheimer's Disease Neuroimaging Initiative)**
- Access: https://adni.loni.usc.edu/ (free academic registration required)
- Modalities used: T1-weighted MRI structural scans
- Classes: CN (337), MCI (521), AD (285)
- Atlas: AAL (Automated Anatomical Labeling) — 90 cortical/subcortical regions

---

## 🔬 Why Quantum-Inspired?

Quantum variational circuits offer:
- **Exponential feature space** in n qubits — captures complex non-linear patterns in brain connectivity
- **Natural angle embedding** — MRI features map elegantly to qubit rotation angles
- **Hybrid optimization** — quantum parameters trained with classical backprop via PennyLane's autodiff

This is a *quantum-inspired* approach — runs on classical hardware using PennyLane's `default.qubit` simulator, making it fully reproducible without quantum hardware.

---

## 📖 Citation

```bibtex
@inproceedings{kalita2026alzheimer,
  title     = {Graph-Enhanced Computer Vision Framework with Quantum-Inspired Classification for Alzheimer's Disease Prediction from MRI},
  author    = {Kalita, Amit and others},
  booktitle = {Proceedings of CVR 2026},
  series    = {Lecture Notes in Networks and Systems},
  publisher = {Springer},
  year      = {2026}
}
```

---

## 🙋 Author

**Amit Kalita**
B.Tech CSE (8th Semester), Dibrugarh University
[GitHub](https://github.com/Amit2004k)

> 📌 *Part of a series of published ML research repos. See also:
> [DDI Prediction](https://github.com/Amit2004k/drug-drug-interaction-llm-xai) |
> [Fraud Detection](https://github.com/Amit2004k/fraud-detection-cost-sensitive-xai) |
> [Fairness Optimization](https://github.com/Amit2004k/fairness-threshold-optimization) |
> [Breast Cancer](https://github.com/Amit2004k/decision-aware-breast-cancer-classification)*

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

⭐ **Star this repo if you work on neuroimaging, graph ML, or quantum ML!**
