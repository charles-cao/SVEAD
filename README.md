# SVEAD: Stochastic Voronoi-based Ensemble Anomaly Detection

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)



## 📊 Algorithm Overview

SVEAD partitions the feature space using Voronoi tessellation with randomly sampled anchor points. For each test sample, the anomaly score is computed based on:

1. **Relative Position**: Distance to nearest anchor normalized by cell radius
2. **Cell Density**: Average density of the assigned Voronoi cell
3. **Ensemble Averaging**: Scores averaged across multiple random tessellations

## 🚀 Installation

### Requirements
```bash
pip install torch numpy scikit-learn
```

**Minimum versions:**
- Python >= 3.8
- PyTorch >= 2.0
- NumPy >= 1.20
- scikit-learn >= 1.0 (optional, for evaluation metrics)


## 📖 Quick Start

### Basic Usage
```python
import numpy as np
from svead import SVEAD

# Generate sample data
X_train = np.random.randn(10000, 10)  # 10k samples, 10 features
X_test = np.random.randn(1000, 10)    # 1k test samples

# Initialize and fit
model = SVEAD(max_samples=256, t=100, random_state=42)
model.fit(X_train)

# Compute anomaly scores
scores = model.decision_function(X_test)
```

### With Evaluation
```python
from sklearn.metrics import roc_auc_score, average_precision_score

# Assuming y_test contains true labels (0: normal, 1: anomaly)
auc_roc = roc_auc_score(y_test, scores)
auc_pr = average_precision_score(y_test, scores)

print(f"AUC-ROC: {auc_roc:.4f}")
print(f"AUC-PR:  {auc_pr:.4f}")
```

## 🎛️ Parameters
```python
SVEAD(
    max_samples=256,      # Number of anchor points per tessellation
    t=100,                # Number of ensemble tessellations
    random_state=None     # Random seed for reproducibility
)
```

### Parameter Guidelines

| Parameter | Description | Typical Range | Impact |
|-----------|-------------|---------------|--------|
| `max_samples` | Anchor points per tessellation | 4-512 | Higher = finer granularity, more memory |
| `n_estimators` | Number of tessellations | 20-200 | Higher = more stable, slower |
| `random_state` | Random seed | Any int | For reproducibility |

### Batch Processing for Large Datasets
```python
# For very large test sets, adjust batch_size
scores = model.decision_function(X_test, batch_size=50000)  # Smaller batches = less memory
```

### Device Selection
```python
# SVEAD automatically selects the best available device
# Priority: CUDA > MPS (Apple Silicon) > CPU

# Check which device is being used
print(f"Using device: {model.device}")
```

## 📄 License


This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use SVEAD in your research, please cite:
```bibtex
@article{cao2026stochastic,
  title={Stochastic Voronoi Ensembles for Anomaly Detection},
  author={Cao, Yang and Yang, Sikun and Yang, Yujiu},
  journal={arXiv preprint arXiv:2601.03664},
  year={2026}
}
```
