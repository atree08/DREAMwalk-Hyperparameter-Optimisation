# DREAMwalk-Hyperparameter-Optimisation
Systematic Bayesian optimization of DREAMwalk parameters for drug-disease association prediction

This repository contains code and results for systematic optimization of DREAMwalk's adaptive random walk parameters for drug-disease association prediction.

## Overview

We conducted Bayesian optimization across 50 trials to identify optimal DREAMwalk configurations, achieving:
- **98.38%** AUROC (+2.6% over baseline)
- **97.45%** AUPR (+6.6% over baseline)

**Key Finding**: Minimal teleportation (tp_factor ≈ 0.0075) dramatically outperforms the default (0.5), revealing that continuous biological pathways are more informative than similarity-based shortcuts.

## Requirements
```bash
pip install -r requirements.txt
```

## Usage

### 1. Run Optimization
```bash
python scripts/run_optimisation.py
```

### 2. Final Validation
```bash
python scripts/run_final_10fold.py
```

### 3. Generate Visualizations
```bash
python scripts/create_visualisations.py
```

## Results

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| AUROC | 95.78% | 98.38% | +2.60% |
| AUPR | 90.83% | 97.45% | +6.62% |

## Optimized Configuration
```python
num_walks: 200
walk_length: 10
tp_factor: 0.0075
p: 0.5
q: 0.5
dimension: 256
window_size: 8
```

## Citation

If you use this work, please cite:
Multi-layer knowledge graph learning for rare disease drug repurposing

## Contact

Atreyi Bhattacharyya - atreyi0112@gmail.com
Shaikh Muskan - shaikhmuskanx@gmail.com
Guan Xue Hui - fionaguan1@gmail.com
