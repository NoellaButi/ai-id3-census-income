# ID3 Decision Tree — Census Income 🌳💼  
From-scratch **ID3** with entropy & information gain; validated on toy sets and scaled to **Census Income**. Includes a scikit-learn baseline.

![Language](https://img.shields.io/badge/language-Python-blue.svg)
![Notebook](https://img.shields.io/badge/tool-Jupyter-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![CI](https://github.com/NoellaButi/ai-id3-census-income/actions/workflows/ci.yml/badge.svg?branch=main)

![Tree Visualization (Graphviz)](reports/assets/census_id3_tree_pruned.png)

---

## ✨ Overview
This repo implements the **ID3 decision tree algorithm** from scratch with:
- **Entropy** and **information gain** splits  
- Handling of categorical features (with simple binning for continuous)  
- Export to **Graphviz** for readable, pruned trees  
- Comparison with `sklearn.tree.DecisionTreeClassifier`

---

## 🔍 Features
- Pure-Python ID3 (no sklearn for training)  
- Preprocessing: categorical encoding / binning, train/test split  
- Evaluation: accuracy, precision, recall, F1, confusion matrix  
- **Graphviz** exports for trees (full + pruned)

---

## 🚦 Quickstart

```bash
# Create env and install deps
python -m venv .venv
source .venv/bin/activate     # Windows: .\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/01_modeling_census_income.ipynb
```

## 📁 Repository Layout
```bash
ai-id3-census-income/
├─ data/                 # raw/processed datasets (PlayTennis, Emails, Census)
├─ notebooks/            # 01_playtennis.ipynb, 02_emails.ipynb, 03_census_income.ipynb
├─ reports/              # metrics JSON, exported/pruned tree images (Graphviz), figures
├─ src/                  # preprocessing + ID3 implementation + helpers
├─ requirements.txt
└─ README.md
```

## 📊 Results (Test Set)

| Model                   | Accuracy | Precision | Recall |   F1   |
|-------------------------|:--------:|:---------:|:------:|:------:|
| **ID3 (binned)**        |  0.819   |   0.663   | 0.538  | 0.594  |
| DecisionTree (sklearn)  |  0.826   |   0.680   | 0.557  | 0.612  |

## 🖼️ Visuals

![Tree Visualization (Graphviz)](reports/assets/census_id3_tree.png)


![Tree Visualization (Graphviz)](reports/assets/census_id3_tree_pruned.png)


![Tree Visualization (Graphviz)](reports/assets/census_sklearn_tree.png)


![Tree Visualization (Graphviz)](reports/assets/census_sklearn_tree_best.png)


![Tree Visualization (Graphviz)](reports/assets/emails_tree.png)


![Tree Visualization (Graphviz)](reports/assets/playtennis_tree.png)


## 🔮 Roadmap
- Post-pruning via validation set or MDL
- Handling of missing values
- Export/import tree as JSON

## 📜 License
MIT (see [LICENSE](LICENSE))

---
