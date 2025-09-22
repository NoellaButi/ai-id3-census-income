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
```

# Run notebook

jupyter notebook notebooks/01_modeling_census_income.ipynb
```
📁 Repository Layout
bash
Copy code
ai-id3-census-income/
├─ data/                 # raw/processed datasets (PlayTennis, Emails, Census)
├─ notebooks/            # 01_playtennis.ipynb, 02_emails.ipynb, 03_census_income.ipynb
├─ reports/              # metrics JSON, confusion matrix, figures
├─ docs/                 # exported/pruned tree images (Graphviz)
├─ src/                  # preprocessing + ID3 implementation + helpers
├─ requirements.txt
└─ README.md
📊 Results (Test Set)
Model	Accuracy	Precision	Recall	F1
ID3 (binned)	0.819	0.663	0.538	0.594
DecisionTree (sklearn)	0.826	0.680	0.557	0.612

Confusion matrix + full metrics also saved in reports/.

🖼️ Visuals
Pruned tree: docs/tree_pruned.png

Full tree (optional): docs/tree_full.png

Confusion matrix: reports/confusion_matrix.png

md
Copy code
![Pruned ID3 Tree](docs/tree_pruned.png)
![Confusion Matrix](reports/confusion_matrix.png)
Tip: if you’re generating Graphviz from code, export with something like:

python
Copy code
from graphviz import Digraph
dot = Digraph(comment="ID3 (Pruned)")
# ... build nodes/edges ...
dot.render("docs/tree_pruned", format="png", cleanup=True)
🔮 Roadmap
 Post-pruning via validation set or MDL

 Handling of missing values

 Export/import tree as JSON

📜 License
MIT (see LICENSE)

yaml
Copy code

> Put your exported images at those paths so the README renders instantly:
> - `docs/tree_pruned.png`
> - `reports/confusion_matrix.png`

---
