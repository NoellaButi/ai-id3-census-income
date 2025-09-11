# ID3 Decision Tree — Census Income 🌳💼  
From-Scratch Implementation of ID3 with Entropy & Information Gain  

![Language](https://img.shields.io/badge/language-Python-blue.svg) 
![Notebook](https://img.shields.io/badge/tool-Jupyter-orange.svg) 
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)  

---

✨ **Overview**  
This project implements the **ID3 decision tree algorithm** from scratch using entropy and information gain.  
It is validated on toy datasets (*PlayTennis*, *Spam Emails*) and scaled to the **Census Income dataset**.  
A scikit-learn DecisionTree baseline is included for comparison.  

🛠️ **Workflow**  
- Load toy datasets and Census Income data  
- Preprocess categorical features (binning + one-hot encoding)  
- Build trees recursively using entropy & information gain  
- Visualize with Graphviz (pruned trees for readability)  
- Evaluate with accuracy, precision, recall, F1  

📁 **Repository Layout**  
```bash
data/           # raw & processed datasets (PlayTennis, Emails, Census)
notebooks/      # Jupyter notebooks (PlayTennis, Emails, Census Income)
reports/        # metrics JSON, visualizations, tree exports
src/            # preprocessing scripts (make_census.py)
requirements.txt
README.md
```

🚦 **Demo**

Open notebooks directly in Jupyter or Colab:
```bash
jupyter notebook notebooks/01_modeling_playtennis.ipynb
jupyter notebook notebooks/02_modeling_emails.ipynb
jupyter notebook notebooks/03_modeling_census_income.ipynb
```

🔍 **Features**
- ID3 algorithm (from scratch) with entropy & info gain
- Handles categorical & binned numerical features
- Visualizes trees with Graphviz
- Evaluates with confusion matrix, precision, recall, F1
- Compares against scikit-learn DecisionTree

🚦 **Results (Test Set)**
```bash
Model              Accuracy   Precision   Recall   F1
------------------------------------------------------
ID3 (binned)         0.819       0.663     0.538   0.594
DecisionTree (sk)    0.826       0.680     0.557   0.612
```

📜 **License**

MIT (see [LICENSE](LICENSE))

---
