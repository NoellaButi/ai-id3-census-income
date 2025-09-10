# ID3 Decision Tree — Census Income 💼📊

### Purpose
From-scratch implementation of **ID3** decision trees (entropy & information gain), tested on the Census Income dataset and compared against a scikit-learn baseline.

### Workflow
1. Preprocess raw Census data → categorical-binned & one-hot versions
2. Train ID3 (from scratch)
3. Train scikit-learn DecisionTree baseline
4. Evaluate (accuracy, precision/recall/F1, confusion matrix)
5. Visualize trees & save metrics

### Results Snapshot
- **ID3** → accuracy ≈ 0.819, F1 ≈ 0.594  
- **DecisionTree (sklearn, tuned)** → accuracy ≈ 0.826, F1 ≈ 0.612  

### Repo Structure
```
id3-census-income/
├── data/
│ ├── raw/ # not committed
│ └── processed/ # not committed
├── notebooks/
│ └── 03_census.ipynb
├── reports/
│ ├── assets/ # tree PNGs (ignored)
│ └── metrics_census.json
├── src/
│ └── preprocessing/
│ └── make_census.py
├── .gitignore
├── README.md
```
### Artifacts
- Metrics JSON → `reports/metrics_census.json`  
- Trees/visuals → saved in Drive, not tracked in GitHub  

---

## 4. Stage & commit
```bash
git add .
git commit -m "Initial commit: ID3 Census Income project with preprocessing, notebook, and metrics"
```
