# ML_STUDIO//PRO

```
root@ml-studio:~$ ./launch --mode=pro█
```

A terminal-style, no-code machine learning workbench built with **Streamlit** and **scikit-learn**.
Load a dataset, configure preprocessing, train or benchmark models, inspect the results, and run predictions, all in a black-and-green hacker UI.

---

## Features

| Tab | What it does |
|---|---|
| **01 // DATA** | Demo datasets (Iris, Wine, Breast Cancer, Diabetes) or upload your own CSV/XLSX. Dataset stats, missing-value chart, column distributions, correlation heatmap. |
| **02 // CONFIG** | Choose the task (classification, regression, clustering), target and features. Set the numeric imputer, scaler, test split, stratification, duplicate removal and random seed. |
| **03 // TRAIN** | Train a single model with tunable hyperparameters (optional 5-fold cross-validation), or **benchmark every algorithm** and get a ranked leaderboard. |
| **04 // EVALUATE** | Confusion matrix, ROC curve and per-class report (classification). Actual vs predicted and residual plots (regression). 2D/3D PCA view, cluster sizes and a K-Means elbow/silhouette sweep (clustering). Permutation feature importance. Download predictions (CSV) and the trained model (`.joblib`). |
| **05 // PREDICT** | Enter feature values by hand and get a prediction, with class probabilities where the model supports them. |

### Supported algorithms

- **Classification:** Random Forest, Extra Trees, Gradient Boosting, SVM, Logistic Regression, KNN, Decision Tree
- **Regression:** Random Forest, Gradient Boosting, SVR, Linear Regression, Ridge, Lasso, KNN, Decision Tree
- **Clustering:** K-Means, Agglomerative, DBSCAN, Gaussian Mixture

### Under the hood

- Every model runs inside a scikit-learn `Pipeline`: impute, scale, one-hot encode categoricals, then fit the model. The exported `.joblib` file therefore accepts raw feature columns.
- Plots use Plotly with a custom black/green theme. No `statsmodels` dependency; trendlines are computed with NumPy.
- Changing the config automatically clears previous runs so results never mix different setups.

---

## Project structure

```
ml_studio_platform/
├── app.py                  # the whole application
├── favicon.png             # transparent "ML" browser-tab icon
├── requirements.txt        # Python dependencies
├── README.md
└── .streamlit/
    └── config.toml         # dark/green theme + monospace font
```

---

## Run locally

```bash
# 1. clone
git clone https://github.com/<your-username>/ml_studio_platform.git
cd ml_studio_platform

# 2. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. launch
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**Python version:** 3.11 to 3.13 is recommended.

---

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository. Keep `app.py` and `favicon.png` in the **repo root**, and keep the `.streamlit` folder name exactly as written (with the leading dot).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **Create app**.
3. Select your repository and branch, and set the main file path to `app.py`.
4. Click **Deploy**. Cloud redeploys automatically on every push.

If the theme or icon doesn't update after a push, use **Manage app → Reboot app**, then hard-refresh the browser (Ctrl+Shift+R).

---

## Usage guide

1. **Data:** pick a demo dataset or upload a CSV/XLSX file.
2. **Config:** select the task and target. Pick features and preprocessing options.
3. **Train:** train one model, or run the benchmark to compare them all.
4. **Evaluate:** review metrics and charts, compute feature importance, and export predictions or the model.
5. **Predict:** try the model on custom inputs.

### Loading an exported model

```python
import joblib
import pandas as pd

bundle = joblib.load("model_Random Forest.joblib")
pipe = bundle["pipeline"]

row = pd.DataFrame([{...}])[bundle["features"]]   # raw feature values
pred = pipe.predict(row)

# for classification, map the encoded prediction back to its label
print(bundle["classes"][int(pred[0])])
```

Load `.joblib` files only from sources you trust, since they are pickle-based.

---

## Tech stack

- [Streamlit](https://streamlit.io), UI
- [scikit-learn](https://scikit-learn.org), models and preprocessing
- [Plotly](https://plotly.com/python/), charts
- [pandas](https://pandas.pydata.org) and [NumPy](https://numpy.org), data handling
- [openpyxl](https://openpyxl.readthedocs.io), Excel upload support

---

## Notes and limitations

- Intended for small to medium tabular datasets. Large files and slow algorithms (for example SVM in benchmark mode) can take a while on free hosting tiers.
- Class labels are encoded alphabetically as strings during classification; the original names are restored in plots and exports.
- DBSCAN and Agglomerative clustering can't assign new points, so the Predict tab only supports K-Means and Gaussian Mixture for clustering.
- Rows with a missing target are dropped; missing feature values are imputed (or the rows are dropped, depending on the config).

---

## Contributing

Issues and pull requests are welcome. For larger changes, please open an issue first to discuss what you'd like to change.