import io
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn import datasets
from sklearn.base import clone
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, auc, calinski_harabasz_score, classification_report,
                             confusion_matrix, davies_bouldin_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_curve,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
from pathlib import Path

_ICON = Path(__file__).parent / "favicon.png"
st.set_page_config(page_title="ML_STUDIO//PRO", page_icon=str(_ICON) if _ICON.exists() else "▌", layout="wide")

# ============================== THEME ==============================
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.stApp {background: #000;}
.block-container {padding-top: 2rem; max-width: 1400px;}
h1, h2, h3, h4 {color: #00ff41 !important; text-shadow: 0 0 8px rgba(0,255,65,.4);}
.credit {display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: 2px; color: #00ff41;
  background: #030b04; border: 1px solid #0f3d1a; border-left: 3px solid #00ff41; padding: .25rem .75rem;
  margin-bottom: .9rem; text-shadow: 0 0 6px rgba(0,255,65,.5);}
.prompt {color: #2e8b45; font-size: .95rem;}
.prompt .cmd {color: #9dffb0;}
.cursor {color: #00ff41; animation: blink 1s steps(2, start) infinite;}
@keyframes blink {to {visibility: hidden;}}
.title {font-size: 2.6rem !important; letter-spacing: 2px; margin: .2rem 0 0 0; padding: 0;}
.title .dim {color: #1f6b34; text-shadow: none;}
.sub {color: #2e8b45; margin-bottom: 1.2rem; border-bottom: 1px dashed #0f3d1a; padding-bottom: .8rem;}
section[data-testid="stSidebar"] {background: #020a03; border-right: 1px solid #0f3d1a;}
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  background: #000; color: #00ff41; border: 1px solid #00ff41; border-radius: 2px;
  letter-spacing: 1px; transition: all .15s;}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
  background: #00ff41; color: #000; box-shadow: 0 0 14px rgba(0,255,65,.6);}
div[data-testid="stMetric"] {background: #030b04; border: 1px solid #0f3d1a; border-left: 3px solid #00ff41;
  padding: .7rem 1rem; border-radius: 2px;}
div[data-testid="stMetricValue"] {color: #00ff41; text-shadow: 0 0 8px rgba(0,255,65,.4);}
div[data-testid="stMetricLabel"] {color: #2e8b45;}
.stTabs [data-baseweb="tab"] {color: #2e8b45; letter-spacing: 1px;}
.stTabs [aria-selected="true"] {color: #00ff41 !important;}
.stTabs [data-baseweb="tab-highlight"] {background-color: #00ff41;}
div[data-testid="stExpander"] {border: 1px solid #0f3d1a; border-radius: 2px;}
code, pre {color: #00ff41 !important;}
</style>
<div class="credit">MADE BY SALAH EDDINE AHADAF</div>
<div class="prompt">root@ml-studio:~$ <span class="cmd">./launch --mode=pro</span><span class="cursor">█</span></div>
<h1 class="title">ML_STUDIO<span class="dim">//PRO</span></h1>
<div class="sub">load · configure · train · benchmark · predict &mdash; no cloud, no fluff.</div>
""", unsafe_allow_html=True)

GREENS = ["#00ff41", "#39ff14", "#00cc33", "#7CFC00", "#00ffaa", "#008f11", "#adff2f", "#00e5a0"]
SCALE = [[0, "#000000"], [1, "#00ff41"]]
DIV = [[0, "#ff0033"], [0.5, "#000000"], [1, "#00ff41"]]


def show(fig, h=420):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#000", plot_bgcolor="#020802", height=h,
                      font=dict(family="monospace", color="#00ff41"), colorway=GREENS,
                      margin=dict(l=10, r=10, t=45, b=10))
    fig.update_xaxes(gridcolor="#0d2b12", zerolinecolor="#0d2b12")
    fig.update_yaxes(gridcolor="#0d2b12", zerolinecolor="#0d2b12")
    st.plotly_chart(fig)


# ============================== STATE ==============================
ss = st.session_state
ss.setdefault("log", [])
ss.setdefault("runs", {})
ss.setdefault("active", None)


def log(msg):
    ss.log.append(f"[{datetime.now():%H:%M:%S}] {msg}")


# ============================== MODEL REGISTRY ==============================
# spec: ("i", min, max, default) | ("f", min, max, default) | ("s", [options])
FOREST = {"n_estimators": ("i", 10, 500, 200), "max_depth": ("i", 2, 30, 12), "min_samples_split": ("i", 2, 20, 2)}
BOOST = {"n_estimators": ("i", 10, 500, 100), "learning_rate": ("f", 0.01, 1.0, 0.1), "max_depth": ("i", 1, 10, 3)}
KNN = {"n_neighbors": ("i", 1, 50, 5), "weights": ("s", ["uniform", "distance"])}
TREE = {"max_depth": ("i", 1, 30, 6), "min_samples_split": ("i", 2, 20, 2)}
MODELS = {
    "Classification": {
        "Random Forest": (RandomForestClassifier, FOREST),
        "Extra Trees": (ExtraTreesClassifier, FOREST),
        "Gradient Boosting": (GradientBoostingClassifier, BOOST),
        "SVM": (SVC, {"C": ("f", 0.1, 20.0, 1.0), "kernel": ("s", ["rbf", "linear", "poly"])}),
        "Logistic Regression": (LogisticRegression, {"C": ("f", 0.01, 20.0, 1.0)}),
        "KNN": (KNeighborsClassifier, KNN),
        "Decision Tree": (DecisionTreeClassifier, TREE),
    },
    "Regression": {
        "Random Forest": (RandomForestRegressor, FOREST),
        "Gradient Boosting": (GradientBoostingRegressor, BOOST),
        "SVR": (SVR, {"C": ("f", 0.1, 20.0, 1.0), "kernel": ("s", ["rbf", "linear", "poly"])}),
        "Linear Regression": (LinearRegression, {}),
        "Ridge": (Ridge, {"alpha": ("f", 0.01, 100.0, 1.0)}),
        "Lasso": (Lasso, {"alpha": ("f", 0.001, 10.0, 0.1)}),
        "KNN": (KNeighborsRegressor, KNN),
        "Decision Tree": (DecisionTreeRegressor, TREE),
    },
    "Clustering": {
        "K-Means": (KMeans, {"n_clusters": ("i", 2, 15, 3)}),
        "Agglomerative": (AgglomerativeClustering, {"n_clusters": ("i", 2, 15, 3),
                                                    "linkage": ("s", ["ward", "complete", "average"])}),
        "DBSCAN": (DBSCAN, {"eps": ("f", 0.1, 10.0, 1.0), "min_samples": ("i", 2, 30, 5)}),
        "Gaussian Mixture": (GaussianMixture, {"n_components": ("i", 2, 15, 3)}),
    },
}
PRIMARY = {"Classification": "F1", "Regression": "R2", "Clustering": "Silhouette"}
PCT = {"Accuracy", "Precision", "Recall", "F1"}
SCALERS = {"Standard": StandardScaler, "MinMax": MinMaxScaler, "Robust": RobustScaler, "None": lambda: "passthrough"}


def default_hp(task, name):
    return {p: (s[3] if s[0] != "s" else s[1][0]) for p, s in MODELS[task][name][1].items()}


def hp_ui(task, name):
    specs, hp = MODELS[task][name][1], {}
    if not specs:
        st.caption("// no tunable hyperparameters")
        return hp
    cols = st.columns(min(len(specs), 3))
    for i, (p, s) in enumerate(specs.items()):
        k = f"hp-{task}-{name}-{p}"
        with cols[i % len(cols)]:
            if s[0] == "i":
                hp[p] = st.slider(p, s[1], s[2], s[3], key=k)
            elif s[0] == "f":
                hp[p] = st.slider(p, float(s[1]), float(s[2]), float(s[3]), key=k)
            else:
                hp[p] = st.selectbox(p, s[1], key=k)
    return hp


def make_model(task, name, hp, seed):
    cls = MODELS[task][name][0]
    kw = dict(hp)
    if "random_state" in cls().get_params():
        kw["random_state"] = seed
    if cls is LogisticRegression:
        kw["max_iter"] = 2000
    return cls(**kw)


# ============================== DATA / PIPELINE ==============================
DEMOS = {"Iris (classification)": datasets.load_iris, "Wine (classification)": datasets.load_wine,
         "Breast Cancer (classification)": datasets.load_breast_cancer,
         "Diabetes (regression)": datasets.load_diabetes}


@st.cache_data(show_spinner=False)
def load_demo(name):
    return DEMOS[name](as_frame=True).frame


@st.cache_data(show_spinner=False)
def read_upload(name, data):
    buf = io.BytesIO(data)
    return pd.read_csv(buf) if name.lower().endswith(".csv") else pd.read_excel(buf)


def prepare(df, cfg):
    d = df.drop_duplicates() if cfg["dedupe"] else df
    tgt = cfg["target"]
    d = d[cfg["feats"] + ([tgt] if tgt else [])]
    if tgt:
        d = d.dropna(subset=[tgt])
    X = d[cfg["feats"]].copy()
    num = X.select_dtypes(include="number").columns
    if cfg["num_imp"] == "drop rows" and len(num):
        d = d[X[num].notna().all(axis=1)]
        X = X.loc[d.index]
    for c in X.columns.difference(num):  # normalise categoricals: str values, real NaN for missing
        s = X[c]
        X[c] = s.astype(str).astype(object).where(s.notna(), np.nan)
    if len(X) < 10:
        raise ValueError("fewer than 10 usable rows after cleaning")
    y, classes = None, None
    if tgt:
        y = d[tgt]
        if cfg["task"] == "Classification":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
            classes = [str(c) for c in le.classes_]
    return X, y, classes


def get_split(df, cfg):
    X, y, classes = prepare(df, cfg)
    if y is None:
        return dict(X_tr=X, X_te=X, y_tr=None, y_te=None, classes=None)
    strat = y if (cfg["task"] == "Classification" and cfg["stratify"] and y.value_counts().min() >= 2) else None
    try:
        parts = train_test_split(X, y, test_size=cfg["test_size"], random_state=cfg["seed"], stratify=strat)
    except ValueError:
        parts = train_test_split(X, y, test_size=cfg["test_size"], random_state=cfg["seed"])
    return dict(X_tr=parts[0], X_te=parts[1], y_tr=parts[2], y_te=parts[3], classes=classes)


def build_pipe(X, est, cfg):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    imp = "median" if cfg["num_imp"] == "drop rows" else cfg["num_imp"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy=imp)), ("sc", SCALERS[cfg["scaler"]]())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])
    return Pipeline([("pre", pre), ("model", est)])


def score(task, split, y_pred, pipe):
    if task == "Classification":
        yt = split["y_te"]
        return {"Accuracy": accuracy_score(yt, y_pred),
                "Precision": precision_score(yt, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(yt, y_pred, average="weighted", zero_division=0),
                "F1": f1_score(yt, y_pred, average="weighted", zero_division=0)}
    if task == "Regression":
        yt = split["y_te"]
        return {"R2": r2_score(yt, y_pred), "RMSE": float(np.sqrt(mean_squared_error(yt, y_pred))),
                "MAE": mean_absolute_error(yt, y_pred)}
    Xt = pipe[:-1].transform(split["X_te"])
    y_pred = np.asarray(y_pred)
    m = y_pred != -1
    out = {"Clusters": int(len(set(y_pred[m]))), "Noise": int((~m).sum())}
    try:
        out["Silhouette"] = silhouette_score(Xt[m], y_pred[m], sample_size=min(5000, int(m.sum())), random_state=0)
        out["Davies-Bouldin"] = davies_bouldin_score(Xt[m], y_pred[m])
        out["Calinski-Harabasz"] = calinski_harabasz_score(Xt[m], y_pred[m])
    except ValueError:
        out.update({"Silhouette": np.nan, "Davies-Bouldin": np.nan, "Calinski-Harabasz": np.nan})
    return out


def run_model(split, cfg, name, hp, cv=False):
    t0, task = time.time(), cfg["task"]
    pipe = build_pipe(split["X_tr"], make_model(task, name, hp, cfg["seed"]), cfg)
    res = dict(name=name, hp=hp, task=task, cfg=cfg, split=split, perm=None, cv=None)
    if cv:
        try:
            res["cv"] = cross_val_score(clone(pipe), split["X_tr"], split["y_tr"], cv=5,
                                        scoring="f1_weighted" if task == "Classification" else "r2")
        except Exception as e:
            log(f"CV skipped: {e}")
    if task == "Clustering":
        y_pred = pipe.fit_predict(split["X_tr"])
    else:
        pipe.fit(split["X_tr"], split["y_tr"])
        y_pred = pipe.predict(split["X_te"])
    res.update(pipe=pipe, y_pred=np.asarray(y_pred), secs=time.time() - t0)
    res["metrics"] = score(task, split, y_pred, pipe)
    return res


def reset_if_cfg_changed(cfg):
    if any(r["cfg"] != cfg for r in ss.runs.values()):
        ss.runs.clear()
        log("config changed -> previous runs cleared")


def fmt(k, v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if pd.isna(v):
        return "n/a"
    return f"{v:.2%}" if k in PCT else f"{v:.4f}"


def leaderboard():
    runs = list(ss.runs.values())
    p = PRIMARY[runs[0]["task"]]
    t = pd.DataFrame([{"Model": r["name"], **r["metrics"], "Time(s)": round(r["secs"], 2)} for r in runs])
    return t.sort_values(p, ascending=False, na_position="last").reset_index(drop=True), p


# ============================== TABS ==============================
t_data, t_cfg, t_train, t_eval, t_pred = st.tabs(
    ["01 // DATA", "02 // CONFIG", "03 // TRAIN", "04 // EVALUATE", "05 // PREDICT"])

# ---------- 01 DATA ----------
df, ds_id = None, None
with t_data:
    src = st.radio("SOURCE", ["Demo dataset", "Upload CSV/XLSX"], horizontal=True, key="src")
    if src == "Demo dataset":
        ds_id = st.selectbox("DATASET", list(DEMOS), key="demo")
        df = load_demo(ds_id)
    else:
        up = st.file_uploader("DROP FILE", type=["csv", "xlsx"], key="upload")
        if up is not None:
            try:
                df, ds_id = read_upload(up.name, up.getvalue()), f"{up.name}-{up.size}"
            except Exception as e:
                st.error(f"parse error: {e}")
    if ss.get("ds_id") != ds_id:
        ss.ds_id = ds_id
        ss.runs, ss.active = {}, None
        if ds_id:
            log(f"dataset loaded: {ds_id}")
    if df is None:
        st.info("// awaiting input...")
    else:
        c = st.columns(5)
        c[0].metric("ROWS", f"{len(df):,}")
        c[1].metric("COLUMNS", df.shape[1])
        c[2].metric("MISSING", f"{int(df.isna().sum().sum()):,}")
        c[3].metric("DUPLICATES", f"{int(df.duplicated().sum()):,}")
        c[4].metric("MEMORY", f"{df.memory_usage(deep=True).sum() / 1024:.0f} KB")
        st.dataframe(df.head(15))
        with st.expander("STATISTICS"):
            st.dataframe(df.describe(include="all").T.astype(str))
        a, b = st.columns(2)
        with a:
            nulls = df.isna().sum()
            nulls = nulls[nulls > 0]
            if nulls.empty:
                st.success("// no missing values")
            else:
                show(px.bar(x=nulls.values, y=nulls.index, orientation="h", title="MISSING VALUES",
                            labels={"x": "count", "y": ""}), 320)
            col = st.selectbox("INSPECT COLUMN", df.columns, key="inspect")
            if pd.api.types.is_numeric_dtype(df[col]):
                show(px.histogram(df, x=col, nbins=30, title=f"DISTRIBUTION: {col}"), 320)
            else:
                vc = df[col].astype(str).value_counts().head(20)
                show(px.bar(x=vc.index, y=vc.values, title=f"TOP VALUES: {col}", labels={"x": "", "y": "count"}), 320)
        with b:
            numdf = df.select_dtypes(include="number")
            if numdf.shape[1] >= 2:
                corr = numdf.corr()
                show(px.imshow(corr, text_auto=".2f" if corr.shape[0] <= 12 else False, zmin=-1, zmax=1,
                               color_continuous_scale=DIV, title="CORRELATION MATRIX"), 480)
            else:
                st.caption("// need ≥2 numeric columns for correlation")

# ---------- 02 CONFIG ----------
cfg = None
with t_cfg:
    if df is None:
        st.warning("// load data first")
    else:
        c1, c2 = st.columns(2)
        cols = df.columns.tolist()
        with c1:
            task = st.selectbox("TASK", list(MODELS), key="task")
            target = None
            if task != "Clustering":
                target = st.selectbox("TARGET (y)", cols, index=len(cols) - 1, key="target")
            pool = [c for c in cols if c != target]
            feats = st.multiselect("FEATURES", pool, default=pool, key=f"feats-{ds_id}-{task}-{target}")
        with c2:
            num_imp = st.selectbox("NUMERIC IMPUTER", ["mean", "median", "most_frequent", "drop rows"], key="imp")
            scaler = st.selectbox("SCALER", list(SCALERS), key="scaler")
            dedupe = st.checkbox("DROP DUPLICATE ROWS", key="dedupe")
            test_size, stratify = 0.2, False
            if task != "Clustering":
                test_size = st.slider("TEST SIZE", 0.1, 0.5, 0.2, 0.05, key="test_size")
                if task == "Classification":
                    stratify = st.checkbox("STRATIFIED SPLIT", value=True, key="stratify")
            seed = int(st.number_input("RANDOM SEED", 0, 10**6, 42, key="seed"))
        ok = True
        if not feats:
            st.error("// select at least one feature")
            ok = False
        if task == "Regression" and not pd.api.types.is_numeric_dtype(df[target]):
            st.error("// regression needs a numeric target")
            ok = False
        if task == "Classification" and df[target].nunique() > 30:
            st.warning(f"// target has {df[target].nunique()} unique values — is this really classification?")
        if ok:
            cfg = dict(task=task, target=target, feats=feats, num_imp=num_imp, scaler=scaler, dedupe=dedupe,
                       test_size=test_size, stratify=stratify, seed=seed)
            st.code(f"PIPELINE  impute({num_imp}) -> scale({scaler}) -> one-hot(cat) -> model\n"
                    f"TASK      {task}   TARGET {target}   FEATURES {len(feats)}   SEED {seed}", language="text")

# ---------- 03 TRAIN ----------
with t_train:
    if cfg is None:
        st.warning("// complete the config first")
    else:
        task = cfg["task"]
        mode = st.radio("MODE", ["Single model", "Benchmark all models"], horizontal=True, key="mode")
        if mode == "Single model":
            name = st.selectbox("ALGORITHM", list(MODELS[task]), key=f"algo-{task}")
            hp = hp_ui(task, name)
            cv = st.checkbox("5-FOLD CROSS-VALIDATION", key="cv") if task != "Clustering" else False
            if st.button("▶ TRAIN", type="primary", key="btn_train"):
                try:
                    reset_if_cfg_changed(cfg)
                    with st.spinner(f"training {name}..."):
                        res = run_model(get_split(df, cfg), cfg, name, hp, cv)
                    ss.runs[name], ss.active = res, name
                    log(f"trained {name} in {res['secs']:.2f}s")
                    st.success(f"{name} trained in {res['secs']:.2f}s")
                except Exception as e:
                    log(f"ERROR {name}: {e}")
                    st.error(f"training failed: {e}")
        else:
            st.caption("// trains every algorithm with default hyperparameters and ranks them")
            if st.button("▶ RUN BENCHMARK", type="primary", key="btn_bench"):
                try:
                    reset_if_cfg_changed(cfg)
                    split, names, fails = get_split(df, cfg), list(MODELS[task]), []
                    bar = st.progress(0.0, text="booting...")
                    for i, n in enumerate(names):
                        bar.progress(i / len(names), text=f"training {n}...")
                        try:
                            ss.runs[n] = run_model(split, cfg, n, default_hp(task, n))
                        except Exception as e:
                            fails.append(f"{n}: {e}")
                    bar.progress(1.0, text="done")
                    if ss.runs:
                        ss.active = leaderboard()[0].iloc[0]["Model"]
                        log(f"benchmark finished, best: {ss.active}")
                    for f in fails:
                        st.warning(f"skipped -> {f}")
                except Exception as e:
                    log(f"ERROR benchmark: {e}")
                    st.error(f"benchmark failed: {e}")
        if ss.runs:
            st.markdown("#### LEADERBOARD")
            lb, p = leaderboard()
            st.dataframe(lb)
            if len(lb) > 1:
                show(px.bar(lb, x="Model", y=p, title=f"RANKING BY {p.upper()}"), 340)

# ---------- 04 EVALUATE ----------
with t_eval:
    if not ss.runs:
        st.warning("// no trained models yet")
    else:
        names = list(ss.runs)
        sel = st.selectbox("MODEL", names, index=names.index(ss.active) if ss.active in names else 0, key="eval_run")
        res = ss.runs[sel]
        task, pipe, split, y_pred = res["task"], res["pipe"], res["split"], res["y_pred"]
        X_te, y_te, classes = split["X_te"], split["y_te"], split["classes"]
        items = list(res["metrics"].items())
        mc = st.columns(min(len(items) + (1 if res["cv"] is not None else 0), 7))
        for i, (k, v) in enumerate(items):
            mc[i % len(mc)].metric(k.upper(), fmt(k, v))
        if res["cv"] is not None:
            mc[len(items) % len(mc)].metric("CV MEAN ± STD", f"{res['cv'].mean():.3f} ± {res['cv'].std():.3f}")

        if task == "Classification":
            a, b = st.columns(2)
            with a:
                idx = list(range(len(classes)))
                cm = confusion_matrix(y_te, y_pred, labels=idx)
                show(px.imshow(cm, x=classes, y=classes, text_auto=True, color_continuous_scale=SCALE,
                               labels=dict(x="Predicted", y="Actual", color="Count"), title="CONFUSION MATRIX"), 420)
            with b:
                if len(classes) == 2 and (hasattr(pipe, "predict_proba") or hasattr(pipe, "decision_function")):
                    s = pipe.predict_proba(X_te)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X_te)
                    fpr, tpr, _ = roc_curve(y_te, s)
                    fig = go.Figure([go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC {auc(fpr, tpr):.3f}",
                                                line=dict(width=3)),
                                     go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                                                line=dict(dash="dot", color="#1f6b34"))])
                    fig.update_layout(title="ROC CURVE", xaxis_title="FPR", yaxis_title="TPR")
                    show(fig, 420)
                else:
                    st.caption("// ROC curve is shown for binary problems only")
            rep = classification_report(y_te, y_pred, labels=list(range(len(classes))), target_names=classes,
                                        output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(rep).T.round(3))

        elif task == "Regression":
            a, b = st.columns(2)
            with a:
                lo, hi = float(min(y_te.min(), y_pred.min())), float(max(y_te.max(), y_pred.max()))
                fig = go.Figure([go.Scatter(x=y_te, y=y_pred, mode="markers", name="samples", opacity=0.7),
                                 go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="ideal",
                                            line=dict(dash="dot", color="#1f6b34"))])
                if len(y_te) > 1 and np.ptp(y_te) > 0:
                    m, c0 = np.polyfit(y_te, y_pred, 1)
                    fig.add_trace(go.Scatter(x=[lo, hi], y=[m * lo + c0, m * hi + c0], mode="lines", name="fit",
                                             line=dict(color="#39ff14", width=2)))
                fig.update_layout(title="ACTUAL vs PREDICTED", xaxis_title="actual", yaxis_title="predicted")
                show(fig, 420)
            with b:
                resid = np.asarray(y_te) - y_pred
                show(px.scatter(x=y_pred, y=resid, labels={"x": "predicted", "y": "residual"},
                                title="RESIDUALS", opacity=0.7).add_hline(y=0, line_dash="dot", line_color="#1f6b34"), 420)
            show(px.histogram(x=resid, nbins=30, title="RESIDUAL DISTRIBUTION", labels={"x": "residual"}), 300)

        else:
            Xt = pipe[:-1].transform(X_te)
            labels = pd.Series(y_pred).map(lambda v: "noise" if v == -1 else f"C{v}")
            a, b = st.columns([2, 1])
            with a:
                n = min(3, Xt.shape[1])
                if n < 2:
                    st.caption("// need ≥2 encoded features for PCA view")
                else:
                    dim = st.radio("PROJECTION", ["2D", "3D"] if n == 3 else ["2D"], horizontal=True, key="proj")
                    comp = PCA(n_components=3 if dim == "3D" else 2).fit_transform(Xt)
                    pdf = pd.DataFrame(comp, columns=[f"PC{i + 1}" for i in range(comp.shape[1])])
                    pdf["cluster"] = labels.values
                    fig = (px.scatter_3d(pdf, x="PC1", y="PC2", z="PC3", color="cluster") if dim == "3D"
                           else px.scatter(pdf, x="PC1", y="PC2", color="cluster"))
                    fig.update_layout(title="CLUSTERS (PCA)")
                    show(fig, 480)
            with b:
                vc = labels.value_counts().sort_index()
                show(px.bar(x=vc.index, y=vc.values, labels={"x": "", "y": "points"}, title="CLUSTER SIZES"), 480)
            if res["name"] == "K-Means":
                with st.expander("K-SWEEP (elbow + silhouette)"):
                    if st.button("RUN SWEEP 2..10", key="btn_sweep"):
                        ks, inert, sil = list(range(2, 11)), [], []
                        for k in ks:
                            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xt)
                            inert.append(km.inertia_)
                            sil.append(silhouette_score(Xt, km.labels_, sample_size=min(3000, len(Xt)), random_state=0))
                        s1, s2 = st.columns(2)
                        with s1:
                            show(px.line(x=ks, y=inert, markers=True, title="INERTIA (ELBOW)",
                                         labels={"x": "k", "y": "inertia"}), 300)
                        with s2:
                            show(px.line(x=ks, y=sil, markers=True, title="SILHOUETTE",
                                         labels={"x": "k", "y": "score"}), 300)

        if task != "Clustering":
            st.markdown("#### FEATURE IMPORTANCE (permutation)")
            if st.button("COMPUTE", key=f"btn_perm-{sel}"):
                with st.spinner("shuffling features..."):
                    n = min(len(X_te), 500)
                    ix = np.random.RandomState(0).choice(len(X_te), n, replace=False)
                    r = permutation_importance(pipe, X_te.iloc[ix], y_te.iloc[ix], n_repeats=5, random_state=0,
                                               scoring="f1_weighted" if task == "Classification" else "r2")
                    res["perm"] = pd.DataFrame({"feature": X_te.columns, "importance": r.importances_mean}) \
                        .sort_values("importance")
            if res["perm"] is not None:
                show(px.bar(res["perm"], x="importance", y="feature", orientation="h",
                            title="DROP IN SCORE WHEN SHUFFLED"), max(300, 28 * len(res["perm"]) + 100))

        st.markdown("#### EXPORT")
        out = X_te.copy()
        if task == "Classification":
            out["actual"] = [classes[i] for i in y_te]
            out["predicted"] = [classes[i] for i in y_pred]
        elif task == "Regression":
            out["actual"], out["predicted"] = y_te, y_pred
        else:
            out["cluster"] = y_pred
        buf = io.BytesIO()
        joblib.dump({"pipeline": pipe, "classes": classes, "features": res["cfg"]["feats"],
                     "target": res["cfg"]["target"], "task": task}, buf)
        d1, d2 = st.columns(2)
        d1.download_button("⬇ PREDICTIONS.CSV", out.to_csv(index=False).encode(), f"predictions_{sel}.csv", "text/csv")
        d2.download_button("⬇ MODEL.JOBLIB", buf.getvalue(), f"model_{sel}.joblib")

# ---------- 05 PREDICT ----------
with t_pred:
    if not ss.runs:
        st.warning("// no trained models yet")
    else:
        names = list(ss.runs)
        psel = st.selectbox("MODEL", names, index=names.index(ss.active) if ss.active in names else 0, key="pred_run")
        res = ss.runs[psel]
        Xtr, feats = res["split"]["X_tr"], res["cfg"]["feats"]
        with st.form("predict_form"):
            vals, fc = {}, st.columns(3)
            for i, f in enumerate(feats):
                s = Xtr[f]
                with fc[i % 3]:
                    if pd.api.types.is_numeric_dtype(s):
                        vals[f] = st.number_input(f, value=float(s.median()) if s.notna().any() else 0.0,
                                                  key=f"in-{psel}-{f}")
                    else:
                        vals[f] = st.selectbox(f, s.dropna().astype(str).unique().tolist() or [""], key=f"in-{psel}-{f}")
            go_btn = st.form_submit_button("▶ PREDICT")
        if go_btn:
            row = pd.DataFrame([vals])[feats]
            try:
                if res["task"] == "Classification":
                    cl = res["split"]["classes"]
                    st.metric("PREDICTED CLASS", cl[int(res["pipe"].predict(row)[0])])
                    if hasattr(res["pipe"], "predict_proba"):
                        pr = res["pipe"].predict_proba(row)[0]
                        show(px.bar(x=cl, y=pr, labels={"x": "class", "y": "probability"}, title="CLASS PROBABILITIES"), 300)
                elif res["task"] == "Regression":
                    st.metric("PREDICTED VALUE", f"{float(res['pipe'].predict(row)[0]):.4f}")
                elif hasattr(res["pipe"], "predict"):
                    st.metric("ASSIGNED CLUSTER", f"C{int(res['pipe'].predict(row)[0])}")
                else:
                    st.info("// this algorithm cannot assign new points to clusters")
            except Exception as e:
                st.error(f"prediction failed: {e}")

# ============================== SIDEBAR ==============================
with st.sidebar:
    st.markdown("### SYSTEM")
    st.code(f"dataset : {ss.get('ds_id') or '-'}\nrows    : {len(df) if df is not None else '-'}\n"
            f"task    : {cfg['task'] if cfg else '-'}\nmodels  : {len(ss.runs)}\nbest    : {ss.active or '-'}",
            language="text")
    st.markdown("### LOG")
    st.code("\n".join(ss.log[-14:]) or "// idle", language="text")
    if st.button("RESET SESSION", key="reset"):
        ss.runs, ss.active, ss.log = {}, None, []
        st.rerun()