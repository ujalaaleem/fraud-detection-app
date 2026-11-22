import io
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
SEED_DEFAULT = 42

st.set_page_config(page_title="FraudScope | ML Pipeline", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.chip {display:inline-block; padding:6px 12px; margin:3px 2px; border-radius:999px;
       background:#eef2ff; color:#3730a3; font-size:0.85rem; font-weight:600;}
.arrow {color:#9ca3af; font-weight:700;}
.card {border:1px solid #e5e7eb; border-radius:12px; padding:14px 18px; background:#fafafa;}
.good {color:#15803d; font-weight:700;}
.bad {color:#b91c1c; font-weight:700;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def make_sample(n=1000, fraud_rate=0.25, seed=SEED_DEFAULT):
    """Synthetic credit card transactions (classes overlap a little, like real data)."""
    rng = np.random.default_rng(seed)
    n_f = int(n * fraud_rate)
    n_l = n - n_f
    legit = pd.DataFrame({
        "amount": rng.gamma(2.0, 30, n_l).round(2),
        "hour_of_day": rng.choice(range(7, 23), n_l),
        "distance_from_home_km": rng.gamma(2.0, 5, n_l).round(1),
        "transactions_last_24h": rng.poisson(1.5, n_l),
        "merchant_risk_score": rng.beta(2, 8, n_l).round(3),
        "is_fraud": 0})
    fraud = pd.DataFrame({
        "amount": rng.gamma(4.0, 120, n_f).round(2),
        "hour_of_day": rng.choice([0, 1, 2, 3, 4, 5, 22, 23], n_f),
        "distance_from_home_km": rng.gamma(4.0, 90, n_f).round(1),
        "transactions_last_24h": rng.poisson(6, n_f),
        "merchant_risk_score": rng.beta(6, 3, n_f).round(3),
        "is_fraud": 1})
    k = int(0.20 * n_f)
    fi = fraud.index[:k]
    fraud.loc[fi, "amount"] = rng.gamma(2.5, 45, k).round(2)
    fraud.loc[fi, "distance_from_home_km"] = rng.gamma(2.5, 12, k).round(1)
    fraud.loc[fi, "transactions_last_24h"] = rng.poisson(2.5, k)
    fraud.loc[fi, "merchant_risk_score"] = rng.beta(3, 6, k).round(3)
    j = int(0.06 * n_l)
    li = legit.index[:j]
    legit.loc[li, "amount"] = rng.gamma(3.0, 110, j).round(2)
    legit.loc[li, "distance_from_home_km"] = rng.gamma(3.0, 60, j).round(1)
    legit.loc[li, "merchant_risk_score"] = rng.beta(4, 4, j).round(3)
    return pd.concat([legit, fraud]).sample(frac=1, random_state=seed).reset_index(drop=True)


SAMPLE_RECORDS = pd.DataFrame([
    [12.5, 14, 2, 1, 0.10, 0],
    [35.0, 19, 5, 2, 0.20, 0],
    [60.0, 11, 8, 1, 0.15, 0],
    [950.0, 3, 420, 8, 0.90, 1],
    [1200.0, 2, 380, 9, 0.85, 1],
    [15.0, 13, 3, 1, 0.10, 0],
    [800.0, 4, 500, 7, 0.95, 1],
    [45.0, 18, 10, 2, 0.25, 0],
], columns=["amount", "hour_of_day", "distance_from_home_km",
           "transactions_last_24h", "merchant_risk_score", "expected"])


def prepare(df, label):
    """Clean the data, turn the label into 0/1, encode text columns."""
    d = df.dropna().reset_index(drop=True)
    y = d[label]
    if not pd.api.types.is_numeric_dtype(y):
        m = {"yes": 1, "no": 0, "true": 1, "false": 0, "positive": 1, "negative": 0,
             "fraud": 1, "legit": 0, "spam": 1, "ham": 0, "1": 1, "0": 0}
        mapped = y.astype(str).str.strip().str.lower().map(m)
        if mapped.isnull().any():
            codes, uniques = pd.factorize(y)
            if len(uniques) != 2:
                raise ValueError("The label column must have exactly two classes.")
            mapped = pd.Series(codes, index=y.index)
        y = mapped
    y = y.astype(int)
    if sorted(y.unique().tolist()) != [0, 1]:
        raise ValueError("The label column must contain exactly two classes (0 and 1).")
    X = d.drop(columns=[label])
    X = pd.get_dummies(X, drop_first=True).astype(float)
    return X, y


def make_models(n_trees, max_depth, svm_c, balanced, seed):
    cw = "balanced" if balanced else None
    return {
        "Random Forest": RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth,
                                                class_weight=cw, random_state=seed),
        "SVM": SVC(kernel="rbf", C=svm_c, probability=True, class_weight=cw, random_state=seed),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=cw),
    }


def metric_row(y_true, y_pred):
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0)}


@st.cache_data(show_spinner="Running the pipeline (clustering + training)...")
def run_pipeline(X, y, test_size, seed, balanced, n_trees, max_depth, svm_c):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(Xtr)                 # learn scale from TRAIN only
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    # K-Means (no labels used)
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xtr_s)
    tr_c, te_c = km.labels_, km.predict(Xte_s)
    flip = accuracy_score(ytr, tr_c) < 0.5             # cluster ids are arbitrary
    tr_km = 1 - tr_c if flip else tr_c
    te_km = 1 - te_c if flip else te_c
    sample_size = min(5000, len(Xtr_s))

    km_info = {
        "metrics": metric_row(yte, te_km),
        "ari": adjusted_rand_score(ytr, tr_km),
        "silhouette": silhouette_score(Xtr_s, tr_km, sample_size=sample_size, random_state=seed),
        "cm": confusion_matrix(yte, te_km, labels=[0, 1]),
        "flipped": bool(flip),
    }
    pca = PCA(n_components=2, random_state=seed).fit(Xtr_s)
    pca_xy = pca.transform(Xtr_s)

    # Train models: A = true labels, B = K-Means cluster labels. Always TEST on true labels.
    experiments = {"A: true labels": ytr.values, "B: cluster labels": tr_km}
    models = make_models(n_trees, max_depth, svm_c, balanced, seed)
    rows, fitted = [], {}
    for mname, m in models.items():
        for ename, y_fit in experiments.items():
            model = clone(m).fit(Xtr_s, y_fit)          # TRAINING
            pred = model.predict(Xte_s)                 # TESTING (true labels used to score)
            key = f"{mname} | {ename}"
            fitted[key] = model
            rows.append({"Model": mname, "Trained on": ename, **metric_row(yte, pred), "key": key})
    results = pd.DataFrame(rows)

    return dict(Xtr=Xtr, Xte=Xte, ytr=ytr.values, yte=yte.values, Xtr_s=Xtr_s, Xte_s=Xte_s,
                scaler=scaler, tr_km=tr_km, km=km_info, pca_xy=pca_xy, results=results,
                fitted=fitted, features=list(X.columns))


@st.cache_data(show_spinner="Running k-fold cross-validation...")
def run_kfold(X, y, model_name, n_trees, max_depth, svm_c, balanced, n_folds, seed):
    model = make_models(n_trees, max_depth, svm_c, balanced, seed)[model_name]
    pipe = make_pipeline(StandardScaler(), model)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    s = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "precision", "recall", "f1"])
    return pd.DataFrame({
        "Fold": range(1, n_folds + 1),
        "Accuracy": s["test_accuracy"], "Precision": s["test_precision"],
        "Recall": s["test_recall"], "F1": s["test_f1"]})


def shifted_table(model, Xte_s, yte, seed):
    """Same model, different kinds of data: shows why precision/recall move."""
    rng = np.random.default_rng(seed)
    yv = np.asarray(yte)
    pos, neg = np.where(yv == 1)[0], np.where(yv == 0)[0]

    def resample(rate, n=400):
        n_pos = int(n * rate)
        idx = np.concatenate([rng.choice(pos, n_pos, replace=True),
                              rng.choice(neg, n - n_pos, replace=True)])
        return Xte_s[idx], yv[idx]

    scen = {"Original test set": (Xte_s, yv, "Baseline: same mix as training data.")}
    x, yy = resample(0.03)
    scen["Rare positives (3%)"] = (x, yy, "Very few real cases, so false alarms outnumber hits and precision tends to drop.")
    x, yy = resample(0.65)
    scen["Positive wave (65%)"] = (x, yy, "Many real cases, so almost every alert is correct and precision tends to rise.")
    x, yy = resample(0.25)
    flip = rng.random(len(yy)) < 0.15
    scen["Noisy labels (15% wrong)"] = (x, np.where(flip, 1 - yy, yy), "Some 'ground truth' is wrong, so both precision and recall fall.")
    scen["Feature drift (noise added)"] = (Xte_s + rng.normal(0, 0.8, Xte_s.shape), yv, "Customer behaviour changes, so the model sees patterns it has not learned.")

    rows = []
    for name, (xs, ys, why) in scen.items():
        pr = model.predict(xs)
        rows.append({"Scenario": name,
                     "Precision": precision_score(ys, pr, zero_division=0),
                     "Recall": recall_score(ys, pr, zero_division=0),
                     "F1": f1_score(ys, pr, zero_division=0),
                     "Why it changes": why})
    return pd.DataFrame(rows)


def gen_records(X, y, n, seed):
    """Auto-generate new records from each class's distribution (not copies of real rows)."""
    rng = np.random.default_rng(seed)
    out, exp = [], []
    for cls, cnt in [(1, n // 2), (0, n - n // 2)]:
        sub = X[y.values == cls]
        mu, sd = sub.mean(), sub.std().fillna(0) + 1e-6
        for _ in range(cnt):
            row = rng.normal(mu.values, sd.values * 0.8)
            row = np.clip(row, X.min().values, X.max().values)
            out.append(np.round(row, 2))
            exp.append(cls)
    rec = pd.DataFrame(out, columns=X.columns)
    rec["expected"] = exp
    return rec.sample(frac=1, random_state=seed).reset_index(drop=True)


def plot_cm(cm, names, title):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(names); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.tight_layout()
    return fig


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🛡️ FraudScope: End-to-End ML Pipeline")
st.caption("Load data → clustering → train a model → evaluate → test on new records. "
           "Every step of the assignment, in order.")

STEPS = ["Load data", "Preprocess", "K-Means (k=2)", "Train/Test split",
         "Train models", "Evaluate", "New records"]
st.markdown(" <span class='arrow'>➜</span> ".join(
    f"<span class='chip'>{i + 1}. {s}</span>" for i, s in enumerate(STEPS)), unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Sidebar settings
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    source = st.radio("Dataset", ["Built-in sample (1,000 rows)", "Upload CSV"])
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("CSV file", type=["csv"])
    test_pct = st.slider("Test set size (%)", 10, 40, 25, 5)
    n_folds = st.slider("k-fold (folds)", 3, 10, 5)
    seed = st.number_input("Random seed", 0, 9999, SEED_DEFAULT)
    balanced = st.checkbox("Balance classes (class weights)", value=False)
    with st.expander("Model settings"):
        n_trees = st.slider("Random Forest trees", 10, 300, 60, 10)
        max_depth = st.slider("Random Forest max depth", 2, 20, 8)
        svm_c = st.slider("SVM C", 0.1, 10.0, 1.0, 0.1)

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
if source == "Upload CSV":
    if uploaded is None:
        st.info("Upload a CSV in the sidebar, or switch to the built-in sample dataset.")
        st.stop()
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = make_sample()

if len(df_raw) > 20000:
    st.warning("Large file: using a random 20,000-row sample to keep the app fast.")
    df_raw = df_raw.sample(20000, random_state=int(seed)).reset_index(drop=True)

with st.sidebar:
    cols = list(df_raw.columns)
    default_idx = cols.index("is_fraud") if "is_fraud" in cols else len(cols) - 1
    label = st.selectbox("Label column (0/1 class)", cols, index=default_idx)

try:
    X, y = prepare(df_raw, label)
except Exception as e:
    st.error(f"Could not prepare the data: {e}")
    st.stop()

pos_name = "Fraud" if "fraud" in label.lower() else "Positive"
neg_name = "Legit" if "fraud" in label.lower() else "Negative"

R = run_pipeline(X, y, test_pct / 100, int(seed), balanced, n_trees, max_depth, svm_c)
res = R["results"]

with st.sidebar:
    st.divider()
    best_key = res.sort_values("F1", ascending=False).iloc[0]["key"]
    keys = list(res["key"])
    sel_key = st.selectbox("Model used in steps 5 to 8", keys, index=keys.index(best_key),
                           help="Defaults to the model with the best F1 on the test set.")
model = R["fitted"][sel_key]
sel_model_name = sel_key.split(" | ")[0]
sel_exp = sel_key.split(" | ")[1]

pred_test = model.predict(R["Xte_s"])
sel_metrics = metric_row(R["yte"], pred_test)

tabs = st.tabs(["🏁 Summary", "1 · Data", "2 · K-Means", "3 · Split", "4 · Train",
                "5 · Evaluate", "6 · New records", "7 · Shifted data", "8 · Try it"])

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Project result at a glance")
    c = st.columns(4)
    c[0].metric("Rows / features", f"{len(X):,} / {X.shape[1]}")
    c[1].metric(f"{pos_name} share", f"{y.mean():.1%}")
    c[2].metric("K-Means vs truth (accuracy)", f"{R['km']['metrics']['Accuracy']:.1%}")
    c[3].metric("Selected model", sel_model_name)
    c = st.columns(4)
    c[0].metric("Test Precision", f"{sel_metrics['Precision']:.1%}")
    c[1].metric("Test Recall", f"{sel_metrics['Recall']:.1%}")
    c[2].metric("Test F1", f"{sel_metrics['F1']:.1%}")
    c[3].metric("Test Accuracy", f"{sel_metrics['Accuracy']:.1%}")
    st.markdown(f"""
<div class='card'>
<b>What this app does</b><br>
1. Loads the dataset and cleans it.<br>
2. Uses <b>K-Means (k=2)</b> to divide the data into two groups, then checks those groups against the true labels.<br>
3. Splits the data into <b>training</b> and <b>test</b> sets ({100 - test_pct}% / {test_pct}%).<br>
4. Trains <b>Random Forest, SVM and Logistic Regression</b>, once on the true labels and once on the K-Means labels.<br>
5. Evaluates with <b>precision, recall, F1</b> (hold-out and k-fold).<br>
6. Tests on <b>new records one by one</b>, then on shifted datasets.<br><br>
Currently selected: <b>{sel_key}</b>. Use the tabs above to follow each step.
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 1 Data
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Step 1 · Load and preprocess the data")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("**First rows of the dataset**")
        st.dataframe(df_raw.head(10))
        st.write("**Summary statistics**")
        st.dataframe(df_raw.describe().round(2))
    with c2:
        st.write("**Class balance**")
        counts = y.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(3.6, 3))
        ax.bar([neg_name, pos_name], [counts.get(0, 0), counts.get(1, 0)], color=["#6366f1", "#ef4444"])
        ax.set_ylabel("Rows")
        fig.tight_layout()
        show(fig)
        st.write("**Missing values**")
        st.write(f"Found {int(df_raw.isnull().sum().sum())} missing cells (rows with any are dropped).")
    st.markdown("**Preprocessing done:** rows with missing values removed → label converted to 0/1 → "
                "text columns one-hot encoded → numeric features scaled with `StandardScaler` "
                "(fitted on training data only, so nothing leaks from the test set).")

# ----------------------------------------------------------------------------
# 2 K-Means
# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Step 2 · Divide the data with K-Means (k = 2)")
    st.write("K-Means does **not** see the labels. It groups similar rows. "
             "Then we compare its two groups with the true labels (ground truth) to check whether it found the right split.")
    km = R["km"]
    c = st.columns(6)
    for i, k in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        c[i].metric(k, f"{km['metrics'][k]:.3f}")
    c[4].metric("ARI", f"{km['ari']:.3f}", help="Adjusted Rand Index: 1 = perfect match with true labels, 0 = random.")
    c[5].metric("Silhouette", f"{km['silhouette']:.3f}", help="How well separated the clusters are (-1 to 1).")
    c1, c2, c3 = st.columns(3)
    with c1:
        show(plot_cm(km["cm"], [neg_name, pos_name], "K-Means vs true labels (test)"))
    p = R["pca_xy"]
    with c2:
        fig, ax = plt.subplots(figsize=(3.8, 3.2))
        ax.scatter(p[:, 0], p[:, 1], c=R["tr_km"], s=8, cmap="coolwarm")
        ax.set_title("K-Means clusters (PCA view)", fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        fig.tight_layout(); show(fig)
    with c3:
        fig, ax = plt.subplots(figsize=(3.8, 3.2))
        ax.scatter(p[:, 0], p[:, 1], c=R["ytr"], s=8, cmap="coolwarm")
        ax.set_title("True labels (PCA view)", fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        fig.tight_layout(); show(fig)
    st.caption("If the two scatter plots look alike, K-Means found the same groups as the real labels. "
               "Cluster numbers 0/1 are arbitrary, so they were matched to the true labels automatically.")

# ----------------------------------------------------------------------------
# 3 Split
# ----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Step 3 · Training data vs testing data")
    n_tr, n_te = len(R["ytr"]), len(R["yte"])
    c = st.columns(3)
    c[0].metric("Training rows", f"{n_tr:,}", help="The model learns from these.")
    c[1].metric("Test rows", f"{n_te:,}", help="Hidden from the model during training.")
    c[2].metric("New records", "5 to 10", help="Created by you or an AI tool, never in the dataset.")
    fig, ax = plt.subplots(figsize=(7, 1.6))
    ax.barh([""], [n_tr], color="#6366f1", label="Training")
    ax.barh([""], [n_te], left=[n_tr], color="#f59e0b", label="Test")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.5), frameon=False)
    ax.set_xlabel("Rows")
    fig.tight_layout(); show(fig)
    st.write(f"Stratified split: {pos_name} share is {R['ytr'].mean():.1%} in training and {R['yte'].mean():.1%} in test.")
    st.table(pd.DataFrame({
        "Data": ["Training set", "Test set", "New records"],
        "Used for": ["Learning the pattern (fit)", "Honest performance check (hold-out)", "Final real-world style check"],
        "Model has seen it?": ["Yes", "No", "No"]}))

# ----------------------------------------------------------------------------
# 4 Train
# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Step 4 · Train the models")
    st.write("**Experiment A** trains on the true labels. **Experiment B** trains on the K-Means cluster labels. "
             "In both, the score is measured on the test set against the **true labels**.")
    show_df = res.drop(columns=["key"]).copy()
    for k in ["Accuracy", "Precision", "Recall", "F1"]:
        show_df[k] = show_df[k].round(3)
    st.dataframe(show_df)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    labels_ = [f"{r['Model']}\n{r['Trained on'][0]}" for _, r in res.iterrows()]
    xs = np.arange(len(res))
    w = 0.25
    ax.bar(xs - w, res["Precision"], w, label="Precision", color="#6366f1")
    ax.bar(xs, res["Recall"], w, label="Recall", color="#ef4444")
    ax.bar(xs + w, res["F1"], w, label="F1", color="#10b981")
    ax.set_xticks(xs); ax.set_xticklabels(labels_, fontsize=8)
    ax.set_ylim(0, 1.05); ax.legend(ncol=3, frameon=False)
    fig.tight_layout(); show(fig)
    st.caption("A = trained on true labels, B = trained on K-Means labels. "
               f"Best by F1: **{best_key}**.")

# ----------------------------------------------------------------------------
# 5 Evaluate
# ----------------------------------------------------------------------------
with tabs[5]:
    st.subheader(f"Step 5 · Evaluate: {sel_key}")
    c = st.columns(4)
    for i, k in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        c[i].metric(k, f"{sel_metrics[k]:.3f}")
    st.caption("Precision: when the model says positive, how often is it right? "
               "Recall: of all real positives, how many did it catch?")
    cm = confusion_matrix(R["yte"], pred_test, labels=[0, 1])
    c1, c2, c3 = st.columns(3)
    with c1:
        show(plot_cm(cm, [neg_name, pos_name], "Confusion matrix (test)"))
    with c2:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(R["Xte_s"])[:, 1]
            pr, rc, _ = precision_recall_curve(R["yte"], prob)
            fig, ax = plt.subplots(figsize=(3.8, 3.2))
            ax.plot(rc, pr, color="#6366f1")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            try:
                ax.set_title(f"Precision-Recall (AUC-ROC {roc_auc_score(R['yte'], prob):.3f})", fontsize=10)
            except Exception:
                ax.set_title("Precision-Recall", fontsize=10)
            ax.set_ylim(0, 1.05)
            fig.tight_layout(); show(fig)
    with c3:
        imp = None
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=R["features"])
        elif hasattr(model, "coef_"):
            imp = pd.Series(np.abs(model.coef_[0]), index=R["features"])
        if imp is not None:
            imp = imp.sort_values().tail(10)
            fig, ax = plt.subplots(figsize=(3.8, 3.2))
            ax.barh(imp.index, imp.values, color="#10b981")
            ax.set_title("Feature importance", fontsize=10)
            fig.tight_layout(); show(fig)
        else:
            st.caption("Feature importance is not available for an RBF SVM.")

    st.markdown("#### k-fold cross-validation (true labels)")
    kf = run_kfold(X, y, sel_model_name, n_trees, max_depth, svm_c, balanced, n_folds, int(seed))
    st.dataframe(kf.round(3))
    mean_row = kf[["Accuracy", "Precision", "Recall", "F1"]].agg(["mean", "std"]).round(3)
    st.write("**Mean and standard deviation across folds**")
    st.dataframe(mean_row)
    st.caption("k-fold rotates which part of the data is the test set. "
               "A small standard deviation means the result does not depend on one lucky split.")

# ----------------------------------------------------------------------------
# 6 New records
# ----------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Step 6 · Test on new records, one by one")
    st.write("These records are **not in the dataset**. Edit them, add rows, or paste records from an AI tool. "
             "Set `expected` to 1 for positive and 0 for negative.")
    mode = st.radio("Source of new records", ["Auto-generated (editable)", "Paste CSV from an AI tool"],
                    horizontal=True)
    feat = R["features"]
    is_default = feat == [c for c in SAMPLE_RECORDS.columns if c != "expected"]
    if mode.startswith("Auto"):
        if "regen" not in st.session_state:
            st.session_state["regen"] = 0
        if st.button("🔄 Generate a different set"):
            st.session_state["regen"] += 1
        if is_default and st.session_state["regen"] == 0:
            base = SAMPLE_RECORDS.copy()
        else:
            base = gen_records(X, y, 8, int(seed) + st.session_state["regen"])
        rec = st.data_editor(base, num_rows="dynamic", key=f"rec_editor_{st.session_state['regen']}")
    else:
        desc = X.describe().loc[["min", "max"]].round(2)
        ranges = "; ".join(f"{c}: {desc.loc['min', c]} to {desc.loc['max', c]}" for c in feat)
        prompt = (f"Generate 8 realistic NEW records as CSV with this header: {','.join(feat)},expected. "
                  f"Use values inside these ranges: {ranges}. "
                  f"expected = 1 for {pos_name.lower()} and 0 for {neg_name.lower()}. "
                  f"Make about half of each class. Output only the CSV.")
        with st.expander("📋 Prompt to give an AI tool"):
            st.code(prompt, language="text")
        txt = st.text_area("Paste the CSV here", value=",".join(feat + ["expected"]) + "\n", height=200)
        try:
            rec = pd.read_csv(io.StringIO(txt))
        except Exception:
            rec = pd.DataFrame()

    missing = [c for c in feat + ["expected"] if c not in rec.columns]
    if rec.empty or missing:
        st.info("Waiting for valid records." + (f" Missing columns: {', '.join(missing)}" if missing and not rec.empty else ""))
    else:
        rec = rec.dropna().reset_index(drop=True)
        Xn = rec[feat].astype(float)
        yn = rec["expected"].astype(int).values
        Xn_s = R["scaler"].transform(Xn)
        out = []
        for i in range(len(Xn)):                       # one by one
            p1 = model.predict_proba(Xn_s[[i]])[0, 1] if hasattr(model, "predict_proba") else float(model.predict(Xn_s[[i]])[0])
            pd_ = int(model.predict(Xn_s[[i]])[0])
            out.append({"Record": i + 1, **Xn.iloc[i].to_dict(),
                        f"{pos_name} probability": round(p1, 3),
                        "Predicted": pd_, "Expected": int(yn[i]),
                        "Result": "✅ correct" if pd_ == yn[i] else "❌ wrong"})
        out = pd.DataFrame(out)
        st.dataframe(out)
        preds = out["Predicted"].values
        c = st.columns(4)
        c[0].metric("Precision (new)", f"{precision_score(yn, preds, zero_division=0):.3f}",
                    delta=f"{precision_score(yn, preds, zero_division=0) - sel_metrics['Precision']:+.3f} vs test")
        c[1].metric("Recall (new)", f"{recall_score(yn, preds, zero_division=0):.3f}",
                    delta=f"{recall_score(yn, preds, zero_division=0) - sel_metrics['Recall']:+.3f} vs test")
        c[2].metric("F1 (new)", f"{f1_score(yn, preds, zero_division=0):.3f}")
        c[3].metric("Records correct", f"{int((preds == yn).sum())} / {len(yn)}")
        st.caption("With only 5 to 10 records, one mistake changes precision and recall a lot, "
                   "so these numbers are less reliable than the test-set numbers.")
        pick = st.selectbox("Inspect a single record", list(out["Record"]))
        row = out[out["Record"] == pick].iloc[0]
        st.write(f"**Record {pick}:** predicted **{pos_name if row['Predicted'] == 1 else neg_name}**, "
                 f"expected **{pos_name if row['Expected'] == 1 else neg_name}** → {row['Result']}")
        st.progress(float(row[f"{pos_name} probability"]), text=f"{pos_name} probability")

# ----------------------------------------------------------------------------
# 7 Shifted data
# ----------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Step 7 · Why precision and recall change on different data")
    st.write("The same trained model is tested on re-sampled versions of the test set.")
    sh = shifted_table(model, R["Xte_s"], R["yte"], int(seed))
    sh_show = sh.copy()
    for k in ["Precision", "Recall", "F1"]:
        sh_show[k] = sh_show[k].round(3)
    st.dataframe(sh_show)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    xs = np.arange(len(sh))
    ax.bar(xs - 0.2, sh["Precision"], 0.4, label="Precision", color="#6366f1")
    ax.bar(xs + 0.2, sh["Recall"], 0.4, label="Recall", color="#ef4444")
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace(" (", "\n(") for s in sh["Scenario"]], fontsize=8)
    ax.set_ylim(0, 1.05); ax.legend(ncol=2, frameon=False)
    fig.tight_layout(); show(fig)

# ----------------------------------------------------------------------------
# 8 Try it
# ----------------------------------------------------------------------------
with tabs[8]:
    st.subheader("Step 8 · Try one transaction yourself")
    st.write(f"Enter values and the selected model (**{sel_key}**) predicts the class.")
    vals = {}
    grid = st.columns(3)
    for i, f in enumerate(R["features"]):
        lo, hi, med = float(X[f].min()), float(X[f].max()), float(X[f].median())
        with grid[i % 3]:
            vals[f] = st.number_input(f, value=med, help=f"Typical range in the data: {lo:.2f} to {hi:.2f}")
    if st.button("Predict", type="primary"):
        row = pd.DataFrame([vals])[R["features"]]
        s = R["scaler"].transform(row)
        p1 = model.predict_proba(s)[0, 1] if hasattr(model, "predict_proba") else float(model.predict(s)[0])
        st.metric(f"{pos_name} probability", f"{p1:.1%}")
        if p1 >= 0.5:
            st.error(f"Likely {pos_name.upper()}")
        else:
            st.success(f"Looks {neg_name.lower()}")
        for f in R["features"]:
            lo, hi = float(X[f].min()), float(X[f].max())
            if vals[f] < lo or vals[f] > hi:
                st.warning(f"'{f}' is outside the range the model was trained on ({lo:.2f} to {hi:.2f}). "
                           "The prediction may be unreliable.")
