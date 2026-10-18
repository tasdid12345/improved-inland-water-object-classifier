"""Baseline MLP vs class-balanced, sigmoid-calibrated LightGBM on UCI Image Segmentation."""
import json
import socket
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

SEED = 42
SEEDS = [42, 43, 44]  # model-initialisation seeds (the split is fixed at SEED)
SOURCE_URL = "https://archive.ics.uci.edu/dataset/50/image+segmentation"
socket.setdefaulttimeout(60)  # 60 s timeout on download sockets (fetch_ucirepo has no timeout arg)


def load_data():
    """Fetch UCI dataset 50 with retry + exponential backoff on transient failures."""
    for attempt in range(5):
        try:
            return fetch_ucirepo(id=50)
        except Exception as e:  # network / HTTP 429 / 5xx
            wait = 2 ** attempt
            print(f"Download failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    sys.exit("ERROR: could not download the dataset after 5 attempts")


# Both models use the same scaler + estimator template; only the estimator differs.
def make_base(seed, hidden_layer_sizes=(64,), alpha=1e-4):
    """Baseline: standard MLP on standardized features."""
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, max_iter=500, random_state=seed))


def make_lgbm(seed, num_leaves=31, learning_rate=0.05):
    """LightGBM with balanced class weights (estimator is wrapped in calibration later)."""
    return make_pipeline(StandardScaler(), LGBMClassifier(
        class_weight="balanced", n_estimators=300, num_leaves=num_leaves,
        learning_rate=learning_rate, random_state=seed, verbose=-1))


def evaluate(model, X_tr, y_tr, X_te, y_te, labels):
    """Fit on train, score on test: macro-F1 (main), accuracy and log-loss (secondary)."""
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)
    pred = proba.argmax(axis=1)
    return {"f1": f1_score(y_te, pred, average="macro"),
            "acc": accuracy_score(y_te, pred),
            "logloss": log_loss(y_te, proba, labels=labels)}


def main():
    t0 = time.time()
    ds = load_data()
    X = ds.data.features.select_dtypes(include="number").to_numpy(dtype=float)
    y = LabelEncoder().fit_transform(ds.data.targets.iloc[:, 0].astype(str))
    labels = np.unique(y)
    n_rows = len(y)
    print(f"Loaded {n_rows} rows, {X.shape[1]} features, {len(labels)} classes")
    if n_rows < 100 or len(labels) < 2:
        sys.exit("ERROR: dataset unusable")

    # Single stratified 80/20 split with fixed seed
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Split: {len(y_tr)} train / {len(y_te)} test")

    # Hyper-parameter selection: CV on training data only, same splitter and 4 candidates each
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    base_grid = {"mlpclassifier__hidden_layer_sizes": [(64,), (128,)],
                 "mlpclassifier__alpha": [1e-4, 1e-2]}
    lgbm_grid = {"lgbmclassifier__num_leaves": [15, 31],
                 "lgbmclassifier__learning_rate": [0.05, 0.1]}
    gs_base = GridSearchCV(make_base(SEED), base_grid, cv=cv, scoring="f1_macro").fit(X_tr, y_tr)
    gs_imp = GridSearchCV(make_lgbm(SEED), lgbm_grid, cv=cv, scoring="f1_macro").fit(X_tr, y_tr)
    base_params = {k.split("__")[1]: v for k, v in gs_base.best_params_.items()}
    imp_params = {k.split("__")[1]: v for k, v in gs_imp.best_params_.items()}
    print(f"Baseline CV best: {base_params} (CV macro-F1 {gs_base.best_score_:.4f})")
    print(f"Improved CV best: {imp_params} (CV macro-F1 {gs_imp.best_score_:.4f})")

    # Final fits on the full training set, evaluated once on the test set for each seed
    res = {"baseline": [], "improved": []}
    for seed in SEEDS:
        print(f"Seed {seed}: fitting baseline MLP and improved calibrated LightGBM")
        res["baseline"].append(evaluate(make_base(seed, **base_params), X_tr, y_tr, X_te, y_te, labels))
        improved = CalibratedClassifierCV(make_lgbm(seed, **imp_params), method="sigmoid", cv=3)
        res["improved"].append(evaluate(improved, X_tr, y_tr, X_te, y_te, labels))

    def agg(name, key):
        vals = np.array([r[key] for r in res[name]])
        return float(vals.mean()), float(vals.std())

    b_f1, b_f1_sd = agg("baseline", "f1")
    i_f1, i_f1_sd = agg("improved", "f1")
    b_acc, _ = agg("baseline", "acc")
    i_acc, _ = agg("improved", "acc")
    b_ll, _ = agg("baseline", "logloss")
    i_ll, _ = agg("improved", "logloss")
    print(f"Test macro-F1: baseline {b_f1:.4f} vs improved {i_f1:.4f}")

    # Optional chart
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["MLP baseline", "Calibrated LightGBM"], [b_f1, i_f1],
           yerr=[b_f1_sd, i_f1_sd], capsize=6, color=["gray", "tab:blue"])
    ax.set_ylabel("Test macro-F1 (mean ± sd over 3 seeds)")
    ax.set_title("UCI Image Segmentation")
    fig.tight_layout()
    fig.savefig("figures/macro_f1.png", dpi=120)
    plt.close(fig)

    results = {
        "question": "Can a class-balanced, calibrated LightGBM model beat a standard MLP neural-network "
                    "baseline at classifying outdoor scene regions on the public UCI Image Segmentation "
                    "dataset, as a stand-in for detecting objects on inland waters?",
        "method": "Stratified 80/20 split (seed 42); baseline MLP vs class-balanced LightGBM with "
                  "sigmoid calibration; small CV grids on the training set; test set used once per seed.",
        "dataset": {
            "name": "UCI Image Segmentation",
            "source_url": SOURCE_URL,
            "n_rows": int(n_rows),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 (test)",
        "higher_is_better": True,
        "baseline": "StandardScaler + scikit-learn MLPClassifier (hidden layer and alpha chosen by 5-fold CV on train)",
        "improved": "StandardScaler + LightGBM with class_weight='balanced' (num_leaves and learning_rate chosen "
                    "by 5-fold CV on train), probabilities calibrated with sigmoid (Platt) calibration (3-fold)",
        "metrics": {
            "baseline_score": b_f1,
            "improved_score": i_f1,
            "improvement": i_f1 - b_f1,
            "baseline_score_std": b_f1_sd,
            "improved_score_std": i_f1_sd,
            "baseline_accuracy": b_acc,
            "improved_accuracy": i_acc,
            "baseline_logloss": b_ll,
            "improved_logloss": i_ll,
            "n_seeds": len(SEEDS),
            "runtime_seconds": time.time() - t0,
        },
        "notes": "The paper's own deep network is not described, so the MLP is a stand-in baseline. "
                 "The dataset is a stand-in for inland-water object detection, not the paper's data. "
                 "The split is fixed; seeds 42-44 vary only model initialisation. The test set has "
                 "only about 460 rows, so differences may be within noise. The classes are nearly "
                 "balanced, so class weighting has limited effect. Hyper-parameter grids are deliberately tiny.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json")


if __name__ == "__main__":
    main()
