"""
Baseline: TF-IDF + Sentyment + Logistic Regression + XGBoost
Tracking przez MLflow.

     python src/models/baseline.py

Aby uruchomic  MLflow UI:  mlflow ui
"""

import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED  = Path("data/processed")
OUTPUT_DIR = Path("wyniki_baseline")
OUTPUT_DIR.mkdir(exist_ok=True)


def wczytaj_dane():
    train = pd.read_csv(PROCESSED / "train_clean.csv", encoding="utf-8-sig")
    val   = pd.read_csv(PROCESSED / "val_clean.csv",   encoding="utf-8-sig")
    test  = pd.read_csv(PROCESSED / "test_clean.csv",  encoding="utf-8-sig")
    return train, val, test


def wektoryzuj(train, val, test):
    tfidf = TfidfVectorizer(
        max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2,
    )
    X_train_tfidf = tfidf.fit_transform(train["Tresc"].fillna(""))
    X_val_tfidf   = tfidf.transform(val["Tresc"].fillna(""))
    X_test_tfidf  = tfidf.transform(test["Tresc"].fillna(""))

    if "Sentyment" in train.columns:
        enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
        X_train_sent = enc.fit_transform(train[["Sentyment"]])
        X_val_sent   = enc.transform(val[["Sentyment"]])
        X_test_sent  = enc.transform(test[["Sentyment"]])
        X_train = hstack([X_train_tfidf, X_train_sent])
        X_val   = hstack([X_val_tfidf,   X_val_sent])
        X_test  = hstack([X_test_tfidf,  X_test_sent])
        print("  Cechy: TF-IDF i Sentyment")
    else:
        X_train, X_val, X_test = X_train_tfidf, X_val_tfidf, X_test_tfidf
        print("  Cechy: TF-IDF, bez Sentymentu")

    return X_train, X_val, X_test, tfidf


def metryki(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


def zapisz_confusion_matrix(y_true, y_pred, nazwa: str) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Prawda", "Fake"], yticklabels=["Prawda", "Fake"])
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Prawdziwe")
    ax.set_title(f"Confusion Matrix — {nazwa}")
    path = str(OUTPUT_DIR / f"cm_{nazwa}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def zapisz_roc(y_true, y_prob, nazwa: str, auc: float) -> str:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC — {nazwa}")
    ax.legend()
    path = str(OUTPUT_DIR / f"roc_{nazwa}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def main():
    train, val, test = wczytaj_dane()
    print(f"  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    X_train, X_val, X_test, tfidf = wektoryzuj(train, val, test)
    y_train, y_val, y_test = train["Etykieta"], val["Etykieta"], test["Etykieta"]

    MODELE = {
        "LogisticRegression_C1": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=42
        ),
        "LogisticRegression_C0.1": LogisticRegression(
            C=0.1, max_iter=1000, solver="lbfgs", random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        ),
    }

    mlflow.set_experiment("fake_news_baseline")
    wyniki = []

    for nazwa, model in MODELE.items():
        print(f"Model: {nazwa}")
        with mlflow.start_run(run_name=nazwa):
            mlflow.log_params(model.get_params())
            mlflow.log_params({
                "tfidf_max_features": tfidf.max_features,
                "tfidf_ngram_range": str(tfidf.ngram_range),
                "sentyment_feature": "Sentyment" in train.columns,
                "train_size": len(train),
            })

            model.fit(X_train, y_train)

            for split_name, X_vec, y_true in [
                ("val",  X_val,  y_val),
                ("test", X_test, y_test),
            ]:
                y_pred = model.predict(X_vec)
                y_prob = model.predict_proba(X_vec)[:, 1]
                m = metryki(y_true, y_pred, y_prob)

                mlflow.log_metrics({f"{split_name}_{k}": v for k, v in m.items()})

                print(f"\n[{split_name.upper()}]")
                for k, v in m.items():
                    print(f"  {k:12s}: {v:.4f}")
                print(classification_report(
                    y_true, y_pred, target_names=["Prawda", "Fake news"]
                ))

                cm_path  = zapisz_confusion_matrix(y_true, y_pred, f"{nazwa}_{split_name}")
                roc_path = zapisz_roc(y_true, y_prob, f"{nazwa}_{split_name}", m["roc_auc"])
                mlflow.log_artifact(cm_path)
                mlflow.log_artifact(roc_path)

                if split_name == "test":
                    wyniki.append({"model": nazwa, **m})

            mlflow.sklearn.log_model(model, artifact_path="model")

    df_wyniki = pd.DataFrame(wyniki).set_index("model").round(4)
    df_wyniki = df_wyniki.sort_values("f1", ascending=False)
    print("Porownanie modeli na test set:")
    print(df_wyniki.to_string())
    df_wyniki.to_csv(OUTPUT_DIR / "porownanie_modeli.csv")

    # Zapis najlepszego modelu (LR C1) do uzycia w app.py
    # Zapis modelu dla app.py — zawsze tylko TF-IDF, bez sentymentu
    tfidf_only = TfidfVectorizer(
        max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2,
    )
    X_train_only = tfidf_only.fit_transform(train["Tresc"].fillna(""))
    lr_only = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    lr_only.fit(X_train_only, y_train)
    with open(OUTPUT_DIR / "lr_model.pkl", "wb") as f:
        pickle.dump((tfidf_only, lr_only), f)
    print("Model zapisany -> wyniki_baseline/lr_model.pkl")

    print("\nMLflow UI: mlflow ui")


if __name__ == "__main__":
    main()