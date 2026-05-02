"""
Fine-tuning HerBERT-a na  klasyfikacji fake news.
Model: allegro/herbert-base-cased

    python src/models/herbert_finetune.py
"""

import json
import pandas as pd
import numpy as np
import torch
import mlflow
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


MODEL_NAME  = "allegro/herbert-base-cased"
PROCESSED   = Path("data/processed")
OUTPUT_DIR  = Path("wyniki_herbert")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_LEN    = 256
BATCH_SIZE = 16
EPOCHS     = 3
LR         = 2e-5
WARMUP     = 0.1


class FakeNewsDataset(Dataset):
    def __init__(self, teksty, etykiety, tokenizer):
        self.encodings = tokenizer(
            teksty,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(etykiety, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings.get(
                "token_type_ids",
                torch.zeros_like(self.encodings["input_ids"])
            )[idx],
            "labels": self.labels[idx],
        }



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall":    recall_score(labels, preds),
        "roc_auc":   roc_auc_score(labels, probs),
    }


def zapisz_confusion_matrix(y_true, y_pred, nazwa: str) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax,
                xticklabels=["Prawda", "Fake"], yticklabels=["Prawda", "Fake"])
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Prawdziwe")
    ax.set_title(f"Confusion Matrix — {nazwa}")
    path = str(OUTPUT_DIR / f"cm_{nazwa}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def zapisz_roc(y_true, y_prob, nazwa: str, auc: float) -> str:
    from sklearn.metrics import roc_curve
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = pd.read_csv(PROCESSED / "train_clean.csv", encoding="utf-8-sig")
    val   = pd.read_csv(PROCESSED / "val_clean.csv",   encoding="utf-8-sig")
    test  = pd.read_csv(PROCESSED / "test_clean.csv",  encoding="utf-8-sig")
    print(f"  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    print(f"\nLadowanie modelu: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    train_ds = FakeNewsDataset(
        train["Tresc"].fillna("").tolist(),
        train["Etykieta"].tolist(), tokenizer,
    )
    val_ds = FakeNewsDataset(
        val["Tresc"].fillna("").tolist(),
        val["Etykieta"].tolist(), tokenizer,
    )
    test_ds = FakeNewsDataset(
        test["Tresc"].fillna("").tolist(),
        test["Etykieta"].tolist(), tokenizer,
    )

    total_steps  = (len(train_ds) // BATCH_SIZE) * EPOCHS
    warmup_steps = int(total_steps * WARMUP)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        fp16=(device == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    mlflow.set_experiment("fake_news_herbert")
    with mlflow.start_run(run_name="herbert-base-cased"):
        mlflow.log_params({
            "model":      MODEL_NAME,
            "max_len":    MAX_LEN,
            "batch_size": BATCH_SIZE,
            "epochs":     EPOCHS,
            "lr":         LR,
            "warmup":     WARMUP,
            "device":     device,
        })

        trainer.train()

# Ewaluacja na validation set 
        print("\nEwaluacja val set...")
        val_out   = trainer.predict(val_ds)
        val_preds = np.argmax(val_out.predictions, axis=1)
        val_probs = torch.softmax(torch.tensor(val_out.predictions), dim=1).numpy()[:, 1]
        val_true  = val["Etykieta"].values

        val_m = {f"val_{k}": v for k, v in {
            "accuracy":  accuracy_score(val_true, val_preds),
            "f1":        f1_score(val_true, val_preds),
            "precision": precision_score(val_true, val_preds),
            "recall":    recall_score(val_true, val_preds),
            "roc_auc":   roc_auc_score(val_true, val_probs),
        }.items()}
        mlflow.log_metrics(val_m)

# Ewaluacja na test set 
        print("Ewaluacja test set...")
        test_out   = trainer.predict(test_ds)
        test_preds = np.argmax(test_out.predictions, axis=1)
        test_probs = torch.softmax(torch.tensor(test_out.predictions), dim=1).numpy()[:, 1]
        test_true  = test["Etykieta"].values

        test_m = {
            "accuracy":  accuracy_score(test_true, test_preds),
            "f1":        f1_score(test_true, test_preds),
            "precision": precision_score(test_true, test_preds),
            "recall":    recall_score(test_true, test_preds),
            "roc_auc":   roc_auc_score(test_true, test_probs),
        }
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})

        
        print("\n VAL SET ")
        for k, v in val_m.items():
            print(f"  {k:20s}: {v:.4f}")

        print("\n TEST SET ")
        for k, v in test_m.items():
            print(f"  {'test_'+k:20s}: {v:.4f}")

        print("\nClassification report (test):")
        print(classification_report(
            test_true, test_preds, target_names=["Prawda", "Fake news"]
        ))



        cm_path  = zapisz_confusion_matrix(test_true, test_preds, "herbert_test")
        roc_path = zapisz_roc(test_true, test_probs, "herbert_test", test_m["roc_auc"])
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)

        # Zapis wynikow do json pliku
        wyniki = {
            "model": MODEL_NAME,
            "config": {
                "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
                "epochs": EPOCHS, "lr": LR,
            },
            "val":  {k.replace("val_", ""): round(v, 4) for k, v in val_m.items()},
            "test": {k: round(v, 4) for k, v in test_m.items()},
        }
        wyniki_path = OUTPUT_DIR / "wyniki_herbert.json"
        with open(wyniki_path, "w", encoding="utf-8") as f:
            json.dump(wyniki, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(str(wyniki_path))
        print(f"\nWyniki zapisane:{wyniki_path}")

        # Zapisanie modelu
        model_path = str(OUTPUT_DIR / "model_final")
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        mlflow.log_artifact(model_path)
        print(f"Model zapisany:  {model_path}")


if __name__ == "__main__":
    main()