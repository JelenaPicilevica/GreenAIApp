import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler

from drift.dataset import DatasetLoader
from drift.features import FeatureBuilder
from drift.model import ModelBuilder
from drift.trainer import Trainer
from drift.tuner import OptunaTuner


def plot_confusion_matrix(cm):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.show()


def plot_roc(y_true, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("ROC Curve")
    plt.show()

    print("ROC AUC:", auc)


def main():

    print("\n🚀 START PIPELINE\n")

    # =====================
    # DATA
    # =====================
    loader = DatasetLoader()
    train_df, val_df, test_df = loader.load()

    # =====================
    # FEATURES
    # =====================
    fb = FeatureBuilder()
    fb.fit(train_df)

    X_train, y_train = fb.build(train_df)
    X_val, y_val = fb.build(val_df)
    X_test, y_test = fb.build(test_df)

    # =====================
    # SCALE
    # =====================
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_val = torch.tensor(X_val.toarray(), dtype=torch.float32)
    X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)

    y_train = torch.tensor(y_train)
    y_val = torch.tensor(y_val)
    y_test = torch.tensor(y_test)

    # =====================
    # TUNING
    # =====================
    tuner = OptunaTuner(X_train, y_train, X_val, y_val)
    params = tuner.tune()

    # =====================
    # MODEL
    # =====================
    model = ModelBuilder.build(
        X_train.shape[1],
        params["n_layers"],
        params["hidden"],
        params["dropout"]
    )

    # =====================
    # TRAIN
    # =====================
    trainer = Trainer(model, params)
    model = trainer.train(X_train, y_train, X_val, y_val)

    print("\nEvaluating...")

    # =====================
    # CALIBRATION
    # =====================
    with torch.no_grad():
        val_probs = torch.softmax(model(X_val), dim=1)[:, 1].numpy()

    calibrator = LogisticRegression()
    calibrator.fit(val_probs.reshape(-1, 1), y_val.numpy())

    # =====================
    # TEST
    # =====================
    with torch.no_grad():
        raw_probs = torch.softmax(model(X_test), dim=1)[:, 1].numpy()

    probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

    # =====================
    # BEST THRESHOLD (PR curve)
    # =====================
    precision, recall, thresholds = precision_recall_curve(y_test.numpy(), probs)

    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-6)

    # ограничение threshold (ключевая штука)
    valid = thresholds > 0.2
    f1_scores = f1_scores * valid

    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_t = thresholds[best_idx]

    preds = (probs > best_t).astype(int)

    # =====================
    # CONFIDENCE ANALYSIS
    # =====================
    confidence = np.abs(probs - 0.5)

    LOW = 0.3
    HIGH = 0.7

    low_zone = probs < LOW
    mid_zone = (probs >= LOW) & (probs <= HIGH)
    high_zone = probs > HIGH

    print("\n===== CONFIDENCE ANALYSIS =====")
    print("Total:", len(probs))

    print("\nCounts:")
    print("LOW (уверенно нет):", low_zone.sum())
    print("MID (сомнение → DistilBERT):", mid_zone.sum())
    print("HIGH (уверенно drift):", high_zone.sum())

    print("\nPercent:")
    print("LOW:", low_zone.mean())
    print("MID:", mid_zone.mean())
    print("HIGH:", high_zone.mean())

    # =====================
    # QUALITY BY ZONES
    # =====================
    def eval_zone(name, mask):
        if mask.sum() == 0:
            print(name, "empty")
            return

        acc = accuracy_score(y_test.numpy()[mask], preds[mask])
        print(f"{name}: samples={mask.sum()}, acc={acc:.4f}")

    print("\n===== ZONE QUALITY =====")
    eval_zone("LOW", low_zone)
    eval_zone("MID", mid_zone)
    eval_zone("HIGH", high_zone)

    # =====================
    # METRICS
    # =====================
    print("\nFINAL METRICS")
    print("F1:", best_f1)
    print("ACC:", accuracy_score(y_test.numpy(), preds))
    print("Best threshold:", best_t)

    # =====================
    # CONFUSION MATRIX
    # =====================
    cm = confusion_matrix(y_test.numpy(), preds)
    print("\nConfusion Matrix:\n", cm)
    plot_confusion_matrix(cm)

    # =====================
    # ROC
    # =====================
    plot_roc(y_test.numpy(), probs)

    # =====================
    # SAVE EVERYTHING
    # =====================
    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), "models/drift_model.pt")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(calibrator, "models/calibrator.pkl")
    joblib.dump(fb, "models/feature_builder.pkl")
    joblib.dump(best_t, "models/threshold.pkl")

    print("\nAll artifacts saved to models/")


if __name__ == "__main__":
    main()