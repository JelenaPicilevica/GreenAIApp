# reuse/train.py

import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score
)

from src.tasks.reuse.dataset import ReuseDatasetLoader
from src.tasks.reuse.pipeline import ReusePipeline


# =====================
# VISUALIZATION
# =====================
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
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
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.title("REUSE NN ROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()

    return auc


# =====================
# EVALUATION
# =====================
def evaluate(y_true, preds, probs):

    print("\n===== REUSE NN =====")

    f1 = f1_score(y_true, preds)
    acc = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)

    print(f"F1:        {f1:.4f}")
    print(f"ACC:       {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"ROC AUC:   {auc:.4f}")

    cm = confusion_matrix(y_true, preds)
    plot_confusion_matrix(cm, "REUSE NN")

    plot_roc(y_true, probs)


# =====================
# MAIN
# =====================
def main():

    print("\n🚀 START REUSE PIPELINE\n")

    # =====================
    # DATA
    # =====================
    loader = ReuseDatasetLoader()
    train_df, val_df, test_df = loader.load()

    # =====================
    # PIPELINE
    # =====================
    pipeline = ReusePipeline()
    pipeline.fit(train_df, val_df)

    # =====================
    # PREDICT
    # =====================
    preds, probs = pipeline.predict(test_df)

    y_true = test_df["label"].values

    # =====================
    # EVALUATE
    # =====================
    evaluate(y_true, preds, probs)


if __name__ == "__main__":
    main()