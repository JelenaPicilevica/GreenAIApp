import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score
)


class Evaluator:
    """
    Evaluates model predictions using classification metrics
    and visualization (confusion matrix and ROC curve).
    """

    @staticmethod
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

    @staticmethod
    def plot_roc(y_true, probs, title):
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(title)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.show()

        return auc

    @staticmethod
    def evaluate(name, y_true, probs):
        print(f"\n===== {name} =====")

        preds = (probs > 0.5).astype(int)

        f1 = f1_score(y_true, preds)
        acc = accuracy_score(y_true, preds)
        auc = roc_auc_score(y_true, probs)

        print("F1:", f1)
        print("ACC:", acc)
        print("ROC AUC:", auc)

        Evaluator.plot_confusion_matrix(confusion_matrix(y_true, preds), name)
        Evaluator.plot_roc(y_true, probs, f"{name} ROC")

        return f1, acc, auc

    @staticmethod
    def debug_hybrid(y_true, nn_probs, hybrid_probs, low=0.3, high=0.7):
        """
        Analyzes how hybrid model improves over NN:
        - FP / FN reduction
        - routing efficiency
        """

        print("\n===== DEBUG =====")

        nn_preds = (nn_probs > 0.5).astype(int)
        hybrid_preds = (hybrid_probs > 0.5).astype(int)

        nn_fp = np.sum((y_true == 0) & (nn_preds == 1))
        hybrid_fp = np.sum((y_true == 0) & (hybrid_preds == 1))

        nn_fn = np.sum((y_true == 1) & (nn_preds == 0))
        hybrid_fn = np.sum((y_true == 1) & (hybrid_preds == 0))

        print(f"\nFP: {nn_fp} → {hybrid_fp} (fixed {nn_fp - hybrid_fp})")
        print(f"FN: {nn_fn} → {hybrid_fn} (fixed {nn_fn - hybrid_fn})")

        mid_zone = (nn_probs > low) & (nn_probs < high)

        total_routed = np.sum(mid_zone)
        same = np.sum((nn_preds == hybrid_preds) & mid_zone)

        print("\nEfficiency:")
        print(f"Useful: {total_routed - same}")
        print(f"Waste: {same}")

        if total_routed > 0:
            print(f"Efficiency: {(total_routed - same) / total_routed:.2f}")
        else:
            print("Efficiency: N/A (no routed samples)")