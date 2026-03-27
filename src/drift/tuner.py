import numpy as np
import optuna
import torch
from sklearn.metrics import precision_recall_curve

from model import ModelBuilder

"""
Hyperparameter tuning with:
- compact logging
- F1 + Accuracy
- Top-5 summary
"""


class OptunaTuner:

    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.best_f1 = 0.0
        self.trials_history = []

    def objective(self, trial):

        n_layers = trial.suggest_int("n_layers", 2, 4)
        hidden = trial.suggest_int("hidden", 128, 512)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd = trial.suggest_float("wd", 1e-6, 1e-2, log=True)
        batch = trial.suggest_categorical("batch", [128, 256, 512])

        model = ModelBuilder.build(
            self.X_train.shape[1],
            n_layers,
            hidden,
            dropout
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(6):
            idx = torch.randperm(len(self.X_train))

            for i in range(0, len(self.X_train), batch):
                j = idx[i:i + batch]

                xb = self.X_train[j]
                yb = self.y_train[j]

                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            probs = torch.softmax(model(self.X_val), dim=1)[:, 1].numpy()

        precision, recall, thresholds = precision_recall_curve(self.y_val.numpy(), probs)

        f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-6)

        valid = thresholds > 0.2
        f1_scores = f1_scores * valid

        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]

        preds = (probs > thresholds[best_idx]).astype(int)
        best_acc = (preds == self.y_val.numpy()).mean()

        self.trials_history.append({
            "trial": trial.number,
            "f1": best_f1,
            "acc": best_acc,
            "params": trial.params
        })

        is_best = best_f1 > self.best_f1
        if is_best:
            self.best_f1 = best_f1

        arrow = " ↑ BEST" if is_best else ""

        print(
            f"[T{trial.number}] "
            f"F1={best_f1:.4f} | ACC={best_acc:.4f} | "
            f"drop={dropout:.2f}; lr={lr:.4g}; wd={wd:.4g}; batch={batch}"
            f"{arrow}"
        )

        return best_f1

    def tune(self, n_trials=50):

        print("\n========== OPTUNA TUNING ==========\n")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)

        print("\n========== BEST RESULT ==========")
        print(
            f"Best F1: {study.best_value:.4f}\n"
            f"Best params: {study.best_params}"
        )

        return study.best_params