# reuse/neuralNetwork/tuner.py

import optuna
import torch
from sklearn.metrics import f1_score

from src.core.model import ModelBuilder


class OptunaTuner:

    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.best_f1 = 0.0

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
            logits = model(self.X_val)
            preds = torch.argmax(logits, dim=1).numpy()

        f1 = f1_score(self.y_val.numpy(), preds)

        is_best = f1 > self.best_f1
        if is_best:
            self.best_f1 = f1

        print(f"[T{trial.number}] F1={f1:.4f}" + (" ↑ BEST" if is_best else ""))

        return f1

    def tune(self, n_trials=30):

        print("\n========== REUSE TUNING ==========\n")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)

        print("\nBest params:", study.best_params)

        return study.best_params