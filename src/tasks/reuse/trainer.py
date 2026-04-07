# reuse/neuralNetwork/trainer.py

import numpy as np
import torch
import os

from sklearn.metrics import f1_score, precision_score


class Trainer:

    def __init__(self, model, params):
        self.model = model
        self.params = params

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["wd"]
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=20
        )

        # 🔥 сильный штраф за FP
        self.loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 4.0])  # увеличили штраф для FP
        )

        self.best_threshold = 0.6

    # 🔥 подбор threshold с контролем precision
    def _find_best_threshold(self, probs, y_true):

        best_t = 0.6
        best_f1 = 0

        for t in np.linspace(0.4, 0.95, 60):

            preds = (probs > t).astype(int)

            precision = precision_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds)

            # 🔥 мягкий приоритет precision
            score = f1 + 0.6 * precision

            if score > best_f1:
                best_f1 = score
                best_t = t

        return best_t

    def train(self, X_train, y_train, X_val, y_val):

        print("\n========== REUSE TRAINING ==========")

        best_f1 = 0
        patience = 5
        counter = 0
        best_model = None

        for epoch in range(20):

            print(f"\nEpoch {epoch}")

            self.model.train()
            idx = torch.randperm(len(X_train))

            for i in range(0, len(X_train), self.params["batch"]):
                j = idx[i:i + self.params["batch"]]

                xb, yb = X_train[j], y_train[j]

                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()

                if i == 0:
                    print("Batch loss:", loss.item())

            self.scheduler.step()

            # ===== VALIDATION =====
            self.model.eval()
            with torch.no_grad():
                probs = torch.softmax(self.model(X_val), dim=1)[:, 1].numpy()

            best_t = self._find_best_threshold(probs, y_val.numpy())
            preds = (probs > best_t).astype(int)

            f1 = f1_score(y_val.numpy(), preds)
            precision = precision_score(y_val.numpy(), preds, zero_division=0)

            print(f"F1: {f1:.4f} | Precision: {precision:.4f} | T={best_t:.2f}")

            if f1 > best_f1:
                best_f1 = f1
                self.best_threshold = max(best_t, 0.6)  # 🔥 защита
                counter = 0
                best_model = self.model.state_dict()
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping")
                    break

        self.model.load_state_dict(best_model)

        BASE_DIR = os.path.dirname(__file__)
        MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../models"))

        os.makedirs(MODEL_DIR, exist_ok=True)

        torch.save({
            "model": self.model.state_dict(),
            "threshold": self.best_threshold
        }, os.path.join(MODEL_DIR, "reuse_model.pt"))

        print("\nModel saved with threshold:", self.best_threshold)

        return self.model