import numpy as np
import torch

from sklearn.metrics import f1_score


class Trainer:

    """
    Handles neural network training loop including:
    - optimization (AdamW)
    - learning rate scheduling (CosineAnnealing)
    - class-weighted loss to penalize false positives
    - mini-batch training
    - validation using F1 score with threshold search
    - early stopping based on validation performance
    - saving best model weights

    Output:
    - trained PyTorch model with best validation F1
    """

    def __init__(self, model, params):
        self.model = model
        self.params = params

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["wd"]
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20
        )

        # Сlass weights to reduce FP
        # This is done as predicting drift incorrectly (FP) is costly
        self.loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([2.0, 1.0])
        )

    def train(self, X_train, y_train, X_val, y_val):

        print("\n========== TRAINING START ==========")

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

            with torch.no_grad():
                probs = torch.softmax(self.model(X_val), dim=1)[:, 1].numpy()

            best_epoch = 0
            for t in np.linspace(0.2, 0.8, 50):
                preds = (probs > t).astype(int)
                best_epoch = max(best_epoch, f1_score(y_val.numpy(), preds))

            print("F1:", best_epoch)

            if best_epoch > best_f1:
                best_f1 = best_epoch
                counter = 0
                best_model = self.model.state_dict()
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping")
                    break

        self.model.load_state_dict(best_model)

        print("\nTraining finished (no saving at this stage)")

        return self.model