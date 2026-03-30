import torch.nn as nn


class ModelBuilder:

    @staticmethod
    def build(input_dim, n_layers, hidden, dropout):

        print("\nBuilding model...")
        print("Input dim:", input_dim)

        layers = []
        in_dim = input_dim

        for i in range(n_layers):
            print(f"Layer {i}: {in_dim} -> {hidden}")

            layers += [
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                nn.Dropout(dropout)
            ]

            in_dim = hidden
            hidden = hidden // 2

        layers.append(nn.Linear(in_dim, 2))

        return nn.Sequential(*layers)