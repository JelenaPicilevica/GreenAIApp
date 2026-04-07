import torch
from datasets import load_dataset
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================
# CONFIG
# =====================
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5


# =====================
# DATASET
# =====================
class PawsDataset(Dataset):
    def __init__(self, sentence1, sentence2, labels, tokenizer, max_len=128):
        self.labels = labels

        # Tokenization
        self.encodings = tokenizer(
            sentence1,
            sentence2,
            truncation=True,
            padding=True,
            max_length=max_len
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# =====================
# TRAIN
# =====================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for i, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 50 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")

    return total_loss / len(loader)


# =====================
# EVALUATION
# =====================
def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    f1 = f1_score(all_labels, all_preds)
    print(f"📊 Val F1: {f1:.4f}")
    return f1


# =====================
# MAIN
# =====================
def main():
    print("🚀 START")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # =====================
    # LOAD DATA
    # =====================
    print("Loading dataset...")
    dataset = load_dataset("paws", "labeled_final")
    print("Dataset loaded!")

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()

    # =====================
    # TOKENIZER
    # =====================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # =====================
    # LABELS
    # =====================
    train_labels = (train_df["label"] == 0).astype(int).tolist()
    val_labels = (val_df["label"] == 0).astype(int).tolist()

    # =====================
    # DATASETS
    # =====================
    print("Tokenizing...")

    train_ds = PawsDataset(
        train_df["sentence1"].tolist(),
        train_df["sentence2"].tolist(),
        train_labels,
        tokenizer,
        MAX_LEN
    )

    val_ds = PawsDataset(
        val_df["sentence1"].tolist(),
        val_df["sentence2"].tolist(),
        val_labels,
        tokenizer,
        MAX_LEN
    )

    print("Tokenization done!")

    # =====================
    # DATALOADERS
    # =====================
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    # =====================
    # MODEL
    # =====================
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # =====================
    # TRAIN LOOP
    # =====================
    best_f1 = 0

    for epoch in range(EPOCHS):
        print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")

        loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Loss: {loss:.4f}")

        val_f1 = evaluate(model, val_loader, device)

        # Saving best
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "../../../../models/best_model_transformer.pt")
            print(" Saved best model")

    print("\n🏁 Done")
    print("Best F1:", best_f1)


if __name__ == "__main__":
    main()