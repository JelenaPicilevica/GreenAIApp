import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification


class TransformerFallback:
    """
    Transformer trained on PAWS for DRIFT detection
    """

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_path=None,
        device=None,
        max_length=128
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        print("\n[TransformerFallback] Loading model...")

        # =====================
        # PATH
        # =====================
        if model_path is None:
            BASE_DIR = os.path.dirname(__file__)
            model_path = os.path.join(BASE_DIR, "../../../models/best_model_transformer.pt")

        print(f"[TransformerFallback] Loading weights from: {model_path}")

        # 🔥 safety check (very useful for debugging)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # =====================
        # TOKENIZER
        # =====================
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # =====================
        # 🔥 CORRECT MODEL INIT
        # =====================
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 2  # must match training

        self.model = AutoModelForSequenceClassification.from_config(config)

        # =====================
        # 🔥 LOAD TRAINED WEIGHTS
        # =====================
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        print(f"[TransformerFallback] Ready on {self.device}")

    def predict_drift(self, s1, s2):
        with torch.no_grad():

            inputs = self.tokenizer(
                s1,
                s2,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            logits = outputs.logits

            # model is trained for drift → take probability of class 1
            drift_prob = torch.softmax(logits, dim=1)[0, 1].item()

        return drift_prob

    def is_drift(self, s1, s2, threshold=0.5):
        drift_prob = self.predict_drift(s1, s2)
        return drift_prob >= threshold, drift_prob