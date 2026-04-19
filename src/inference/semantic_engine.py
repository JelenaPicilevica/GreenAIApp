import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from openai import OpenAI
from src.cache.cache import Cache

client = OpenAI()


class SemanticEngine:

    def __init__(self):

        base = os.getcwd()

        self.cache = Cache(os.path.join(base, "data", "cache_dataset.csv"))

        try:
            import joblib

            self.pipeline = joblib.load(
                os.path.join(base, "models", "drift_feature_pipeline.pkl")
            )

            self.model = torch.load(
                os.path.join(base, "models", "drift_model.pt"),
                map_location="cpu",
                weights_only=False
            )

            self.model.eval()

        except:
            self.pipeline = None
            self.model = None

    # =====================
    def normalize(self, t):
        return t.lower().strip().replace("?", "")

    # =====================
    def embed(self, text):
        r = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        e = np.array(r.data[0].embedding, dtype=np.float32)
        return e / np.linalg.norm(e)

    # =====================
    def similarity(self, a, b):
        return float(np.dot(a, b))

    # =====================
    def drift(self, q, p):

        if self.model is None:
            return 0.0

        try:
            df = pd.DataFrame([{
                "sentence1": q,
                "sentence2": p,
                "drift": 0
            }])

            X, _ = self.pipeline.transform(df)

            if hasattr(X, "toarray"):
                X = X.toarray()

            x = torch.tensor(X, dtype=torch.float32)

            with torch.no_grad():
                probs = F.softmax(self.model(x), dim=1)

            return float(probs[0][1])

        except:
            return 0.0

    # =====================
    def analyze(self, query):

        qn = self.normalize(query)

        for item in self.cache.data:
            if qn == self.normalize(item["question"]):
                return {"status": "auto", "answer": item["answer"]}

        emb = self.embed(query)

        nd, d = [], []

        for item in self.cache.data:

            if item["embedding"].shape != emb.shape:
                continue

            score = self.similarity(emb, item["embedding"])

            if score >= 0.5:

                drift_val = self.drift(query, item["question"])

                rec = {
                    "question": item["question"],
                    "answer": item["answer"],
                    "score": score,
                    "drift": drift_val
                }

                (nd if drift_val < 0.5 else d).append(rec)

        if not nd and not d:
            return {"status": "no_match"}

        return {
            "status": "candidates",
            "no_drift": sorted(nd, key=lambda x: x["score"], reverse=True)[:3],
            "drifted": sorted(d, key=lambda x: x["score"], reverse=True)[:3]
        }

    # =====================
    def call_llm(self, messages):

        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        return r.choices[0].message.content

    # =====================
    def is_clarification(self, text):

        triggers = ["clarify", "more details", "what do you mean", "specify"]

        return any(t in text.lower() for t in triggers)

    # =====================
    # CONTEXT-AWARE QUESTION
    # =====================
    def build_question(self, conversation):

        formatted = []

        for m in conversation:
            role = m["role"].upper()
            formatted.append(f"{role}: {m['content']}")

        context = "\n".join(formatted)

        question = self.call_llm([
            {
                "role": "system",
                "content": (
                    "Extract the FINAL, CLEAN, standalone user question.\n"
                    "Remove typos and use clarified meaning.\n"
                    "Return ONLY the question."
                )
            },
            {"role": "user", "content": context}
        ])

        return question.strip().replace("\n", "").replace('"', '')

    # =====================
    def save_to_cache(self, conversation, answer):

        q = self.build_question(conversation)

        short = self.call_llm([
            {"role": "system", "content": "Summarize in max 3 sentences."},
            {"role": "user", "content": answer}
        ])

        emb = self.embed(q)

        self.cache.add(q, short, emb)