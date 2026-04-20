import os
import csv
import ast
import numpy as np


class Cache:

    def __init__(self, path):
        self.path = path
        self.data = []

        if os.path.exists(self.path):
            print(f"LOADING CACHE FROM: {self.path}")
            self.load()
        else:
            print(" Cache file not found")

    # =====================
    # LOAD
    # =====================
    def load(self):

        with open(self.path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)

            for i, row in enumerate(reader):
                try:
                    question = row[0]
                    answer = row[1]

                    emb_raw = row[2].strip('"')

                    embedding = np.array(
                        ast.literal_eval(emb_raw),
                        dtype=np.float32
                    )

                    self.data.append({
                        "question": question,
                        "answer": answer,
                        "embedding": embedding
                    })

                except Exception as e:
                    print(f"Skipping row: {e}")

        print(f"✅ CACHE LOADED: {len(self.data)}")

    # =====================
    # SAVE
    # =====================
    def add(self, question, answer, embedding):

        # sanitize text (no commas/newlines)
        question = (
            question.replace("\n", " ")
                    .replace("\r", " ")
                    .replace(",", " ")
        )

        answer = (
            answer.replace("\n", " ")
                  .replace("\r", " ")
                  .replace("<br>", " ")
                  .replace(",", " ")
        )

        # ensure list
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        # embedding wrapped in quotes
        embedding_str = f"\"{embedding}\""

        line = f"{question},{answer},{embedding_str}\n"

        # ensuring newline before append
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path, "rb+") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)

        # update memory
        self.data.append({
            "question": question,
            "answer": answer,
            "embedding": np.array(embedding, dtype=np.float32)
        })

        print("✅ Saved to cache")