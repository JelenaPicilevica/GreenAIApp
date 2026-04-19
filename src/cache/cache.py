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

            for row in reader:
                try:
                    question = row[0]
                    answer = row[1]

                    # parse embedding safely
                    embedding = np.array(ast.literal_eval(row[2]), dtype=np.float32)

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

        # ensure list format (NOT numpy)
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        # stringify like in dataset
        embedding_str = str(embedding)

        # append to CSV
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                question,
                answer,
                embedding_str
            ])

        # also keep in memory
        self.data.append({
            "question": question,
            "answer": answer,
            "embedding": np.array(embedding, dtype=np.float32)
        })

        print("✅ Saved to cache (correct format)")