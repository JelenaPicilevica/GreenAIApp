import json
import csv
import os
import urllib.request
from tqdm import tqdm
from openai import OpenAI


class CacheBuilder:

    def __init__(self, squad_path, output_path):
        self.squad_path = squad_path
        self.output_path = output_path
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ---------------- DOWNLOAD ----------------

    def ensure_squad(self):
        if not os.path.exists(self.squad_path):
            print("Downloading SQuAD dataset...")

            os.makedirs(os.path.dirname(self.squad_path), exist_ok=True)

            url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
            urllib.request.urlretrieve(url, self.squad_path)

            print("✅ Download complete")

    # ---------------- LOAD ----------------

    def load_squad(self, limit=100000):
        self.ensure_squad()

        with open(self.squad_path, encoding="utf-8") as f:
            data = json.load(f)

        samples = []

        for article in data["data"]:
            for p in article["paragraphs"]:
                for qa in p["qas"]:
                    if qa["answers"]:
                        samples.append({
                            "prompt": qa["question"].strip(),
                            "answer": qa["answers"][0]["text"].strip()
                        })

                        if len(samples) >= limit:
                            return samples

        return samples

    # ---------------- EMBEDDINGS ----------------

    def embed_batch(self, texts):
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [e.embedding for e in response.data]

    # ---------------- BUILD ----------------

    def build(self, limit=100000, batch_size=200):

        print("📥 Loading SQuAD...")
        data = self.load_squad(limit)

        prompts = [x["prompt"] for x in data]

        print(f"🔢 Total prompts: {len(prompts)}")
        print("🔄 Generating embeddings...")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt", "answer", "embedding"])

            idx = 0

            for i in tqdm(range(0, len(prompts), batch_size)):
                batch = prompts[i:i+batch_size]

                try:
                    emb_batch = self.embed_batch(batch)

                    # Safety check
                    if len(emb_batch) != len(batch):
                        print(f"⚠️ Mismatch at batch {i}, skipping...")
                        continue

                    for j, emb in enumerate(emb_batch):
                        item = data[idx]

                        writer.writerow([
                            item["prompt"],
                            item["answer"],
                            json.dumps(emb)
                        ])

                        idx += 1

                except Exception as e:
                    print(f"❌ Error at batch {i}: {e}")
                    continue

        print(f"✅ DONE: saved {idx} records")