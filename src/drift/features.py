"""
Feature engineering with logging.
"""

import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class FeatureBuilder:

    def __init__(self):
        self.word_tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.char_tfidf = TfidfVectorizer(max_features=500, analyzer='char', ngram_range=(3, 5))
        self.svd = TruncatedSVD(n_components=128, random_state=42)
        self.neg_words = {"not", "no", "never", "n't"}

    def fit(self, train_df):
        print("\nFitting TF-IDF + SVD...")

        train_text = list(train_df["sentence1"]) + list(train_df["sentence2"])

        self.word_tfidf.fit(train_text)
        self.char_tfidf.fit(train_text)
        self.svd.fit(self.word_tfidf.transform(train_text))

    def build(self, df):

        print(f"\nBuilding features for {len(df)} samples...")

        y = df["label"].values

        w1 = self.word_tfidf.transform(df["sentence1"])
        w2 = self.word_tfidf.transform(df["sentence2"])

        c1 = self.char_tfidf.transform(df["sentence1"])
        c2 = self.char_tfidf.transform(df["sentence2"])

        # sparse diff
        diff = w1 - w2
        diff.data = np.abs(diff.data)

        char_diff = c1 - c2
        char_diff.data = np.abs(char_diff.data)

        # sparse product
        prod = w1.multiply(w2)

        w1n, w2n = normalize(w1), normalize(w2)
        cos_tfidf = np.sum(w1n.multiply(w2n), axis=1).A1

        e1 = self.svd.transform(w1)
        e2 = self.svd.transform(w2)

        cos_svd = np.sum(normalize(e1) * normalize(e2), axis=1)
        diff_svd = np.abs(e1 - e2)

        manual = []

        for i, (a, b) in enumerate(zip(df["sentence1"], df["sentence2"])):

            l1, l2 = a.lower().split(), b.lower().split()
            s1, s2 = set(l1), set(l2)

            jaccard = len(s1 & s2) / (len(s1 | s2) + 1e-6)

            neg_diff = abs(
                sum(w in self.neg_words for w in l1) -
                sum(w in self.neg_words for w in l2)
            )

            nums1 = re.findall(r"\d+", a)
            nums2 = re.findall(r"\d+", b)

            num_diff = len(set(nums1) ^ set(nums2))
            num_overlap = len(set(nums1) & set(nums2))

            word_overlap = len(s1 & s2)
            word_diff = len(s1 ^ s2)

            pos = {w: i for i, w in enumerate(l2)}
            pairs = [(i, pos[w]) for i, w in enumerate(l1) if w in pos]

            inv = 0
            for i1 in range(len(pairs)):
                for j1 in range(i1 + 1, len(pairs)):
                    if pairs[i1][1] > pairs[j1][1]:
                        inv += 1

            inv_norm = inv / (len(pairs) + 1e-6)

            b1 = set(zip(l1, l1[1:]))
            b2 = set(zip(l2, l2[1:]))

            bigram = len(b1 & b2) / (len(b1 | b2) + 1e-6)

            align1 = sum(1 for w in l1 if w in s2) / (len(l1) + 1e-6)
            align2 = sum(1 for w in l2 if w in s1) / (len(l2) + 1e-6)

            len_ratio = min(len(l1), len(l2)) / (max(len(l1), len(l2)) + 1e-6)

            interaction = cos_tfidf[i] * cos_svd[i]

            manual.append([
                jaccard, neg_diff,
                num_diff, num_overlap,
                word_overlap, word_diff,
                inv_norm, bigram,
                align1, align2,
                len_ratio,
                interaction
            ])

        manual = np.array(manual) * 3.0

        X = hstack([
            diff,
            prod,
            char_diff,
            csr_matrix(diff_svd),
            csr_matrix(cos_tfidf.reshape(-1, 1)),
            csr_matrix(cos_svd.reshape(-1, 1)),
            csr_matrix(manual)
        ]).tocsr()

        print("\n[FEATURES DEBUG]")
        print("Shape:", X.shape)

        return X.astype(np.float32), y