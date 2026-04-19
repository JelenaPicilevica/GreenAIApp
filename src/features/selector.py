"""
Feature selection + explainability

What this class does:
- Selects best subset of features using L1 LogisticRegression
- Shows top important features
- Shows correct feature group importance (mean / max / count)
- Explains which manual features are actually used

Designed specifically for current FeatureBuilder
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


class FeatureSelector:

    def __init__(self, C=0.05):
        self.C = C
        self.selected_idx = None

        # feature size from FeatureBuilder
        self.dim_diff = 1000
        self.dim_prod = 1000
        self.dim_char = 500
        self.dim_svd = 128
        self.dim_cos = 2
        self.dim_manual = 12

        self._build_ranges()

    def _build_ranges(self):
        i = 0
        self.ranges = {}

        self.ranges["diff"] = (i, i + self.dim_diff)
        i += self.dim_diff

        self.ranges["prod"] = (i, i + self.dim_prod)
        i += self.dim_prod

        self.ranges["char"] = (i, i + self.dim_char)
        i += self.dim_char

        self.ranges["svd"] = (i, i + self.dim_svd)
        i += self.dim_svd

        self.ranges["cos"] = (i, i + self.dim_cos)
        i += self.dim_cos

        self.ranges["manual"] = (i, i + self.dim_manual)

    def _get_group(self, idx):
        for name, (start, end) in self.ranges.items():
            if start <= idx < end:
                return name
        return "unknown"

    def fit(self, X_train, y_train, X_val, y_val):

        print("\n" + "=" * 60)
        print(" AUTO FEATURE SELECTION (L1 + SAGA)")
        print("=" * 60)

        # =====================
        # BASE MODEL
        # =====================
        lr = LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=2000,
            n_jobs=-1,
            C=self.C
        )

        lr.fit(X_train, y_train)

        weights = lr.coef_[0]
        importance = np.abs(weights)
        idx_sorted = np.argsort(importance)[::-1]

        total = X_train.shape[1]

        print(f"\n Total features: {total}")

        zero_features = (importance < 1e-6).sum()
        print(f" Zeroed by L1: {zero_features} ({100 * zero_features / total:.1f}%)")

        # =====================
        #  GROUP ANALYSIS
        # =====================
        print("\n Feature group analysis:")

        group_stats = {
            g: {"sum": 0.0, "count": 0, "max": 0.0}
            for g in self.ranges.keys()
        }

        for i in range(len(importance)):
            g = self._get_group(i)
            val = importance[i]

            group_stats[g]["sum"] += val
            group_stats[g]["count"] += 1
            group_stats[g]["max"] = max(group_stats[g]["max"], val)

        print("\nGroup     | count | mean_imp | max_imp")
        print("-------------------------------------------")

        sorted_groups = sorted(
            group_stats.items(),
            key=lambda x: x[1]["sum"] / x[1]["count"],
            reverse=True
        )

        for g, stats in sorted_groups:
            mean_imp = stats["sum"] / stats["count"]
            print(f"{g:<9} | {stats['count']:5d} | {mean_imp:8.4f} | {stats['max']:7.4f}")

        # =====================
        #  MANUAL FEATURES ANALYSIS
        # =====================
        manual_names = [
            "jaccard",
            "neg_diff",
            "num_diff",
            "num_overlap",
            "word_overlap",
            "word_diff",
            "inv_norm",
            "bigram",
            "align1",
            "align2",
            "len_ratio",
            "interaction"
        ]

        start, _ = self.ranges["manual"]

        print("\n Manual feature usage:")

        used = []
        unused = []

        for i, name in enumerate(manual_names):
            idx = start + i
            w = weights[idx]

            if abs(w) > 1e-6:
                used.append((name, w))
            else:
                unused.append(name)

        print("\n USED manual features:")
        for name, w in sorted(used, key=lambda x: abs(x[1]), reverse=True):
            print(f"{name:<15} | weight={w:+.4f}")

        print("\n UNUSED manual features:")
        if unused:
            for name in unused:
                print(name)
        else:
            print("None (all used)")

        # =====================
        # 🔍 SEARCH BEST SUBSET
        # =====================
        ratios = [0.3, 0.4, 0.5, 0.6, 0.7]

        best_f1 = 0
        best_idx = None
        best_k = total

        print("\n🔍 Feature subset search:\n")

        for r in ratios:
            k = int(total * r)
            selected = idx_sorted[:k]

            X_tr = X_train[:, selected]
            X_v = X_val[:, selected]

            lr_tmp = LogisticRegression(
                solver="saga",
                max_iter=1000,
                n_jobs=-1
            )

            lr_tmp.fit(X_tr, y_train)

            preds = lr_tmp.predict(X_v)
            f1 = f1_score(y_val, preds)

            print(f"{k:4d} features -> F1 {f1:.4f}")

            if (f1 > best_f1) or (abs(f1 - best_f1) < 1e-4 and k < best_k):
                best_f1 = f1
                best_idx = selected
                best_k = k

        # =====================
        # FINAL
        # =====================
        print("\n" + "-" * 60)
        print(" FINAL SELECTION")
        print("-" * 60)
        print(f" Best F1:        {best_f1:.4f}")
        print(f" Features used:  {best_k}")
        print(f" Reduction:      {100 * (1 - best_k / total):.1f}%")

        self.selected_idx = best_idx
        return best_idx

    def transform(self, X):
        return X[:, self.selected_idx]