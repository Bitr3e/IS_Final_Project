"""
Paradigm 2 - UNSUPERVISED LEARNING: cluster board positions with no labels.

K-Means groups ~5,000 boards using 8 simple features, and PCA squashes those
features to 2 dimensions so the groups can be plotted. The algorithm is never
told the move number, the winner or the best move.
"""

import numpy as np
import plotting                      # picks the Agg backend when --save is used
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import config
from game import CORNERS, EDGES, all_positions
from plotting import banner, finish_plot

SEED = config.SEED

FEATURES = ["X pieces", "O pieces", "X centre", "O centre",
            "X corners", "O corners", "X edges", "O edges"]


def strategic(b):
    return [b.count(1), b.count(-1), int(b[4] == 1), int(b[4] == -1),
            sum(b[i] == 1 for i in CORNERS), sum(b[i] == -1 for i in CORNERS),
            sum(b[i] == 1 for i in EDGES), sum(b[i] == -1 for i in EDGES)]


def run_unsupervised():
    banner("UNSUPERVISED LEARNING: Grouping Board Positions (K-Means + PCA)")

    positions = all_positions()
    X = np.array([strategic(b) for b in positions], dtype=float)
    print(f"Dataset : {len(positions)} board positions, 8 features each "
          f"({', '.join(FEATURES)})")
    print("Labels  : NONE. The algorithm is never told the move number, the "
          "winner or the best move.")

    Xs = StandardScaler().fit_transform(X)

    ks = list(range(2, 8))
    sils = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
        sils.append(silhouette_score(Xs, km.labels_))
    best_k = ks[int(np.argmax(sils))]
    print("\nChoosing K (silhouette score, higher = better separated):")
    for k, s in zip(ks, sils):
        print(f"  K={k}: {s:.3f}{'  <-- best' if k == best_k else ''}")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(Xs)
    labels = km.labels_

    stats = []
    for c in range(best_k):
        m = labels == c
        stats.append({"c": c, "n": int(m.sum()),
                      "pieces": float((X[m, 0] + X[m, 1]).mean()),
                      "xc": float(X[m, 2].mean()), "oc": float(X[m, 3].mean())})
    stats.sort(key=lambda s: s["pieces"])
    names = {}
    for rank, s in enumerate(stats, 1):
        phase = ("Opening" if s["pieces"] <= 3.5
                 else "Midgame" if s["pieces"] <= 6.5 else "Endgame")
        ctr = (", X holds centre" if s["xc"] > 0.6
               else ", O holds centre" if s["oc"] > 0.6 else "")
        names[s["c"]] = f"C{rank}: {phase}{ctr}"

    print(f"\nK-Means discovered {best_k} types of position on its own:")
    print(f"  {'Cluster':<28}{'Boards':>7}{'Avg pieces':>12}{'X centre':>10}{'O centre':>10}")
    for s in stats:
        print(f"  {names[s['c']]:<28}{s['n']:>7}{s['pieces']:>12.1f}"
              f"{s['xc'] * 100:>9.0f}%{s['oc'] * 100:>9.0f}%")

    pca = PCA(n_components=2).fit(Xs)
    Z = pca.transform(Xs)
    var = pca.explained_variance_ratio_
    print(f"\nPCA: 8 features -> 2 components, keeping {var.sum() * 100:.1f}% "
          f"of the variance")

    palette = plt.cm.tab10(np.arange(best_k))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].plot(ks, sils, "s-", color="#f0a030")
    axes[0].axvline(best_k, ls="--", color="gray")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Silhouette score")
    axes[0].set_title(f"Choosing K (best K = {best_k})")

    for s in stats:
        m = labels == s["c"]
        axes[1].scatter(Z[m, 0], Z[m, 1], s=8, color=palette[s["c"]],
                        label=names[s["c"]], alpha=0.7)
    axes[1].set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    axes[1].set_title("Board positions in PCA space")
    axes[1].legend(fontsize=7, markerscale=2)

    axes[2].bar([names[s["c"]].split(":")[0] for s in stats],
                [s["pieces"] for s in stats],
                color=[palette[s["c"]] for s in stats])
    axes[2].set_ylabel("Average pieces on the board")
    axes[2].set_title("Clusters line up with how far the game has gone")
    for k, s in enumerate(stats):
        axes[2].text(k, s["pieces"] + 0.1, f"{s['pieces']:.1f}", ha="center")

    fig.suptitle("Unsupervised Learning - K-Means + PCA on Board Positions",
                 fontweight="bold", y=1.04)
    finish_plot(fig, "2_unsupervised_kmeans_pca.png")


if __name__ == "__main__":
    run_unsupervised()
