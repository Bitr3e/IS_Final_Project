"""
Paradigm 1 - SUPERVISED LEARNING: learn the best move from labeled positions.

Labels come from a perfect (minimax) player, so every training example has its
correct answer. Four classifiers are compared; the best one is used for the
live example.
"""

import numpy as np
import plotting                      # picks the Agg backend when --save is used
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config
from game import (EXAMPLE, PRIORITY, all_positions, legal, optimal_moves,
                  play, text_board, to_move, winner)
from plotting import banner, draw_board, finish_plot, square_heatmap

SEED = config.SEED


def threats(b, who):
    """1 for every empty square where `who` would win immediately."""
    return [1 if b[i] == 0 and winner(play(b, i, who)) == who else 0
            for i in range(9)]


def encode(b):
    """45 inputs from the point of view of the player to move:
    9 'my piece' flags, 9 'opponent piece' flags, 9 'empty' flags,
    9 'I can win here' flags, 9 'I must block here' flags."""
    p = to_move(b)
    mine = [1 if c == p else 0 for c in b]
    opp = [1 if c == -p else 0 for c in b]
    empty = [1 if c == 0 else 0 for c in b]
    return mine + opp + empty + threats(b, p) + threats(b, -p)


def move_scores(model, Xm):
    """Score for each of the 9 squares (-inf for squares the model never saw)."""
    raw = (model.predict_proba(Xm) if hasattr(model, "predict_proba")
           else model.decision_function(Xm))
    full = np.full((len(Xm), 9), -np.inf)
    full[:, model.classes_] = raw
    return full


def run_supervised():
    banner("SUPERVISED LEARNING: Learning the Best Tic-Tac-Toe Move")

    positions = [b for b in all_positions() if winner(b) == 0 and legal(b)]
    X = np.array([encode(b) for b in positions])
    opt = [optimal_moves(b) for b in positions]
    y = np.array([min(o, key=PRIORITY.index) for o in opt])   # the label

    print(f"Dataset : {len(positions)} game positions (every reachable board "
          f"where the game is still going)")
    print("Input   : 45 numbers describing the board (pieces, empty squares, "
          "'I can win here', 'I must block here')")
    print("Label   : the best move, computed by a perfect minimax player "
          "(every example has its correct answer)")

    idx = np.arange(len(positions))
    tr, te = train_test_split(idx, test_size=0.25, random_state=SEED)
    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
    empty_mask = X_te[:, 18:27] == 1                # columns 18-26 = 'empty' flags

    models = {
        "SVM (RBF kernel)": SVC(kernel="rbf", C=10, gamma="scale"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }

    print(f"\nTrained on {len(tr)} positions, tested on {len(te)} positions "
          f"the models have NEVER seen:")
    print(f"  {'Model':<18}{'Optimal move':>14}{'Exact label':>14}")
    optimal_rate, exact_rate = {}, {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        scores = move_scores(model, X_te)
        scores[~empty_mask] = -np.inf               # never play on a taken square
        pred = scores.argmax(axis=1)
        optimal_rate[name] = np.mean([pred[i] in opt[te[i]] for i in range(len(te))])
        exact_rate[name] = np.mean(pred == y_te)
        print(f"  {name:<18}{optimal_rate[name] * 100:13.1f}%{exact_rate[name] * 100:13.1f}%")
    print("  ('Optimal move' = the chosen move is one a perfect player could "
          "also choose;\n   several squares are often equally good.)")

    best = max(optimal_rate, key=optimal_rate.get)
    print(f"\nBest model: {best}")

    # Live example: X threatens to win on square 3, so O must block there.
    example = EXAMPLE
    sc = move_scores(models[best], np.array([encode(example)]))[0]
    sc[[i for i in range(9) if example[i] != 0]] = -np.inf
    choice = int(sc.argmax())
    good = [m + 1 for m in optimal_moves(example)]
    print("\nLive example (O to move, X threatens to win on square 3):")
    print(text_board(example))
    print(f"  {best} plays square {choice + 1}; "
          f"perfect-play moves are {good}  -> "
          f"{'GOOD' if choice + 1 in good else 'MISS'}")

    # ---- Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    names = list(optimal_rate)
    vals = [optimal_rate[n] * 100 for n in names]
    axes[0].bar(names, vals, color=["#2a6fdb", "#50a050", "#f0a030", "#8fb4f0"])
    axes[0].set_ylim(max(0, min(vals) - 10), 100)
    axes[0].set_ylabel("Optimal-move rate on unseen positions (%)")
    axes[0].set_title("Algorithm comparison")
    axes[0].tick_params(axis="x", labelsize=8)
    for k, v in enumerate(vals):
        axes[0].text(k, v + 0.4, f"{v:.1f}%", ha="center")

    rf = models["Random Forest"]
    imp = rf.feature_importances_
    square_heatmap(axes[1], sum(imp[k * 9:(k + 1) * 9] for k in range(5)),
                   "What the Random Forest looks at\n(importance of each square)",
                   fmt="{:.3f}")
    draw_board(axes[2], example, highlight=choice,
               title=f"{best} recommends square {choice + 1} for O (green)")

    fig.suptitle("Supervised Learning - Best-Move Classifier", fontweight="bold", y=1.04)
    finish_plot(fig, "1_supervised_best_move.png")


if __name__ == "__main__":
    run_supervised()
