"""
Tic-Tac-Toe Bot - Three Machine Learning Paradigms (Final Lab Project)
======================================================================
One game, three ways of learning:

Supervised    : SVM / Random Forest / Decision Tree / KNN learn to pick the
                best move. Labels come from a perfect (minimax) player.
Unsupervised  : K-Means + PCA group ~5,000 board positions WITHOUT labels.
Reinforcement : Q-Learning bot teaches itself the game by self-play, then
                you can play against it.

Run:
    python ml_3paradigm.py                # interactive menu (live demo)
    python ml_3paradigm.py --all          # run the three paradigms in order
    python ml_3paradigm.py --all --save   # no pop-up windows, only save PNGs
"""

import argparse
import os
import random
import sys
from functools import lru_cache

import numpy as np
import matplotlib

if "--save" in sys.argv:           # choose the backend before pyplot loads
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
OUT_DIR = "outputs"
SAVE_ONLY = False

# --------------------------------------------------------------------------
# Game basics.  A board is a tuple of 9 numbers: 1 = X, -1 = O, 0 = empty.
# Squares are numbered 0-8 in code and shown to the player as 1-9.
# X always moves first.
# --------------------------------------------------------------------------
EMPTY = (0,) * 9
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
         (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
CORNERS, EDGES = [0, 2, 6, 8], [1, 3, 5, 7]
PRIORITY = [4, 0, 2, 6, 8, 1, 3, 5, 7]       # tie-break: centre, corners, edges
SYMBOL = {1: "X", -1: "O", 0: "."}
# Demo position used in the supervised and reinforcement parts: O to move,
# X has squares 1 and 2 and threatens to complete the top row on square 3.
EXAMPLE = (1, 1, 0, 0, -1, 0, 0, 0, 0)


def winner(b):
    for a, c, d in LINES:
        if b[a] != 0 and b[a] == b[c] == b[d]:
            return b[a]
    return 0


def to_move(b):
    return 1 if b.count(1) == b.count(-1) else -1


def legal(b):
    return [i for i in range(9) if b[i] == 0]


def play(b, i, p):
    l = list(b)
    l[i] = p
    return tuple(l)


@lru_cache(maxsize=None)
def minimax_value(b):
    """Value for the player to move: +1 forced win, 0 draw, -1 forced loss."""
    if winner(b) != 0:
        return -1                      # the previous player just won
    moves = legal(b)
    if not moves:
        return 0
    p = to_move(b)
    return max(-minimax_value(play(b, m, p)) for m in moves)


def optimal_moves(b):
    p = to_move(b)
    vals = {m: -minimax_value(play(b, m, p)) for m in legal(b)}
    best = max(vals.values())
    return [m for m, v in vals.items() if v == best]


def all_positions():
    """Every board reachable in a legal game (including finished ones)."""
    seen, stack = {EMPTY}, [EMPTY]
    while stack:
        b = stack.pop()
        if winner(b) or not legal(b):
            continue
        p = to_move(b)
        for m in legal(b):
            nb = play(b, m, p)
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return sorted(seen)


def text_board(b):
    rows = []
    for r in range(3):
        cells = [SYMBOL[b[3 * r + c]] if b[3 * r + c] else str(3 * r + c + 1)
                 for c in range(3)]
        rows.append(" " + " | ".join(cells))
    return "\n---+---+---\n".join(rows)


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
def banner(title):
    line = "=" * 66
    print(f"\n{line}\n  {title}\n{line}")


def finish_plot(fig, filename):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  [plot saved -> {path}]")
    if not SAVE_ONLY:
        plt.show()
    plt.close(fig)


def draw_board(ax, b, highlight=None, title=""):
    ax.set_xlim(0, 3)
    ax.set_ylim(3, 0)
    ax.set_aspect("equal")
    for i in range(9):
        r, c = divmod(i, 3)
        face = "#bfeabf" if i == highlight else "white"
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=face, edgecolor="black"))
        if b[i] != 0:
            ax.text(c + .5, r + .5, SYMBOL[b[i]], ha="center", va="center",
                    fontsize=34, fontweight="bold",
                    color="#d03030" if b[i] == 1 else "#2a6fdb")
        else:
            ax.text(c + .5, r + .5, str(i + 1), ha="center", va="center",
                    fontsize=12, color="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def draw_q_values(ax, b, q, title=""):
    """Board with the bot's Q-value written in every empty square."""
    moves = legal(b)
    best = max(moves, key=lambda i: q[i])
    draw_board(ax, b, highlight=best, title=title)
    for i in moves:
        r, c = divmod(i, 3)
        ax.texts[[t.get_text() for t in ax.texts].index(str(i + 1))].set_text("")
        ax.text(c + .5, r + .5, f"{q[i]:+.2f}", ha="center", va="center",
                fontsize=13, fontweight="bold" if i == best else "normal")


def square_heatmap(ax, values, title, fmt="{:.2f}", cmap="Blues"):
    grid = np.asarray(values).reshape(3, 3)
    ax.imshow(grid, cmap=cmap)
    for (r, c), v in np.ndenumerate(grid):
        ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=12,
                color="white" if v > (grid.max() + grid.min()) / 2 else "black")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


# --------------------------------------------------------------------------
# 1) SUPERVISED LEARNING - learn the best move from labeled positions
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# 2) UNSUPERVISED LEARNING - cluster board positions with no labels
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# 3) REINFORCEMENT LEARNING - Q-Learning by self-play
# --------------------------------------------------------------------------
def random_policy(b):
    return random.choice(legal(b))


def minimax_policy(b):
    return random.choice(optimal_moves(b))


def greedy_move(Q, b):
    moves = legal(b)
    q = Q.get(b)
    if q is None:
        return random.choice(moves)
    best = max(q[i] for i in moves)
    return random.choice([i for i in moves if q[i] == best])


def play_game(policy_x, policy_o):
    b = EMPTY
    while True:
        p = to_move(b)
        b = play(b, (policy_x if p == 1 else policy_o)(b), p)
        w = winner(b)
        if w:
            return w
        if not legal(b):
            return 0


def evaluate(bot, opponent, n=200):
    """Bot plays n games as X and n games as O. Returns (win, draw, loss)."""
    w = d = l = 0
    for seat in (1, -1):
        for _ in range(n):
            res = play_game(bot, opponent) if seat == 1 else play_game(opponent, bot)
            if res == 0:
                d += 1
            elif res == seat:
                w += 1
            else:
                l += 1
    t = 2 * n
    return w / t, d / t, l / t


def train_self_play(episodes=60000, alpha=0.3, gamma=0.95, eval_every=3000):
    """Both sides share one Q-table (always from the mover's point of view).
    Reward: +1 for winning move, 0 for a draw, small discount for every step.
    A loss appears as -1 because the opponent's win is backed up with a
    flipped sign (negamax form of the Q-learning update)."""
    rng = random.Random(SEED)
    Q, curve = {}, []

    def snapshot(ep):
        w, d, l = evaluate(lambda b: greedy_move(Q, b), random_policy, n=150)
        curve.append((ep, w, d, l))

    snapshot(0)
    for ep in range(1, episodes + 1):
        eps = max(0.05, 1 - ep / (episodes * 0.6))      # explore less over time
        b = EMPTY
        while True:
            p = to_move(b)
            moves = legal(b)
            q = Q.setdefault(b, np.zeros(9))
            if rng.random() < eps:                      # explore
                a = rng.choice(moves)
            else:                                       # exploit
                best = max(q[i] for i in moves)
                a = rng.choice([i for i in moves if q[i] == best])
            nb = play(b, a, p)
            if winner(nb):
                target, done = 1.0, True
            elif not legal(nb):
                target, done = 0.0, True
            else:
                qn = Q.setdefault(nb, np.zeros(9))
                target, done = -gamma * max(qn[i] for i in legal(nb)), False
            q[a] += alpha * (target - q[a])             # Q-learning update
            if done:
                break
            b = nb
        if ep % eval_every == 0:
            snapshot(ep)
    return Q, curve


_BOT = {"Q": None, "curve": None}


def get_bot(verbose=True):
    if _BOT["Q"] is None:
        if verbose:
            print("Training the bot by self-play (about 10-30 seconds) ...")
        random.seed(SEED)
        _BOT["Q"], _BOT["curve"] = train_self_play()
    return _BOT["Q"]


def run_reinforcement():
    banner("REINFORCEMENT LEARNING: Q-Learning Bot (Self-Play)")
    print("No dataset. The AGENT plays the ENVIRONMENT (the game) against "
          "itself:")
    print("  reward +1 for a win, 0 for a draw, a loss is learned as -1")
    print("  Policy = pick the move with the highest Q-value\n")

    _BOT["Q"] = None
    Q = get_bot(verbose=False)
    curve = _BOT["curve"]
    random.seed(SEED)

    print(f"  {'Episodes':>9}   win vs random   draw   loss")
    for ep, w, d, l in curve:
        if ep in (0, 3000, 9000, 21000, 42000, 60000):
            print(f"  {ep:>9}   {w * 100:11.1f}%   {d * 100:4.1f}%  {l * 100:4.1f}%")

    bot = lambda b: greedy_move(Q, b)
    vs_rand = evaluate(bot, random_policy, n=1000)
    vs_perfect = evaluate(bot, minimax_policy, n=200)
    print(f"\nFinal test, {len(Q)} positions learned:")
    print(f"  vs random player  (2000 games): win {vs_rand[0] * 100:.1f}%  "
          f"draw {vs_rand[1] * 100:.1f}%  loss {vs_rand[2] * 100:.1f}%")
    print(f"  vs PERFECT player  (400 games): win {vs_perfect[0] * 100:.1f}%  "
          f"draw {vs_perfect[1] * 100:.1f}%  loss {vs_perfect[2] * 100:.1f}%")
    qe = Q[EXAMPLE]
    pick = max(legal(EXAMPLE), key=lambda i: qe[i])
    print(f"\nLive example (O to move, X threatens square 3): the bot's "
          f"Q-values are")
    print("  " + "  ".join(f"sq{i + 1}={qe[i]:+.2f}" for i in legal(EXAMPLE)))
    print(f"  -> it chooses square {pick + 1} "
          f"({'GOOD, it blocks' if pick == 2 else 'MISS'})")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    eps_ = [c[0] for c in curve]
    axes[0].plot(eps_, [c[1] * 100 for c in curve], "o-", color="#50a050", label="win")
    axes[0].plot(eps_, [c[2] * 100 for c in curve], "s-", color="#f0a030", label="draw")
    axes[0].plot(eps_, [c[3] * 100 for c in curve], "^-", color="#d03030", label="loss")
    axes[0].set_xlabel("Training episodes (self-play games)")
    axes[0].set_ylabel("% of games vs a random player")
    axes[0].set_title("Learning curve")
    axes[0].legend()

    xs = np.arange(2)
    width = 0.25
    for k, (lab, col) in enumerate([("win", "#50a050"), ("draw", "#f0a030"),
                                    ("loss", "#d03030")]):
        vals = [vs_rand[k] * 100, vs_perfect[k] * 100]
        axes[1].bar(xs + (k - 1) * width, vals, width, color=col, label=lab)
        for x, v in zip(xs, vals):
            axes[1].text(x + (k - 1) * width, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    axes[1].set_xticks(xs, ["vs random", "vs perfect player"])
    axes[1].set_ylabel("% of games")
    axes[1].set_title("Final strength of the bot")
    axes[1].set_ylim(0, 125)
    axes[1].legend(loc="upper center", ncol=3)

    draw_q_values(axes[2], EXAMPLE, qe,
                  "Learned Q-values when O must block\n(highest value = bot's choice, green)")

    fig.suptitle("Reinforcement Learning - Q-Learning Self-Play", fontweight="bold", y=1.04)
    finish_plot(fig, "3_reinforcement_qlearning.png")


def play_vs_bot():
    banner("PLAY AGAINST THE BOT")
    Q = get_bot()
    ans = input("Do you want to go first as X? (y/n): ").strip().lower()
    human = -1 if ans == "n" else 1
    print(f"You are {SYMBOL[human]}. Enter a square number 1-9.")
    b = EMPTY
    while True:
        p = to_move(b)
        print("\n" + text_board(b))
        if p == human:
            while True:
                s = input("Your move (1-9): ").strip()
                if s.isdigit() and 1 <= int(s) <= 9 and b[int(s) - 1] == 0:
                    m = int(s) - 1
                    break
                print("  Invalid. Pick an empty square from 1 to 9.")
        else:
            m = greedy_move(Q, b)
            print(f"Bot plays square {m + 1}")
        b = play(b, m, p)
        w = winner(b)
        if w or not legal(b):
            print("\n" + text_board(b))
            print("\n" + ("You win!" if w == human else "The bot wins!" if w else "It's a draw."))
            return


# --------------------------------------------------------------------------
# Menu / entry point
# --------------------------------------------------------------------------
def run_all():
    run_supervised()
    run_unsupervised()
    run_reinforcement()
    banner("DONE - all three paradigms executed")


def menu():
    while True:
        print("\n" + "=" * 66)
        print("  TIC-TAC-TOE BOT - THREE MACHINE LEARNING PARADIGMS")
        print("=" * 66)
        print("  1  Supervised    (SVM / Random Forest / Tree / KNN)")
        print("  2  Unsupervised  (K-Means + PCA)")
        print("  3  Reinforcement (Q-Learning self-play)")
        print("  4  Play against the bot")
        print("  5  Run all three paradigms")
        print("  0  Exit")
        try:
            choice = input("Choose: ").strip()
        except EOFError:
            return
        if choice == "1":
            run_supervised()
        elif choice == "2":
            run_unsupervised()
        elif choice == "3":
            run_reinforcement()
        elif choice == "4":
            try:
                play_vs_bot()
            except (EOFError, KeyboardInterrupt):
                print("\nGame cancelled.")
        elif choice == "5":
            run_all()
        elif choice == "0":
            print("Goodbye!")
            return
        else:
            print("Please enter 0-5.")


def main():
    global SAVE_ONLY
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe three-paradigm ML demo")
    parser.add_argument("--all", action="store_true", help="run everything, no menu")
    parser.add_argument("--save", action="store_true",
                        help="don't open plot windows, only save PNGs")
    args = parser.parse_args()
    SAVE_ONLY = args.save
    random.seed(SEED)
    if args.all:
        run_all()
    else:
        menu()


if __name__ == "__main__":
    main()
