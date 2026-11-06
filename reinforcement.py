"""
Paradigm 3 - REINFORCEMENT LEARNING: a Q-learning bot trained by self-play.

No dataset: the agent plays the environment (the game) against itself.
Reward +1 for a win, 0 for a draw; a loss is learned as -1 because the
opponent's win is backed up with a flipped sign (negamax form of the update).
Run this file directly to train and play in one step.
"""

import random

import numpy as np
import plotting                      # picks the Agg backend when --save is used
import matplotlib.pyplot as plt

import config
from game import (EMPTY, EXAMPLE, SYMBOL, legal, optimal_moves, play,
                  text_board, to_move, winner)
from plotting import banner, draw_q_values, finish_plot

SEED = config.SEED


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
    """Both sides share one Q-table (always from the mover's point of view)."""
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


if __name__ == "__main__":
    random.seed(SEED)
    try:
        ans = input("Train + evaluate the Q-learning bot (t) or play against it (p)? "
                    ).strip().lower()
    except EOFError:
        ans = "t"                       # non-interactive: just run the paradigm
    if ans == "p":
        play_vs_bot()
    else:
        run_reinforcement()
