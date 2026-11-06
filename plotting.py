"""Console and matplotlib helpers shared by the three paradigms."""

import os
import sys

import matplotlib

import config

if "--save" in sys.argv:           # choose the backend before pyplot loads
    matplotlib.use("Agg")
    config.set_save_only(True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from game import SYMBOL, legal


def banner(title):
    line = "=" * 66
    print(f"\n{line}\n  {title}\n{line}")


def finish_plot(fig, filename):
    os.makedirs(config.OUT_DIR, exist_ok=True)
    path = os.path.join(config.OUT_DIR, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  [plot saved -> {path}]")
    if not config.SAVE_ONLY:
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
