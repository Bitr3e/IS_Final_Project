"""
Tic-Tac-Toe Bot - Three Machine Learning Paradigms (Final Lab Project)
======================================================================
One game, three ways of learning:

Supervised    : SVM / Random Forest / Decision Tree / KNN learn to pick the
                best move. Labels come from a perfect (minimax) player.
Unsupervised  : K-Means + PCA group ~5,000 board positions WITHOUT labels.
Reinforcement : Q-Learning bot teaches itself the game by self-play, then
                you can play against it.

The paradigms live in supervised.py, unsupervised.py and reinforcement.py;
this file connects them for the live demo.

Run:
    python ml_demo.py                # interactive menu (live demo)
    python ml_demo.py --all          # run the three paradigms in order
    python ml_demo.py --all --save   # no pop-up windows, only save PNGs
"""

import argparse
import random

import config
from config import SEED
from plotting import banner
from reinforcement import play_vs_bot, run_reinforcement
from supervised import run_supervised
from unsupervised import run_unsupervised


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
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe three-paradigm ML demo")
    parser.add_argument("--all", action="store_true", help="run everything, no menu")
    parser.add_argument("--save", action="store_true",
                        help="don't open plot windows, only save PNGs")
    args = parser.parse_args()
    config.set_save_only(args.save)
    random.seed(SEED)
    if args.all:
        run_all()
    else:
        menu()


if __name__ == "__main__":
    main()
