# Tic-Tac-Toe Bot: Three Machine Learning Paradigms (Final Lab Project)

One program (`ml_demo.py`) that uses one game, Tic-Tac-Toe, to demonstrate Supervised, Unsupervised, and Reinforcement Learning from Module 9. The last menu option lets the audience play against the trained bot.

## Setup and run

```
pip install -r requirements.txt
python ml_demo.py              # interactive menu (use this for the live demo)
python ml_demo.py --all        # run all three paradigms in order
python ml_demo.py --all --save # no pop-up windows, only save PNGs to ./outputs
```

The whole run takes about 10 seconds. Menu option 4 trains the bot first (about 10 to 30 seconds), so run option 3 or 4 once before presenting.

## What each paradigm does

| Paradigm | Algorithm(s) | Task | Result |
|---|---|---|---|
| Supervised (classification) | **SVM (RBF kernel)**, compared with Random Forest, Decision Tree, KNN | Given a board, predict the best move. Labels come from a perfect minimax player. | SVM picks a perfect-play move on 95.2% of 1,130 unseen positions |
| Unsupervised (clustering and dimensionality reduction) | **K-Means** + **PCA** | Group 5,478 board positions with no labels | Finds position types that line up with game progress and centre control; PCA keeps 58.5% of variance in 2D |
| Reinforcement | **Q-Learning** (self-play) | Bot teaches itself Tic-Tac-Toe by playing itself 60,000 times | Never loses: 0% losses against a random player and 100% draws against a perfect player |

## Why these algorithms (for the presentation)

- **SVM**: finds the widest-margin boundary between classes, and the RBF kernel handles non-linear patterns. Here it predicts one of 9 squares. We compare it against three other classifiers to justify the choice.
- **Feature engineering**: we added "I can win here" and "I must block here" flags to the 27 basic board inputs. This raised SVM from 90.5% to 95.2% optimal-move rate.
- **K-Means + PCA**: K-Means groups positions using 8 board features (pieces, centre, corners, edges for X and O). It is never told the move number, winner or best move. PCA squashes 8 features to 2 so the groups can be plotted.
- **Q-Learning**: simplest RL algorithm that shows the full agent, environment, action, reward and policy loop. Both sides share one Q-table, so the bot improves by playing itself. A loss is learned as -1 because the opponent's win is backed up with a flipped sign (the negamax form of the update).

## How it maps to the code

- `ml_demo.py`: the connector you run for the demo - menu, `run_all()` and `main()`.
- `game.py`: shared rules and the perfect player (`legal()`, `optimal_moves()` via `minimax_value()`, `all_positions()`).
- `supervised.py`: `encode()` (45 inputs), `optimal_moves()` labels, `run_supervised()`.
- `unsupervised.py`: `strategic()` (8 features), `run_unsupervised()`.
- `reinforcement.py`: `train_self_play()` (the Q-learning update), `greedy_move()` (the learned policy), `evaluate()`, `play_vs_bot()`.
- `plotting.py` / `config.py`: shared plot helpers, the `--save` switch, the random seed and the output folder.

Each paradigm can also be run on its own: `python supervised.py`, `python unsupervised.py`, `python reinforcement.py`.

## Notes for the demo

- "Optimal move" accuracy counts a prediction as correct when it is any move a perfect player could choose, since several squares are often equally good. Exact-label accuracy (matching one specific tie-broken move) is also printed: 91.7% for SVM.
- Against a random player the bot wins about 88% and draws the rest. It is not losing; a random opponent sometimes blocks by luck.
- The unsupervised silhouette scores are modest (about 0.24) because board positions are discrete. Say so if asked, and point out that the clusters are still interpretable.
- The bot cannot be beaten by a human playing correctly; the best you can do is draw.
