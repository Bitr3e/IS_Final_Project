"""Shared settings for the Tic-Tac-Toe three-paradigm demo."""

SEED = 42
OUT_DIR = "outputs"
SAVE_ONLY = False


def set_save_only(value):
    """Tell the plotting helpers whether to skip pop-up windows."""
    global SAVE_ONLY
    SAVE_ONLY = bool(value)
