"""Charts drawn only from the Cornell LRMC report tables. Not demo output."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def tournament_accuracy():
    # Table 2 in the Cornell report
    years = ["1999", "2000", "2002", "2003"]
    correct = [38, 41, 39, 33]
    incorrect = [22, 19, 21, 27]
    x = np.arange(len(years))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - 0.18, correct, 0.36, color="#2c5f8a", label="Correct")
    ax.bar(x + 0.18, incorrect, 0.36, color="#c0392b", label="Incorrect")
    ax.set_xticks(x, years)
    ax.set_ylabel("Tournament games")
    ax.set_title("Original study: same-year tournament predictions")
    ax.legend()
    _style(ax)
    fig.text(
        0.01,
        0.01,
        "Source: Cornell LRMC report, Table 2. Overall accuracy about 63%.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "original_tournament_accuracy.png", dpi=140)
    plt.close(fig)


def next_year_accuracy():
    # Table 3 in the Cornell report
    labels = ["1999 to 2000", "2000 to 2001", "2002 to 2003", "2003 to 2004"]
    correct = [35, 43, 41, 44]
    pct = [c / 60 * 100 for c in correct]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(labels[::-1], pct[::-1], color="#2c5f8a")
    ax.axvline(68, color="#c0392b", linestyle="--", label="Report overall ~68%")
    ax.set_xlabel("Percent of tournament games predicted correctly")
    ax.set_xlim(0, 100)
    ax.set_title("Original study: next-year tournament predictions")
    ax.legend(loc="lower right")
    _style(ax)
    fig.text(
        0.01,
        0.01,
        "Source: Cornell LRMC report, Table 3 (60 tournament games per year).",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "original_next_year_accuracy.png", dpi=140)
    plt.close(fig)


def final_four_2000():
    # 2000 Final Four spreads from the Cornell report
    matchups = [
        "Florida vs\nNorth Carolina",
        "Michigan St vs\nWisconsin",
        "Michigan St vs\nFlorida (final)",
    ]
    pred = [20, 14, 6]
    actual = [12, 12, 13]
    x = np.arange(len(matchups))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - 0.18, pred, 0.36, color="#2c5f8a", label="LRMC predicted spread")
    ax.bar(x + 0.18, actual, 0.36, color="#7f8c8d", label="Actual spread")
    ax.set_xticks(x, matchups)
    ax.set_ylabel("Point spread (favorite)")
    ax.set_title("Original study: 2000 Final Four point spreads")
    ax.legend()
    _style(ax)
    fig.text(
        0.01,
        0.01,
        "Source: Cornell LRMC report, 2000 Final Four case. Spreads in points.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "original_final_four_2000.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    tournament_accuracy()
    next_year_accuracy()
    final_four_2000()
    print("wrote original-study figures to", OUT)
