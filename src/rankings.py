"""Ranking tables, matchup evaluation, and figure helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import home_and_home_pairs, margin_histogram_frame
from .logistic_model import (
    MatchupLogistic,
    expected_margin,
    fit_margin_logistic,
    fit_matchup_logistic,
)
from .markov_model import team_stationary


def build_rankings(games: pd.DataFrame, home_court_points: float = 2.0) -> dict:
    pairs = home_and_home_pairs(games)
    margin_model = fit_margin_logistic(pairs, home_court_points=home_court_points)
    pi = team_stationary(games, margin_model)
    matchup_model = fit_matchup_logistic(games, pi)

    ranking = (
        pi.rename("pi")
        .reset_index()
        .rename(columns={"index": "team"})
        .assign(rank=lambda d: np.arange(1, len(d) + 1))
    )
    return {
        "pairs": pairs,
        "margin_model": margin_model,
        "matchup_model": matchup_model,
        "pi": pi,
        "ranking": ranking,
    }


def evaluate_games(
    games: pd.DataFrame,
    pi: pd.Series,
    matchup_model: Optional[MatchupLogistic] = None,
    neutral: bool = True,
) -> pd.DataFrame:
    """Score each game: predict favorite by higher pi (and optional p_ij)."""
    rows = []
    for _, g in games.iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in pi.index or a not in pi.index:
            continue
        diff = float(pi[h] - pi[a])
        pred_home = diff > 0
        actual_home = bool(g["home_win"])
        p_home = (
            float(matchup_model.predict_proba(diff, neutral=neutral))
            if matchup_model is not None
            else None
        )
        rows.append(
            {
                "season": g["season"],
                "home_team": h,
                "away_team": a,
                "home_margin": g["home_margin"],
                "pi_home": float(pi[h]),
                "pi_away": float(pi[a]),
                "pi_diff": diff,
                "pred_home_win": pred_home,
                "actual_home_win": actual_home,
                "correct": pred_home == actual_home,
                "p_home_win": p_home,
                "pred_margin_home": expected_margin(float(pi[h]), float(pi[a])),
            }
        )
    return pd.DataFrame(rows)


def accuracy_summary(eval_df: pd.DataFrame) -> dict:
    if eval_df.empty:
        return {"n": 0, "correct": 0, "accuracy": float("nan")}
    correct = int(eval_df["correct"].sum())
    n = len(eval_df)
    return {"n": n, "correct": correct, "accuracy": correct / n}


def save_figures(games: pd.DataFrame, result: dict, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1) Home margin histogram
    fig, ax = plt.subplots(figsize=(8, 4.5))
    margins = margin_histogram_frame(games)["home_margin"]
    bins = range(int(margins.min()) - 1, int(margins.max()) + 2)
    ax.hist(margins, bins=bins, color="#2c5f8a", edgecolor="white")
    ax.set_title("Home score margin distribution")
    ax.set_xlabel("Home points − away points")
    ax.set_ylabel("Games")
    fig.tight_layout()
    p1 = out_dir / "home_margin_hist.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    paths.append(p1)

    # 2) Logistic fit for s^H_x
    pairs = result["pairs"]
    model = result["margin_model"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not pairs.empty:
        grouped = pairs.groupby("home_margin_x")["won_road"].mean()
        ax.scatter(grouped.index, grouped.values, color="#2c5f8a", label="Observed s^H_x")
    xs = np.linspace(-30, 40, 200)
    ax.plot(xs, model.s_home(xs), color="#c0392b", label=f"Logistic fit (a={model.coef:.4f}, b={model.intercept:.4f})")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Road win probability vs home margin")
    ax.set_xlabel("Home margin x")
    ax.set_ylabel("P(win on road)")
    ax.legend()
    fig.tight_layout()
    p2 = out_dir / "logistic_margin_fit.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    paths.append(p2)

    # 3) Top rankings
    top = result["ranking"].head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["team"][::-1], top["pi"][::-1], color="#2c5f8a")
    ax.set_title("Top teams by steady-state probability")
    ax.set_xlabel("π")
    fig.tight_layout()
    p3 = out_dir / "top_rankings.png"
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    paths.append(p3)

    # 4) Win rate vs pi differential
    eval_df = evaluate_games(games, result["pi"], result["matchup_model"], neutral=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not eval_df.empty:
        tmp = eval_df.copy()
        tmp["bucket"] = pd.cut(tmp["pi_diff"], bins=12)
        summary = tmp.groupby("bucket", observed=False)["actual_home_win"].mean()
        centers = [iv.mid for iv in summary.index]
        ax.scatter(centers, summary.values, color="#2c5f8a", label="Observed home win rate")
        xs = np.linspace(tmp["pi_diff"].min(), tmp["pi_diff"].max(), 200)
        ax.plot(
            xs,
            result["matchup_model"].predict_proba(xs, neutral=False),
            color="#c0392b",
            label="Matchup logistic",
        )
    ax.set_title("Home win rate vs steady-state differential")
    ax.set_xlabel("π_home − π_away")
    ax.set_ylabel("P(home win)")
    ax.legend()
    fig.tight_layout()
    p4 = out_dir / "winrate_vs_pi_diff.png"
    fig.savefig(p4, dpi=140)
    plt.close(fig)
    paths.append(p4)

    return paths
