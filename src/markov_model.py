"""Markov transition matrix and steady-state ranking probabilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .logistic_model import MarginLogistic


def build_transition_matrix(
    games: pd.DataFrame,
    margin_model: MarginLogistic,
) -> Tuple[np.ndarray, List[str]]:
    """Build row-stochastic T where T[i, j] = P(move from team i to team j).

    Construction (Kvam/Sokol-style margin LRMC):
    For each true home/away game with home margin x:
      r = P(home is better | x)
      From home: weight r to stay, (1-r) to away
      From away: weight (1-r) to stay, r to home
    Each team's outgoing weights are averaged over its games, then row-normalized.
    Neutral games use r at margin x with the same home-labeled side.
    """
    teams = sorted(set(games["home_team"]).union(set(games["away_team"])))
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    weights = np.zeros((n, n), dtype=float)
    game_counts = np.zeros(n, dtype=float)

    for _, g in games.iterrows():
        h, a = g["home_team"], g["away_team"]
        i, j = index[h], index[a]
        x = float(g["home_margin"])
        r = float(margin_model.r_home(x))
        r = min(max(r, 1e-6), 1.0 - 1e-6)

        # Home team's transition given this game
        weights[i, i] += r
        weights[i, j] += 1.0 - r
        game_counts[i] += 1.0

        # Away team's transition given this game
        weights[j, j] += 1.0 - r
        weights[j, i] += r
        game_counts[j] += 1.0

    # Average by games played, then enforce stochastic rows.
    for i in range(n):
        if game_counts[i] > 0:
            weights[i, :] /= game_counts[i]
        else:
            weights[i, i] = 1.0
        row_sum = weights[i, :].sum()
        if row_sum <= 0:
            weights[i, :] = 0.0
            weights[i, i] = 1.0
        else:
            weights[i, :] /= row_sum

    return weights, teams


def steady_state(transition: np.ndarray) -> np.ndarray:
    """Solve pi T = pi, sum(pi)=1 via left-eigenvector / linear system."""
    n = transition.shape[0]
    # (T^T - I) pi^T = 0, replace last equation with sum pi = 1
    a = transition.T - np.eye(n)
    a[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.lstsq(a, b, rcond=None)[0]
    pi = np.clip(pi, 0.0, None)
    s = pi.sum()
    if s <= 0:
        return np.ones(n) / n
    return pi / s


def team_stationary(games: pd.DataFrame, margin_model: MarginLogistic) -> pd.Series:
    T, teams = build_transition_matrix(games, margin_model)
    pi = steady_state(T)
    return pd.Series(pi, index=teams, name="pi").sort_values(ascending=False)
