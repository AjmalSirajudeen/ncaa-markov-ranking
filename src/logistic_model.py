"""Logistic models for LRMC transition probabilities and matchup odds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class MarginLogistic:
    """Fits s^H_x = P(A beats B on the road | A beat B by x at home)."""

    coef: float
    intercept: float
    home_court_points: float = 2.0
    n_obs: int = 0

    def s_home(self, x: np.ndarray | float) -> np.ndarray | float:
        """Road-win probability given home margin x."""
        z = self.coef * np.asarray(x, dtype=float) + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def r_home(self, x: np.ndarray | float) -> np.ndarray | float:
        """P(home team is better | home margin x), with neutral-court offset h."""
        return self.s_home(np.asarray(x, dtype=float) + self.home_court_points)

    def r_away_better_given_home_margin(self, x: np.ndarray | float) -> np.ndarray | float:
        return 1.0 - self.r_home(x)


@dataclass
class MatchupLogistic:
    """P(i beats j) from steady-state gap x = pi_i - pi_j (neutral court)."""

    coef: float
    intercept: float
    n_obs: int = 0

    def predict_proba(self, pi_diff: np.ndarray | float, neutral: bool = True) -> np.ndarray | float:
        x = np.asarray(pi_diff, dtype=float)
        # On a true home court the intercept absorbs home advantage; for NCAA
        # tournament (neutral), use intercept=0 as in the original report.
        intercept = 0.0 if neutral else self.intercept
        z = self.coef * x + intercept
        return 1.0 / (1.0 + np.exp(-z))


def fit_margin_logistic(
    pairs: pd.DataFrame,
    home_court_points: float = 2.0,
) -> MarginLogistic:
    if pairs.empty or pairs["won_road"].nunique() < 2:
        # Fallback close to the original paper's (a, b) if pairs are insufficient.
        return MarginLogistic(
            coef=0.0504, intercept=0.8052, home_court_points=home_court_points, n_obs=len(pairs)
        )

    X = pairs[["home_margin_x"]].to_numpy(dtype=float)
    y = pairs["won_road"].to_numpy(dtype=int)
    model = LogisticRegression(solver="lbfgs")
    model.fit(X, y)
    return MarginLogistic(
        coef=float(model.coef_[0][0]),
        intercept=float(model.intercept_[0]),
        home_court_points=home_court_points,
        n_obs=len(pairs),
    )


def fit_matchup_logistic(
    games: pd.DataFrame,
    pi: pd.Series,
) -> MatchupLogistic:
    """Fit P(home wins) ~ logistic(a * (pi_home - pi_away) + b)."""
    rows = []
    for _, g in games.iterrows():
        if g["home_team"] not in pi.index or g["away_team"] not in pi.index:
            continue
        rows.append(
            {
                "diff": float(pi[g["home_team"]] - pi[g["away_team"]]),
                "home_win": int(g["home_win"]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or frame["home_win"].nunique() < 2:
        return MatchupLogistic(coef=0.0628, intercept=0.2039, n_obs=len(frame))

    X = frame[["diff"]].to_numpy(dtype=float)
    y = frame["home_win"].to_numpy(dtype=int)
    model = LogisticRegression(solver="lbfgs")
    model.fit(X, y)
    return MatchupLogistic(
        coef=float(model.coef_[0][0]),
        intercept=float(model.intercept_[0]),
        n_obs=len(frame),
    )


def expected_margin(pi_i: float, pi_j: float, scale: Optional[float] = None) -> float:
    """Map steady-state gap to an expected point margin.

    The original report printed a large constant (OCR-noisy). We default to
    fitting scale externally; if omitted, use a conservative 400 * 100-scale gap
    heuristic only when pi values are probabilities summing to 1.
    """
    diff = pi_i - pi_j
    if scale is None:
        # Typical pi gaps are O(1e-3); scale ~ few thousand recovers ~10-pt spreads.
        scale = 4000.0
    return scale * diff
