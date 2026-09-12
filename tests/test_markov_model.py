"""Tests for Markov matrix properties and ranking sanity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_games, normalize_games
from src.logistic_model import MarginLogistic, fit_margin_logistic
from src.features import home_and_home_pairs
from src.markov_model import build_transition_matrix, steady_state, team_stationary
from src.rankings import build_rankings, evaluate_games, accuracy_summary


def _tiny_games() -> pd.DataFrame:
    rows = [
        # A vs B home-and-home
        dict(season=2024, home_team="A", away_team="B", home_score=80, away_score=70, neutral=False),
        dict(season=2024, home_team="B", away_team="A", home_score=68, away_score=72, neutral=False),
        # A vs C
        dict(season=2024, home_team="A", away_team="C", home_score=90, away_score=60, neutral=False),
        dict(season=2024, home_team="C", away_team="A", home_score=70, away_score=75, neutral=False),
        # B vs C
        dict(season=2024, home_team="B", away_team="C", home_score=78, away_score=74, neutral=False),
        dict(season=2024, home_team="C", away_team="B", home_score=71, away_score=69, neutral=False),
    ]
    return normalize_games(pd.DataFrame(rows))


def test_transition_rows_stochastic():
    games = _tiny_games()
    pairs = home_and_home_pairs(games)
    model = fit_margin_logistic(pairs)
    T, teams = build_transition_matrix(games, model)
    assert T.shape == (len(teams), len(teams))
    np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(T >= -1e-12)


def test_steady_state_sums_to_one():
    games = _tiny_games()
    model = MarginLogistic(coef=0.05, intercept=0.8, home_court_points=2.0)
    T, _ = build_transition_matrix(games, model)
    pi = steady_state(T)
    assert pi.shape == (T.shape[0],)
    np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-8)
    assert np.all(pi >= -1e-12)


def test_stronger_team_ranks_higher():
    games = _tiny_games()
    result = build_rankings(games)
    # A swept / dominated margins vs B and C in the fixture
    assert result["pi"]["A"] > result["pi"]["B"]
    assert result["pi"]["A"] > result["pi"]["C"]


def test_demo_pipeline_runs():
    games = load_games()
    result = build_rankings(games)
    assert len(result["ranking"]) >= 8
    eval_df = evaluate_games(games, result["pi"])
    summary = accuracy_summary(eval_df)
    assert summary["n"] == len(games)
    assert 0.0 <= summary["accuracy"] <= 1.0
