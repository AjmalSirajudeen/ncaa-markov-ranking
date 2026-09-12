"""Load and normalize NCAA game-level results.

Preferred public source for a full historical rebuild:
Kaggle March Machine Learning Mania files
  - MRegularSeasonCompactResults.csv
  - MTeams.csv
  - MNCAATourneyCompactResults.csv (optional, for evaluation)

Place downloaded files in data/raw/. This repo ships a small synthetic demo
season in data/sample/ so the pipeline can run without a Kaggle account.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

REQUIRED_COLS = [
    "season",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_raw_dir() -> Path:
    return _project_root() / "data" / "raw"


def default_sample_path() -> Path:
    return _project_root() / "data" / "sample" / "demo_season.csv"


def normalize_games(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean frame with REQUIRED_COLS plus derived fields."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[REQUIRED_COLS].copy()
    out["season"] = out["season"].astype(int)
    out["home_team"] = out["home_team"].astype(str).str.strip()
    out["away_team"] = out["away_team"].astype(str).str.strip()
    out["home_score"] = out["home_score"].astype(int)
    out["away_score"] = out["away_score"].astype(int)

    bad = out["home_team"] == out["away_team"]
    if bad.any():
        raise ValueError(f"Found {bad.sum()} games where home_team == away_team")

    out["home_margin"] = out["home_score"] - out["away_score"]
    out["home_win"] = (out["home_margin"] > 0).astype(int)
    # Neutral-site flag if present; otherwise assume true home/away.
    if "neutral" in df.columns:
        out["neutral"] = df["neutral"].astype(bool)
    else:
        out["neutral"] = False
    return out.reset_index(drop=True)


def load_normalized_csv(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return normalize_games(pd.read_csv(path))


def load_kaggle_compact(
    results_path: Path | str,
    teams_path: Optional[Path | str] = None,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """Convert Kaggle compact results (WTeamID/LTeamID/WLoc) to home/away form."""
    results_path = Path(results_path)
    raw = pd.read_csv(results_path)
    needed = {"Season", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"}
    if not needed.issubset(raw.columns):
        raise ValueError(
            f"{results_path.name} does not look like Kaggle compact results. "
            f"Expected columns including {sorted(needed)}"
        )

    if season is not None:
        raw = raw[raw["Season"] == season].copy()
        if raw.empty:
            raise ValueError(f"No games found for season={season}")

    id_to_name = None
    if teams_path is not None:
        teams = pd.read_csv(teams_path)
        if {"TeamID", "TeamName"}.issubset(teams.columns):
            id_to_name = dict(zip(teams["TeamID"], teams["TeamName"]))

    def label(team_id: int) -> str:
        if id_to_name is None:
            return str(int(team_id))
        return str(id_to_name.get(int(team_id), team_id))

    rows = []
    for _, g in raw.iterrows():
        loc = str(g["WLoc"]).upper()
        if loc == "H":
            home_id, away_id = g["WTeamID"], g["LTeamID"]
            home_score, away_score = g["WScore"], g["LScore"]
            neutral = False
        elif loc == "A":
            home_id, away_id = g["LTeamID"], g["WTeamID"]
            home_score, away_score = g["LScore"], g["WScore"]
            neutral = False
        else:  # neutral — treat winner as 'home' for margin bookkeeping
            home_id, away_id = g["WTeamID"], g["LTeamID"]
            home_score, away_score = g["WScore"], g["LScore"]
            neutral = True

        rows.append(
            {
                "season": int(g["Season"]),
                "home_team": label(home_id),
                "away_team": label(away_id),
                "home_score": int(home_score),
                "away_score": int(away_score),
                "neutral": neutral,
            }
        )

    return normalize_games(pd.DataFrame(rows))


def load_games(
    path: Optional[Path | str] = None,
    season: Optional[int] = None,
    prefer_sample: bool = True,
    teams_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Load games from an explicit path, data/raw Kaggle files, or the demo sample.

    Supported inputs:
    1. Simple schema CSV: season, home_team, away_team, home_score, away_score [, neutral]
    2. Kaggle compact results: Season, WTeamID, LTeamID, WScore, LScore, WLoc
       (optional MTeams.csv via teams_path or data/raw/MTeams.csv)
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Games file not found: {p}\n"
                "Expected a CSV with columns "
                "season,home_team,away_team,home_score,away_score "
                "or Kaggle compact columns "
                "Season,WTeamID,LTeamID,WScore,LScore,WLoc. "
                "See data/sample/games_template.csv for the simple schema."
            )
        peek = pd.read_csv(p, nrows=2)
        if {"WTeamID", "LTeamID", "WLoc"}.issubset(peek.columns):
            teams = Path(teams_path) if teams_path else default_raw_dir() / "MTeams.csv"
            return load_kaggle_compact(
                p, teams_path=teams if teams.exists() else None, season=season
            )
        games = load_normalized_csv(p)
        if season is not None:
            games = games[games["season"] == season].copy()
            if games.empty:
                raise ValueError(f"No games found for season={season} in {p}")
        return games.reset_index(drop=True)

    raw_dir = default_raw_dir()
    compact = raw_dir / "MRegularSeasonCompactResults.csv"
    teams = Path(teams_path) if teams_path else raw_dir / "MTeams.csv"
    if compact.exists():
        return load_kaggle_compact(
            compact, teams_path=teams if teams.exists() else None, season=season
        )

    if prefer_sample:
        games = load_normalized_csv(default_sample_path())
        if season is not None:
            games = games[games["season"] == season].copy()
        return games.reset_index(drop=True)

    raise FileNotFoundError(
        "No game data found. Pass --path to your CSV, or place "
        "MRegularSeasonCompactResults.csv in data/raw/. "
        "See data/sample/games_template.csv for the expected simple schema."
    )
