"""Feature construction for LRMC logistic regression."""

from __future__ import annotations

import pandas as pd


def home_and_home_pairs(games: pd.DataFrame) -> pd.DataFrame:
    """Build home-and-home observations used to estimate s^H_x.

    For teams A and B that play once at each home:
      x = A's home margin vs B
      y = 1 if A also wins the return game at B's home, else 0

    Neutral-site games are excluded.
    """
    g = games.loc[~games["neutral"]].copy()
    g["pair"] = g.apply(
        lambda r: tuple(sorted((r["home_team"], r["away_team"]))), axis=1
    )

    rows = []
    for _, pair_games in g.groupby("pair"):
        if len(pair_games) < 2:
            continue
        # Prefer exact one home each; if more games, take first home for each side.
        homes = {}
        for _, r in pair_games.iterrows():
            homes.setdefault(r["home_team"], r)
        if len(homes) < 2:
            continue

        teams = list(homes.keys())
        for a in teams:
            b = [t for t in teams if t != a][0]
            home_game = homes[a]
            if home_game["away_team"] != b:
                continue
            road_game = homes.get(b)
            if road_game is None or road_game["away_team"] != a:
                continue
            # A wins on the road if A (away) outscores B (home) in return game.
            a_won_road = int(road_game["away_score"] > road_game["home_score"])
            rows.append(
                {
                    "team_a": a,
                    "team_b": b,
                    "home_margin_x": int(home_game["home_margin"]),
                    "won_road": a_won_road,
                    "season": int(home_game["season"]),
                }
            )

    return pd.DataFrame(rows)


def margin_histogram_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Simple home-margin table for plotting."""
    g = games.loc[~games["neutral"], ["home_margin"]].copy()
    return g
