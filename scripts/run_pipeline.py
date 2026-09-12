"""Run the LRMC ranking pipeline and write rankings + figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_games
from src.rankings import accuracy_summary, build_rankings, evaluate_games, save_figures


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "NCAA LRMC ranking pipeline. Pass your own games CSV with --path, "
            "or place Kaggle compact files in data/raw/. With no inputs, runs the "
            "bundled synthetic demo."
        )
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help=(
            "Path to a games CSV. Accepts the simple schema "
            "(season,home_team,away_team,home_score,away_score[,neutral]) "
            "or Kaggle MRegularSeasonCompactResults-style columns."
        ),
    )
    parser.add_argument(
        "--teams",
        type=str,
        default=None,
        help="Optional Kaggle MTeams.csv path (used when --path is Kaggle compact format).",
    )
    parser.add_argument("--season", type=int, default=None, help="Keep only this season")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "outputs"),
        help="Directory for rankings CSV and figures",
    )
    args = parser.parse_args()

    games = load_games(path=args.path, season=args.season, teams_path=args.teams)
    if args.path:
        source = args.path
    elif (ROOT / "data/raw/MRegularSeasonCompactResults.csv").exists():
        source = "data/raw (Kaggle)"
    else:
        source = "data/sample/demo_season.csv"
    print(f"Loaded {len(games)} games from {source}")
    if games["season"].nunique() == 1:
        print(f"Season: {int(games['season'].iloc[0])}")
    else:
        seasons = sorted(games["season"].unique().tolist())
        print(f"Seasons: {seasons[0]}-{seasons[-1]} ({len(seasons)} seasons)")

    result = build_rankings(games)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    ranking_path = out_dir / "rankings.csv"
    result["ranking"].to_csv(ranking_path, index=False)

    eval_df = evaluate_games(games, result["pi"], result["matchup_model"], neutral=False)
    summary = accuracy_summary(eval_df)
    eval_path = out_dir / "regular_season_eval.csv"
    eval_df.to_csv(eval_path, index=False)

    meta = {
        "n_games": int(len(games)),
        "n_teams": int(result["pi"].shape[0]),
        "n_home_and_home_pairs": int(len(result["pairs"])),
        "margin_logistic": {
            "a": result["margin_model"].coef,
            "b": result["margin_model"].intercept,
            "h": result["margin_model"].home_court_points,
            "n_obs": result["margin_model"].n_obs,
        },
        "matchup_logistic": {
            "a": result["matchup_model"].coef,
            "b": result["matchup_model"].intercept,
            "n_obs": result["matchup_model"].n_obs,
        },
        "regular_season_pi_favorite_accuracy": summary,
        "data_source_note": source,
    }
    meta_path = out_dir / "run_summary.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    fig_paths = save_figures(games, result, fig_dir)

    print(f"Teams ranked: {meta['n_teams']}")
    print(
        "Margin logistic: a={a:.4f}, b={b:.4f} (n={n})".format(
            a=meta["margin_logistic"]["a"],
            b=meta["margin_logistic"]["b"],
            n=meta["margin_logistic"]["n_obs"],
        )
    )
    print(
        "Regular-season accuracy (higher π as favorite): "
        f"{summary['correct']}/{summary['n']} = {summary['accuracy']:.1%}"
    )
    print(f"Wrote {ranking_path}")
    print(f"Wrote {eval_path}")
    print(f"Wrote {meta_path}")
    for p in fig_paths:
        print(f"Wrote {p}")
    print("\nTop 10:")
    print(result["ranking"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
