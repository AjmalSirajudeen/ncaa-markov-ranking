# Data

## Bring your own games (simple schema)

Use a CSV with these columns. A blank template is in `sample/games_template.csv`.

| Column | Required | Description |
| --- | --- | --- |
| `season` | yes | Integer season year |
| `home_team` | yes | Home team name or ID |
| `away_team` | yes | Away team name or ID |
| `home_score` | yes | Home points |
| `away_score` | yes | Away points |
| `neutral` | no | Neutral site flag (`true`/`false`) |

```bash
python scripts/run_pipeline.py --path /path/to/your_games.csv
python scripts/run_pipeline.py --path /path/to/your_games.csv --season 2024 --out-dir outputs/my_run
```

## Kaggle compact format

Place in `raw/` (gitignored) or pass with `--path` / `--teams`:

- `MRegularSeasonCompactResults.csv`
- `MTeams.csv` (optional, for readable team names)
- `MNCAATourneyCompactResults.csv` (optional)

```bash
python scripts/run_pipeline.py --season 2003
```

## Bundled demo

`sample/demo_season.csv` is synthetic and only for smoke-testing the pipeline.
