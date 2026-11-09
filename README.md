# superliga-prognose

Match forecasts for the Danish Superliga, measured honestly against the betting market.

The question this project tries to answer: **how close can a transparent statistical model get to the closing odds, and where does it go wrong?** The market is used as the benchmark, not as something to beat for profit. This project gives no betting advice.

## What it does today

- Downloads every Superliga result since 2012/13 with closing odds from [football-data.co.uk](https://www.football-data.co.uk/denmark.php) and caches the CSV locally.
- Fits a Poisson goal model with an attack and a defence rating per team plus a home advantage. Older matches count less, so ratings follow form.
- Forecasts any match as expected goals and home/draw/away probabilities.
- Runs a walk-forward backtest. Each forecast uses only matches played before kick-off, and the results are scored against the closing odds (margin removed) with Brier score and log loss.

No third-party packages are needed. It uses only the Python standard library (3.11+).

## Usage

```bash
pip install -e .                 # or: export PYTHONPATH=src

superliga ratings                # current attack/defence ratings
superliga predict Brondby "FC Copenhagen"
superliga backtest --from-season 2022/2023
superliga --offline backtest     # use the cached CSV only
```

Run the tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## First result (9 Nov 2026)

Walk-forward from 2022/23, 836 matches:

|        | Brier  | log loss |
|--------|--------|----------|
| model  | 0.6211 | 1.0347   |
| market | 0.6073 | 1.0143   |

The plain Poisson model sits a little behind the market, as expected. The roadmap is about closing that gap and understanding where it comes from.

## Roadmap

1. **Model**
   - Dixon-Coles correction for low scores (0-0, 1-0, 0-1, 1-1). A plain Poisson model under-predicts draws.
   - Tune the time decay and training window on held-out seasons instead of guessing them.
   - Handle promoted teams with a sensible prior instead of a cold start.
   - Shrink ratings early in the season.
2. **Evaluation**
   - Reliability (calibration) tables per outcome.
   - Results per season and per team.
   - Ranked probability score.
   - Bootstrap intervals so small differences are not over-read.
3. **Data**
   - Validate the CSV schema and fail loudly on changes.
   - Map team names across sources.
   - Add the Danish 1st Division to give promoted teams a history.
4. **Output**
   - A small static web page with the next round's forecasts and the running scorecard, rebuilt by a scheduled job.
5. **Engineering**
   - CI that runs the tests on every push.
   - Type checking.
   - Reproducible backtest reports committed per model version.

## Data and licence

Match data comes from football-data.co.uk. It is downloaded at run time and not included in this repository.

The code is released under the MIT licence (see `LICENSE`).
