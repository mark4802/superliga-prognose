"""Command line: superliga predict | ratings | backtest."""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from .data import load
from .evaluate import backtest
from .model import PoissonModel


def _fit_current(matches, training_days: int) -> PoissonModel:
    today = date.today() + timedelta(days=1)
    recent = [m for m in matches if m.played >= today - timedelta(days=training_days)]
    return PoissonModel().fit(recent, as_of=today)


def cmd_predict(args) -> None:
    matches = load(refresh=not args.offline)
    model = _fit_current(matches, args.training_days)
    f = model.forecast(args.home, args.away)
    print(f"{f.home} vs {f.away}")
    print(f"  expected goals  {f.home_xg:.2f} - {f.away_xg:.2f}")
    print(f"  home win {f.home_win:6.1%}   draw {f.draw:6.1%}   away win {f.away_win:6.1%}")


def cmd_ratings(args) -> None:
    matches = load(refresh=not args.offline)
    model = _fit_current(matches, args.training_days)
    current = {m.home for m in matches if m.season == matches[-1].season}
    print(f"home advantage x{model.home_advantage:.2f}")
    print(f"{'team':<16}{'attack':>8}{'defence':>9}")
    for team in sorted(current, key=lambda t: model.attack[t] / model.defence[t], reverse=True):
        print(f"{team:<16}{model.attack[team]:>8.2f}{model.defence[team]:>9.2f}")


def cmd_backtest(args) -> None:
    matches = load(refresh=not args.offline)
    card = backtest(matches, first_season=args.from_season, training_days=args.training_days)
    a = card.averages()
    print(f"walk-forward from {args.from_season}: {a['matches']} matches")
    print(f"{'':<8}{'Brier':>8}{'log loss':>10}")
    print(f"{'model':<8}{a['model_brier']:>8.4f}{a['model_log_loss']:>10.4f}")
    print(f"{'market':<8}{a['market_brier']:>8.4f}{a['market_log_loss']:>10.4f}")
    print("(lower is better; market = closing odds with the margin removed)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="superliga", description=__doc__)
    parser.add_argument("--offline", action="store_true", help="use the cached CSV only")
    parser.add_argument("--training-days", type=int, default=3 * 365)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("predict", help="forecast one match")
    p.add_argument("home")
    p.add_argument("away")
    p.set_defaults(func=cmd_predict)

    r = sub.add_parser("ratings", help="current attack/defence ratings")
    r.set_defaults(func=cmd_ratings)

    b = sub.add_parser("backtest", help="compare the model with the closing odds")
    b.add_argument("--from-season", default="2022/2023")
    b.set_defaults(func=cmd_backtest)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
