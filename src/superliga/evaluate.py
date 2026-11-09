"""Walk-forward evaluation: the model only ever sees matches played before the one it forecasts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta

from .data import Match, Odds
from .model import PoissonModel

OUTCOMES = ("H", "D", "A")


def market_probabilities(odds: Odds) -> tuple[float, float, float]:
    """Implied probabilities with the bookmaker margin removed proportionally."""
    raw = (1 / odds.home, 1 / odds.draw, 1 / odds.away)
    total = sum(raw)
    return tuple(p / total for p in raw)


def brier(probabilities: tuple[float, float, float], result: str) -> float:
    return sum((p - (1.0 if outcome == result else 0.0)) ** 2 for p, outcome in zip(probabilities, OUTCOMES))


def log_loss(probabilities: tuple[float, float, float], result: str) -> float:
    p = probabilities[OUTCOMES.index(result)]
    return -math.log(max(p, 1e-12))


@dataclass
class Scorecard:
    matches: int = 0
    model_brier: float = 0.0
    market_brier: float = 0.0
    model_log_loss: float = 0.0
    market_log_loss: float = 0.0
    rows: list[dict] = field(default_factory=list)

    def add(self, match: Match, model_p: tuple[float, float, float], market_p: tuple[float, float, float]) -> None:
        self.matches += 1
        self.model_brier += brier(model_p, match.result)
        self.market_brier += brier(market_p, match.result)
        self.model_log_loss += log_loss(model_p, match.result)
        self.market_log_loss += log_loss(market_p, match.result)
        self.rows.append({"match": match, "model": model_p, "market": market_p})

    def averages(self) -> dict[str, float]:
        n = self.matches or 1
        return {
            "matches": self.matches,
            "model_brier": self.model_brier / n,
            "market_brier": self.market_brier / n,
            "model_log_loss": self.model_log_loss / n,
            "market_log_loss": self.market_log_loss / n,
        }


def backtest(
    matches: list[Match],
    first_season: str,
    training_days: int = 3 * 365,
    refit_every_days: int = 7,
    decay_per_day: float = 0.0019,
) -> Scorecard:
    card = Scorecard()
    model: PoissonModel | None = None
    last_fit = None
    for match in matches:
        if match.season < first_season or match.odds is None:
            continue
        if last_fit is None or (match.played - last_fit).days >= refit_every_days:
            window_start = match.played - timedelta(days=training_days)
            history = [m for m in matches if window_start <= m.played < match.played]
            if not history:
                continue
            model = PoissonModel(decay_per_day=decay_per_day).fit(history, as_of=match.played)
            last_fit = match.played
        if model is None or not (model.knows(match.home) and model.knows(match.away)):
            continue
        forecast = model.forecast(match.home, match.away)
        card.add(match, forecast.probabilities(), market_probabilities(match.odds))
    return card
