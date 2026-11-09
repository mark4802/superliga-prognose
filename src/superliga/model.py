"""Poisson goal model with team attack and defence ratings.

Expected goals:
    home = attack[home] * defence[away] * home_advantage
    away = attack[away] * defence[home]

Ratings are fitted by weighted maximum likelihood with Maher's iterative
updates. Older matches count less (exponential time decay).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from .data import Match

MAX_GOALS = 10


@dataclass(frozen=True)
class Forecast:
    home: str
    away: str
    home_xg: float
    away_xg: float
    home_win: float
    draw: float
    away_win: float

    def probabilities(self) -> tuple[float, float, float]:
        return (self.home_win, self.draw, self.away_win)


class PoissonModel:
    def __init__(self, decay_per_day: float = 0.0019, iterations: int = 100, tolerance: float = 1e-8):
        self.decay_per_day = decay_per_day
        self.iterations = iterations
        self.tolerance = tolerance
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.home_advantage = 1.0
        self.matches_seen: dict[str, float] = {}

    def fit(self, matches: list[Match], as_of: date) -> "PoissonModel":
        rows = [(m, math.exp(-self.decay_per_day * (as_of - m.played).days)) for m in matches if m.played < as_of]
        if not rows:
            raise ValueError("no matches before the fitting date")
        teams = sorted({m.home for m, _ in rows} | {m.away for m, _ in rows})
        attack = {t: 1.0 for t in teams}
        defence = {t: 1.0 for t in teams}
        home_adv = 1.2
        seen = {t: 0.0 for t in teams}
        for m, w in rows:
            seen[m.home] += w
            seen[m.away] += w

        for _ in range(self.iterations):
            previous = dict(attack)

            scored = {t: 0.0 for t in teams}
            exposure = {t: 0.0 for t in teams}
            for m, w in rows:
                scored[m.home] += w * m.home_goals
                scored[m.away] += w * m.away_goals
                exposure[m.home] += w * defence[m.away] * home_adv
                exposure[m.away] += w * defence[m.home]
            attack = {t: scored[t] / exposure[t] if exposure[t] else 1.0 for t in teams}
            mean_attack = sum(attack.values()) / len(attack)
            attack = {t: a / mean_attack for t, a in attack.items()}

            conceded = {t: 0.0 for t in teams}
            exposure = {t: 0.0 for t in teams}
            for m, w in rows:
                conceded[m.home] += w * m.away_goals
                conceded[m.away] += w * m.home_goals
                exposure[m.home] += w * attack[m.away]
                exposure[m.away] += w * attack[m.home] * home_adv
            defence = {t: conceded[t] / exposure[t] if exposure[t] else 1.0 for t in teams}

            home_goals = sum(w * m.home_goals for m, w in rows)
            home_expected = sum(w * attack[m.home] * defence[m.away] for m, w in rows)
            home_adv = home_goals / home_expected

            if max(abs(attack[t] - previous[t]) for t in teams) < self.tolerance:
                break

        self.attack, self.defence, self.home_advantage, self.matches_seen = attack, defence, home_adv, seen
        return self

    def knows(self, team: str, minimum_weight: float = 3.0) -> bool:
        return self.matches_seen.get(team, 0.0) >= minimum_weight

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        for team in (home, away):
            if team not in self.attack:
                raise KeyError(f"unknown team: {team}")
        home_xg = self.attack[home] * self.defence[away] * self.home_advantage
        away_xg = self.attack[away] * self.defence[home]
        return home_xg, away_xg

    def forecast(self, home: str, away: str) -> Forecast:
        home_xg, away_xg = self.expected_goals(home, away)
        home_pmf = poisson_pmf(home_xg)
        away_pmf = poisson_pmf(away_xg)
        home_win = draw = away_win = 0.0
        for i, p_home in enumerate(home_pmf):
            for j, p_away in enumerate(away_pmf):
                p = p_home * p_away
                if i > j:
                    home_win += p
                elif i == j:
                    draw += p
                else:
                    away_win += p
        total = home_win + draw + away_win
        return Forecast(home, away, home_xg, away_xg, home_win / total, draw / total, away_win / total)


def poisson_pmf(mean: float, max_goals: int = MAX_GOALS) -> list[float]:
    return [math.exp(-mean) * mean**k / math.factorial(k) for k in range(max_goals + 1)]
