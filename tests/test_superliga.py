import math
import random
import unittest
from datetime import date, timedelta

from superliga.data import Match, Odds, parse
from superliga.evaluate import backtest, brier, log_loss, market_probabilities
from superliga.model import PoissonModel, poisson_pmf

CSV = (
    "﻿Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA\n"
    "Denmark,Superliga ,2025/2026,02/08/2025,17:00, Brondby ,Aarhus,2,1,H,2.0,3.5,3.8,1.95,3.4,3.7\n"
    "Denmark,Superliga,2025/2026,01/08/2025,18:00,Viborg,Odense,0,0,D,2.5,3.3,2.9,,,\n"
    "Denmark,Superliga,2025/2026,03/08/2025,14:00,Vejle,Lyngby,,,,,,,,,\n"
)


def sample(rng, mean):
    """Draw from a Poisson distribution (Knuth)."""
    limit, k, p = math.exp(-mean), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def synthetic_league(seasons=3, seed=1):
    rng = random.Random(seed)
    attack = {"Strong": 1.6, "Middle": 1.0, "Weak": 0.6, "Other": 0.8}
    defence = {"Strong": 0.7, "Middle": 1.0, "Weak": 1.4, "Other": 1.1}
    home_adv = 1.25
    matches, day = [], date(2020, 1, 1)
    for _ in range(seasons * 20):
        for home in attack:
            for away in attack:
                if home == away:
                    continue
                hg = sample(rng, attack[home] * defence[away] * home_adv)
                ag = sample(rng, attack[away] * defence[home])
                matches.append(Match("s", day, home, away, hg, ag, Odds(2.5, 3.3, 3.0)))
        day += timedelta(days=7)
    return matches, attack, home_adv


class ParseTest(unittest.TestCase):
    def test_parse_sorts_strips_and_skips_unplayed(self):
        matches = parse(CSV)
        self.assertEqual([m.home for m in matches], ["Viborg", "Brondby"])
        self.assertEqual(matches[1].result, "H")
        self.assertEqual(matches[0].result, "D")

    def test_odds_prefer_market_average_then_pinnacle(self):
        viborg, brondby = parse(CSV)
        self.assertEqual(brondby.odds, Odds(1.95, 3.4, 3.7))
        self.assertEqual(viborg.odds, Odds(2.5, 3.3, 2.9))


class ModelTest(unittest.TestCase):
    def test_pmf_sums_to_one(self):
        self.assertAlmostEqual(sum(poisson_pmf(1.4)), 1.0, places=6)

    def test_recovers_team_order_and_home_advantage(self):
        matches, attack, home_adv = synthetic_league()
        model = PoissonModel(decay_per_day=0).fit(matches, as_of=matches[-1].played + timedelta(days=1))
        fitted = sorted(model.attack, key=model.attack.get)
        self.assertEqual(fitted, sorted(attack, key=attack.get))
        self.assertAlmostEqual(model.home_advantage, home_adv, delta=0.12)

    def test_forecast_probabilities_sum_to_one(self):
        matches, _, _ = synthetic_league(seasons=1)
        model = PoissonModel().fit(matches, as_of=matches[-1].played + timedelta(days=1))
        forecast = model.forecast("Strong", "Weak")
        self.assertAlmostEqual(sum(forecast.probabilities()), 1.0, places=9)
        self.assertGreater(forecast.home_win, forecast.away_win)

    def test_fit_ignores_matches_on_or_after_as_of(self):
        matches, _, _ = synthetic_league(seasons=1)
        cutoff = matches[len(matches) // 2].played
        a = PoissonModel().fit(matches, as_of=cutoff)
        b = PoissonModel().fit([m for m in matches if m.played < cutoff], as_of=cutoff)
        self.assertEqual(a.attack, b.attack)

    def test_unknown_team_raises(self):
        matches, _, _ = synthetic_league(seasons=1)
        model = PoissonModel().fit(matches, as_of=matches[-1].played + timedelta(days=1))
        with self.assertRaises(KeyError):
            model.forecast("Strong", "Nobody")


class EvaluateTest(unittest.TestCase):
    def test_market_probabilities_remove_margin(self):
        p = market_probabilities(Odds(2.0, 3.5, 4.0))
        self.assertAlmostEqual(sum(p), 1.0, places=12)
        self.assertGreater(p[0], p[2])

    def test_scores(self):
        self.assertEqual(brier((1.0, 0.0, 0.0), "H"), 0.0)
        self.assertAlmostEqual(brier((1 / 3, 1 / 3, 1 / 3), "D"), 2 / 3)
        self.assertAlmostEqual(log_loss((0.5, 0.25, 0.25), "H"), math.log(2))

    def test_backtest_only_uses_the_past(self):
        matches, _, _ = synthetic_league(seasons=2)
        later = [Match("t", m.played + timedelta(days=500), m.home, m.away, m.home_goals, m.away_goals, m.odds) for m in matches[:40]]
        card = backtest(matches + later, first_season="t", training_days=10_000)
        self.assertEqual(card.matches, 40)
        for row in card.rows:
            self.assertAlmostEqual(sum(row["model"]), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
