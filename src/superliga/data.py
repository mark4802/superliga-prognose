"""Load Superliga results and closing odds from football-data.co.uk."""

from __future__ import annotations

import csv
import io
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

SOURCE_URL = "https://www.football-data.co.uk/new/DNK.csv"
DEFAULT_CACHE = Path("data") / "DNK.csv"
USER_AGENT = "superliga-prognose/0.1 (personal forecasting project)"


@dataclass(frozen=True)
class Odds:
    home: float
    draw: float
    away: float


@dataclass(frozen=True)
class Match:
    season: str
    played: date
    home: str
    away: str
    home_goals: int
    away_goals: int
    odds: Odds | None

    @property
    def result(self) -> str:
        if self.home_goals > self.away_goals:
            return "H"
        if self.home_goals < self.away_goals:
            return "A"
        return "D"


def download(cache: Path = DEFAULT_CACHE, max_age_hours: float = 12) -> Path:
    """Fetch the CSV unless a fresh enough copy is already cached."""
    if cache.exists() and time.time() - cache.stat().st_mtime < max_age_hours * 3600:
        return cache
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(body)
    return cache


def _float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 1.0 else None


def _odds(row: dict[str, str]) -> Odds | None:
    """Prefer the market average of closing odds, then Pinnacle closing odds."""
    for prefix in ("AvgC", "PSC"):
        prices = [_float(row.get(prefix + side, "")) for side in "HDA"]
        if all(prices):
            return Odds(*prices)
    return None


def parse(text: str) -> list[Match]:
    """Turn the raw CSV into matches sorted by date. Rows without a score are skipped."""
    matches = []
    for row in csv.DictReader(io.StringIO(text.lstrip("﻿"))):
        if not row.get("HG") or not row.get("AG"):
            continue
        matches.append(
            Match(
                season=row["Season"].strip(),
                played=datetime.strptime(row["Date"].strip(), "%d/%m/%Y").date(),
                home=row["Home"].strip(),
                away=row["Away"].strip(),
                home_goals=int(row["HG"]),
                away_goals=int(row["AG"]),
                odds=_odds(row),
            )
        )
    matches.sort(key=lambda m: m.played)
    return matches


def load(cache: Path = DEFAULT_CACHE, refresh: bool = True) -> list[Match]:
    path = download(cache) if refresh or not cache.exists() else cache
    return parse(path.read_text(encoding="utf-8-sig"))
