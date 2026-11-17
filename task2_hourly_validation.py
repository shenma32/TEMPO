import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np

from task2_experiment import METHODS, TICKERS, fixed_method, metrics, pair_return


NEW_YORK = ZoneInfo("America/New_York")
RELEASE_HOURS = (9, 10, 11, 12, 14, 15)
CONTEST_START = "2026-10-14"


def get_chart(ticker: str, cache: Path):
    path = cache / f"{ticker}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=60m&range=730d&includePrePost=false"
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as response:
        payload = json.load(response)
    assert payload["chart"]["error"] is None
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def prepare(cache: Path):
    charts = {}
    for ticker in TICKERS:
        chart = get_chart(ticker, cache)["chart"]["result"][0]
        quote = chart["indicators"]["quote"][0]
        assert len(chart["timestamp"]) == len(quote["open"])
        charts[ticker] = {
            datetime.fromtimestamp(timestamp, NEW_YORK): float(price)
            for timestamp, price in zip(chart["timestamp"], quote["open"])
            if price is not None and np.isfinite(price) and price > 0
        }
    first = charts[TICKERS[0]]
    dates = sorted({
        instant.date() for instant in first
        if instant.hour == 9 and instant.minute == 30 and instant.date().isoformat() < CONTEST_START
    })
    schedule = []
    prices = []
    retained_dates = []
    for day in dates:
        day_times = [datetime(day.year, day.month, day.day, hour, 30, tzinfo=NEW_YORK) for hour in RELEASE_HOURS]
        if any(instant not in charts[ticker] for instant in day_times for ticker in TICKERS):
            continue
        retained_dates.append(day.isoformat())
        for instant in day_times:
            schedule.append({"timestamp": instant.isoformat()})
            prices.append([charts[ticker][instant] for ticker in TICKERS])
    assert len(retained_dates) >= 400
    eligible = [(index, index + 1, index + 2) for index in range(len(schedule) - 2)]
    return schedule, np.asarray(prices), np.zeros_like(prices), eligible, retained_dates


def validate(cache: Path):
    schedule, prices, counts, eligible, dates = prepare(cache)
    names = [name for name in METHODS if name in {"flat", "momentum", "reversal", "related_pair_reversal", "session_switch"}]
    records = {name: [] for name in names}
    for index, entry, exit_index in eligible:
        date = datetime.fromisoformat(schedule[index]["timestamp"]).date().isoformat()
        for name in names:
            pair = fixed_method(name, prices, counts, schedule, index)
            value = pair_return(pair, prices[entry], prices[exit_index])
            records[name].append({"date": date, "return": value, "active": pair is not None})
    development_dates = {date for date in dates if date < "2026-01-01"}
    validation_dates = set(dates) - development_dates
    assert len(development_dates) >= 300 and len(validation_dates) >= 100
    result = {
        "source": "Yahoo Finance 60-minute bars, open price at scheduled release",
        "limitations": "Hourly bar open differs from the competition's one-minute valuation and has no matching news summaries",
        "cached_input_sha256": {ticker: hashlib.sha256((cache / f"{ticker}.json").read_bytes()).hexdigest() for ticker in TICKERS},
        "development_dates": sorted(development_dates),
        "validation_dates": sorted(validation_dates),
        "methods": {},
    }
    for name, rows in records.items():
        result["methods"][name] = {
            "development": metrics([row["return"] for row in rows if row["date"] in development_dates]),
            "validation": metrics([row["return"] for row in rows if row["date"] in validation_dates]),
            "validation_net_5bps_per_leg_side": metrics([
                row["return"] - (0.001 if row["active"] else 0)
                for row in rows if row["date"] in validation_dates
            ]),
            "by_year": {
                year: metrics([row["return"] for row in rows if row["date"].startswith(year)])
                for year in ("2023", "2024", "2025", "2026")
            },
            "by_month": {
                month: metrics([row["return"] for row in rows if row["date"].startswith(month)])
                for month in sorted({row["date"][:7] for row in rows})
            },
            "daily_return": {
                date: metrics([row["return"] for row in rows if row["date"] == date])["cumulative_return"]
                for date in dates
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("research_import/task2_yahoo_1h"))
    parser.add_argument("--output", type=Path, default=Path("outputs/task2_hourly_validation.json"))
    args = parser.parse_args()
    result = validate(args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, values in sorted(result["methods"].items(), key=lambda row: row[1]["validation"]["cumulative_return"], reverse=True):
        print(f"{name:24} development={values['development']['cumulative_return']:+.4%} validation={values['validation']['cumulative_return']:+.4%} validation_net_5bps={values['validation_net_5bps_per_leg_side']['cumulative_return']:+.4%}")
    print("Dates:", len(result["development_dates"]), "development,", len(result["validation_dates"]), "validation")
    print("Report:", args.output)


if __name__ == "__main__":
    main()
