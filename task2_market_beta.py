import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from task2_experiment import TICKERS
from task2_hourly_validation import CONTEST_START, prepare


NEW_YORK = ZoneInfo("America/New_York")


def market_opens(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["chart"]["error"] is None
    chart = payload["chart"]["result"][0]
    assert chart["meta"]["symbol"] == "SPY"
    opens = chart["indicators"]["quote"][0]["open"]
    assert len(chart["timestamp"]) == len(opens)
    return {
        datetime.fromtimestamp(timestamp, NEW_YORK): float(price)
        for timestamp, price in zip(chart["timestamp"], opens)
        if price is not None and np.isfinite(price) and price > 0
    }


def pair_betas(records, betas):
    values = []
    for record in records:
        decision = record["decision"]
        if decision["action"] == "F":
            continue
        assert decision["action"] in ("LS", "SL")
        first = betas[decision["stock1"]]
        second = betas[decision["stock2"]]
        sign = 1 if decision["action"] == "LS" else -1
        values.append({"round_id": record["round_id"],
                       "beta": sign * (first - second) / 2})
    return values


def report(cache, market_path, history_path, live_path):
    schedule, prices, _, _, dates = prepare(cache)
    opens = market_opens(market_path)
    stock_returns = []
    market_returns = []
    matched_dates = set()
    for index in range(1, len(schedule)):
        current = datetime.fromisoformat(schedule[index]["timestamp"])
        previous = datetime.fromisoformat(schedule[index - 1]["timestamp"])
        if current.date() != previous.date() or current not in opens or previous not in opens:
            continue
        assert current.date().isoformat() < CONTEST_START
        hours = (current - previous).total_seconds() / 3600
        assert hours > 0
        stock_returns.append(np.log(prices[index] / prices[index - 1]) / np.sqrt(hours))
        market_returns.append(np.log(opens[current] / opens[previous]) / np.sqrt(hours))
        matched_dates.add(current.date().isoformat())
    stocks = np.asarray(stock_returns)
    market = np.asarray(market_returns)
    assert len(stocks) >= 1000 and len(matched_dates) >= 400
    centered_stocks = stocks - stocks.mean(axis=0)
    centered_market = market - market.mean()
    variance = float(centered_market @ centered_market)
    assert variance > 0
    estimates = centered_stocks.T @ centered_market / variance
    betas = dict(zip(TICKERS, estimates.tolist()))
    history = json.loads(history_path.read_text(encoding="utf-8"))["records"]
    live = json.loads(live_path.read_text(encoding="utf-8"))["records"]
    assert len(history) == 41 and len(live) == 47
    historical_pairs = pair_betas([row for row in history if "net_return" in row], betas)
    live_pairs = pair_betas(live[41:], betas)
    historical_values = np.array([row["beta"] for row in historical_pairs])
    live_values = np.array([row["beta"] for row in live_pairs])
    return {
        "market_proxy": "SPY hourly bar opens, Yahoo Finance chart API",
        "method": "OLS stock beta to SPY on matched within-session log returns divided by sqrt(elapsed hours); pair beta is half the long beta minus half the short beta",
        "sample_first_date": min(matched_dates),
        "sample_last_date": max(matched_dates),
        "sample_dates": len(matched_dates),
        "sample_intervals": len(stocks),
        "spy_source_sha256": hashlib.sha256(market_path.read_bytes()).hexdigest(),
        "stock_betas": betas,
        "historical_pairs": historical_pairs,
        "historical_summary": {
            "active_pairs": len(historical_pairs),
            "mean_signed_beta": float(historical_values.mean()),
            "mean_absolute_beta": float(np.abs(historical_values).mean()),
            "median_absolute_beta": float(np.median(np.abs(historical_values))),
            "maximum_absolute_beta": float(np.abs(historical_values).max()),
        },
        "live_pairs": live_pairs,
        "live_summary": {
            "active_pairs": len(live_pairs),
            "mean_signed_beta": float(live_values.mean()),
            "mean_absolute_beta": float(np.abs(live_values).mean()),
            "maximum_absolute_beta": float(np.abs(live_values).max()),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("research_import/task2_yahoo_1h"))
    parser.add_argument("--market", type=Path, default=Path("research_import/task2_yahoo_1h/SPY.json"))
    parser.add_argument("--history", type=Path, default=Path("outputs/task2_factor_agent.json"))
    parser.add_argument("--live", type=Path, default=Path("outputs/task2_factor_agent_live.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/task2_market_beta.json"))
    args = parser.parse_args()
    result = report(args.cache, args.market, args.history, args.live)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sample_dates": result["sample_dates"],
                      "sample_intervals": result["sample_intervals"],
                      "historical_summary": result["historical_summary"],
                      "live_summary": result["live_summary"]}, indent=2))


if __name__ == "__main__":
    main()
