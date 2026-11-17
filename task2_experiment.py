import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


TICKERS = ("NVDA", "GOOG", "META", "MSFT", "JPM", "AMZN", "AAPL", "AMD", "WMT", "TSLA")
PAIRS = (("NVDA", "AMD"), ("GOOG", "META"), ("MSFT", "AMZN"), ("AAPL", "MSFT"), ("GOOG", "MSFT"))
METHODS = (
    "flat",
    "momentum",
    "reversal",
    "related_pair_reversal",
    "session_switch",
    "news_gated_reversal",
    "online_ridge",
    "online_ridge_news",
    "adaptive_expert",
    "adaptive_expert_flat",
)


def read_data(root: Path, state_path: Path):
    manifest = json.loads((root / "releases.json").read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    releases = manifest["published_rounds"]
    schedule = state["release_schedule"][:len(releases)]
    assert releases and len(releases) == len(schedule)
    assert [row["round_id"] for row in releases] == [row["round_id"] for row in schedule]
    prices = []
    counts = []
    for release, timing in zip(releases, schedule):
        folder = root / "data" / release["folder_url"].split("/")[-1]
        price_path = folder / "price.parquet"
        news_path = folder / "news.parquet"
        for kind, path in (("price", price_path), ("news", news_path)):
            expected = release.get("file_sha256", {}).get(kind)
            if expected is not None:
                assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        price = pd.read_parquet(price_path).set_index("ticker")
        news = pd.read_parquet(news_path).set_index("ticker")
        assert set(price.index) == set(news.index) == set(TICKERS)
        assert price.loc[list(TICKERS), "status"].eq("available").all()
        assert price.loc[list(TICKERS), "price"].gt(0).all()
        assert news.loc[list(TICKERS), "timestamp"].eq(price.loc[list(TICKERS), "timestamp"]).all()
        assert all(datetime.fromisoformat(value) == datetime.fromisoformat(timing["timestamp"]) for value in price["timestamp"])
        assert all(datetime.fromisoformat(value) == datetime.fromisoformat(timing["window_start"]) for value in news["window_start"])
        assert all(datetime.fromisoformat(value) == datetime.fromisoformat(timing["window_end"]) for value in news["window_end"])
        assert datetime.fromisoformat(release["published_at"]) <= datetime.fromisoformat(timing["action_due"])
        prices.append(price.loc[list(TICKERS), "price"].to_numpy(dtype=float))
        counts.append(news.loc[list(TICKERS), "news_count"].to_numpy(dtype=float))
    times = [datetime.fromisoformat(row["timestamp"]) for row in schedule]
    index_by_time = {instant: index for index, instant in enumerate(times)}
    eligible = []
    for index, row in enumerate(schedule):
        entry = index_by_time.get(datetime.fromisoformat(row["entry_at"]))
        exit_index = index_by_time.get(datetime.fromisoformat(row["exit_at"]))
        if entry is not None and exit_index is not None:
            assert index < entry < exit_index
            eligible.append((index, entry, exit_index))
    return state, schedule, np.array(prices), np.array(counts), eligible


def recent_returns(prices, schedule, index):
    if index == 0:
        return None
    current = datetime.fromisoformat(schedule[index]["timestamp"])
    previous = datetime.fromisoformat(schedule[index - 1]["timestamp"])
    if current.date() != previous.date():
        return None
    return np.log(prices[index] / prices[index - 1])


def features(prices, counts, schedule, index, include_news):
    first = recent_returns(prices, schedule, index)
    if first is None:
        return None
    second = recent_returns(prices, schedule, index - 1)
    if second is None:
        second = np.zeros(len(TICKERS))
    first -= np.median(first)
    second -= np.median(second)
    columns = [first / 0.01, second / 0.01]
    if include_news:
        news = np.log1p(counts[index])
        columns.append(news - np.mean(news))
    return np.column_stack(columns)


def choose_extremes(scores, reverse=False, allowed=None):
    if allowed is None:
        allowed = np.arange(len(TICKERS))
    order = allowed[np.argsort(scores[allowed], kind="stable")]
    short_index, long_index = (order[-1], order[0]) if reverse else (order[0], order[-1])
    return int(long_index), int(short_index)


def fixed_method(name, prices, counts, schedule, index):
    if name == "flat":
        return None
    recent = recent_returns(prices, schedule, index)
    if recent is None:
        return None
    if name == "momentum":
        return choose_extremes(recent)
    if name == "reversal":
        return choose_extremes(recent, reverse=True)
    if name == "related_pair_reversal":
        differences = [(abs(recent[TICKERS.index(a)] - recent[TICKERS.index(b)]), a, b) for a, b in PAIRS]
        _, a, b = max(differences)
        ia, ib = TICKERS.index(a), TICKERS.index(b)
        return (ia, ib) if recent[ia] < recent[ib] else (ib, ia)
    if name == "session_switch":
        hour = datetime.fromisoformat(schedule[index]["timestamp"]).hour
        return choose_extremes(recent, reverse=hour >= 12)
    if name == "news_gated_reversal":
        ceiling = np.quantile(counts[index], 0.75)
        allowed = np.flatnonzero(counts[index] <= ceiling)
        return choose_extremes(recent, reverse=True, allowed=allowed)
    raise ValueError(name)


def online_method(name, prices, counts, schedule, eligible, index):
    include_news = name == "online_ridge_news"
    x_now = features(prices, counts, schedule, index, include_news)
    if x_now is None:
        return None
    examples = []
    targets = []
    now = datetime.fromisoformat(schedule[index]["timestamp"])
    for earlier, entry, exit_index in eligible:
        if earlier >= index or datetime.fromisoformat(schedule[exit_index]["timestamp"]) > now:
            continue
        x_past = features(prices, counts, schedule, earlier, include_news)
        if x_past is None:
            continue
        result = prices[exit_index] / prices[entry] - 1
        examples.append(x_past)
        targets.append((result - np.mean(result)) / 0.01)
    if len(examples) < 5:
        return None
    model = Ridge(alpha=30.0, fit_intercept=False)
    model.fit(np.concatenate(examples), np.concatenate(targets))
    scores = model.predict(x_now)
    return choose_extremes(scores)


def adaptive_method(name, prices, counts, schedule, eligible, index):
    completed = []
    now = datetime.fromisoformat(schedule[index]["timestamp"])
    for earlier, entry, exit_index in eligible:
        if earlier >= index or datetime.fromisoformat(schedule[exit_index]["timestamp"]) > now:
            continue
        completed.append((earlier, entry, exit_index))
    if len(completed) < 5:
        return fixed_method("related_pair_reversal", prices, counts, schedule, index)
    scores = {}
    for expert in ("related_pair_reversal", "momentum"):
        outcomes = []
        for earlier, entry, exit_index in completed[-12:]:
            pair = fixed_method(expert, prices, counts, schedule, earlier)
            outcomes.append(pair_return(pair, prices[entry], prices[exit_index]))
        weights = np.exp(np.log(0.5) * np.arange(len(outcomes) - 1, -1, -1) / 6)
        scores[expert] = float(np.average(outcomes, weights=weights))
    winner = max(scores, key=scores.get)
    if name == "adaptive_expert_flat" and scores[winner] <= 0.001:
        return None
    return fixed_method(winner, prices, counts, schedule, index)


def pair_return(pair, entry, exit_prices):
    if pair is None:
        return 0.0
    long_index, short_index = pair
    return 0.5 * (exit_prices[long_index] / entry[long_index] - exit_prices[short_index] / entry[short_index])


def decide(name, prices, counts, schedule, eligible, index):
    if name.startswith("online_"):
        return online_method(name, prices, counts, schedule, eligible, index)
    if name.startswith("adaptive_"):
        return adaptive_method(name, prices, counts, schedule, eligible, index)
    return fixed_method(name, prices, counts, schedule, index)


def metrics(returns):
    values = np.asarray(returns, dtype=float)
    equity = np.concatenate(([1.0], np.cumprod(1 + values)))
    volatility = float(np.std(values, ddof=1))
    return {
        "cumulative_return": float(equity[-1] - 1),
        "sharpe": float(np.mean(values) / volatility) if volatility > 0 else None,
        "max_drawdown": float(np.max(1 - equity / np.maximum.accumulate(equity))),
        "active_rounds": int(np.count_nonzero(values)),
    }


def run(root: Path, state_path: Path):
    state, schedule, prices, counts, eligible = read_data(root, state_path)
    baselines = {row["team_name"]: row["steps"] for row in state["rows"] if row["is_baseline"]}
    assert set(baselines) == {"Codex baseline", "FinMem baseline", "Moira baseline"}
    records = {name: [] for name in (*METHODS, *baselines)}
    for index, entry, exit_index in eligible:
        date = datetime.fromisoformat(schedule[index]["timestamp"]).date().isoformat()
        known = [(earlier, begin, end) for earlier, begin, end in eligible if end <= index]
        for name in METHODS:
            pair = decide(name, prices, counts, schedule, eligible, index)
            causal_pair = decide(name, prices[: index + 1], counts[: index + 1], schedule[: index + 1], known, index)
            assert pair == causal_pair, (name, index)
            value = pair_return(pair, prices[entry], prices[exit_index])
            records[name].append({"round": index + 1, "date": date, "pair": None if pair is None else [TICKERS[pair[0]], TICKERS[pair[1]]], "return": value})
        for name, steps in baselines.items():
            step = steps[index]
            assert step["step"] == index + 1
            pair = None if step["action"] == "F" else (TICKERS.index(step["stock1"]), TICKERS.index(step["stock2"]))
            if step["action"] == "SL":
                pair = (pair[1], pair[0])
            value = pair_return(pair, prices[entry], prices[exit_index])
            records[name].append({"round": index + 1, "date": date, "pair": None if pair is None else [TICKERS[pair[0]], TICKERS[pair[1]]], "return": value})
    development_dates = {"2026-10-14", "2026-10-15", "2026-10-19", "2026-10-20"}
    test_dates = {"2026-10-21", "2026-10-22", "2026-10-23"}
    assert {row["date"] for row in records["flat"]} == development_dates | test_dates
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    report = {"protocol": "release to next release entry, following release exit", "dataset_revision": revision, "manifest_sha256": hashlib.sha256((root / "releases.json").read_bytes()).hexdigest(), "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(), "causality_prefix_audit": "passed", "evaluable_rounds": [index + 1 for index, _, _ in eligible], "development_dates": sorted(development_dates), "test_dates": sorted(test_dates), "methods": {}}
    for name, rows in records.items():
        development = [row["return"] for row in rows if row["date"] in development_dates]
        test = [row["return"] for row in rows if row["date"] in test_dates]
        days = {date: metrics([row["return"] for row in rows if row["date"] == date]) for date in sorted(development_dates | test_dates)}
        report["methods"][name] = {"development": metrics(development), "test": metrics(test), "all": metrics([row["return"] for row in rows]), "daily": days, "records": rows}
    report["selected_by_development"] = max(METHODS, key=lambda name: report["methods"][name]["development"]["cumulative_return"])
    for cost_bps in (5, 10):
        for name, result in report["methods"].items():
            net = [row["return"] - (2 * cost_bps / 10000 if row["pair"] else 0) for row in result["records"]]
            result[f"all_net_{cost_bps}bps_per_leg_side"] = metrics(net)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("research_import/task2_dataset"))
    parser.add_argument("--state", type=Path, default=Path("research_import/task2_live_state.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/task2_experiment.json"))
    args = parser.parse_args()
    report = run(args.data, args.state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, result in sorted(report["methods"].items(), key=lambda item: item[1]["development"]["cumulative_return"], reverse=True):
        print(f"{name:24} dev={result['development']['cumulative_return']:+.4%} test={result['test']['cumulative_return']:+.4%} all={result['all']['cumulative_return']:+.4%} sharpe={result['all']['sharpe']}")
    print("Selected from development:", report["selected_by_development"])
    print("Report:", args.output)


if __name__ == "__main__":
    main()
