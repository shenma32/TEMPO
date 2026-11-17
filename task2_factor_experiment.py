import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from task2_experiment import TICKERS, metrics, read_data
from task2_factor_agent import TRADE_COST, TRADE_MARGIN, choose_action, factor_return, forecast_components
from task2_execution_policy import execution_decision, policy_features
from task2_hourly_validation import prepare


DEVELOPMENT_DATES = {"2026-10-14", "2026-10-15", "2026-10-19", "2026-10-20"}
VALIDATION_DATES = {"2026-10-21", "2026-10-22", "2026-10-23"}


def historical_relation(cache):
    schedule, prices, _, _, dates = prepare(cache)
    assert len(schedule) == 6 * len(dates)
    development = [day for day, date in enumerate(dates) if date < "2026-01-01"]
    validation = [day for day, date in enumerate(dates) if date >= "2026-01-01"]
    assert development and validation
    samples = []
    for day in development:
        start = day * 6
        for slot in range(1, 6):
            hours = (datetime.fromisoformat(schedule[start + slot]["timestamp"]) -
                     datetime.fromisoformat(schedule[start + slot - 1]["timestamp"])).total_seconds() / 3600
            samples.append(np.log(prices[start + slot] / prices[start + slot - 1]) / np.sqrt(hours))
    covariance = LedoitWolf().fit(np.asarray(samples)).covariance_
    volatility = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(volatility, volatility)
    beta = covariance / np.diag(covariance)[None, :]
    residual_volatility = np.sqrt(np.maximum(np.diag(covariance)[:, None] - covariance * beta, 0))
    results = {}
    for label, day_indexes in (("development", development), ("validation", validation)):
        events = []
        chosen = []
        for day in day_indexes:
            start = day * 6
            for slot in (1, 2, 3):
                index = start + slot
                elapsed = (datetime.fromisoformat(schedule[index]["timestamp"]) -
                           datetime.fromisoformat(schedule[index - 1]["timestamp"])).total_seconds() / 3600
                recent = np.log(prices[index] / prices[index - 1])
                future = prices[index + 2] / prices[index + 1] - 1
                candidates = []
                for first in range(len(TICKERS)):
                    for second in range(first + 1, len(TICKERS)):
                        rho = correlation[first, second]
                        if rho < 0.45:
                            continue
                        residual = recent[first] - beta[first, second] * recent[second]
                        sigma = residual_volatility[first, second] * np.sqrt(elapsed)
                        z = residual / sigma
                        if abs(z) <= 2:
                            continue
                        gross = -np.sign(residual) * (future[first] - future[second]) / 2
                        candidates.append({"date": dates[day], "first": TICKERS[first], "second": TICKERS[second],
                                           "z": float(z), "score": float((abs(z) - 2) * rho),
                                           "gross_return": float(gross), "net_return": float(gross - TRADE_COST)})
                events.extend(candidates)
                if candidates:
                    chosen.append(max(candidates, key=lambda row: row["score"]))
        results[label] = {
            "triggers": len(events),
            "trigger_days": len({row["date"] for row in events}),
            "mean_gross_bps_all_triggers": float(np.mean([row["gross_return"] for row in events]) * 10000) if events else None,
            "selected_trades": len(chosen),
            "selected_mean_net_bps": float(np.mean([row["net_return"] for row in chosen]) * 10000) if chosen else None,
            "selected_metrics": metrics([row["net_return"] for row in chosen]) if len(chosen) >= 2 else None,
            "selected_daily_net_bps": {date: float(sum(row["net_return"] for row in chosen if row["date"] == date) * 10000)
                                       for date in sorted({row["date"] for row in chosen})},
        }
    return {"development_dates": len(development), "validation_dates": len(validation),
            "training_returns": len(samples), "pairs_above_correlation_threshold": int(np.sum(np.triu(correlation, 1) >= 0.45)),
            "results": results}


def competition_ablation(root, state_path, report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _, schedule, prices, _, eligible = read_data(root, state_path)
    assert len(report["records"]) == len(schedule)
    eligible_by_index = {index: (entry, exit_index) for index, entry, exit_index in eligible}
    methods = {
        "event": {"kinds": {"event"}, "factors": [], "rows": [], "features": {}, "training": []},
        "event_relation": {"kinds": {"event", "relation"}, "factors": [], "rows": [], "features": {}, "training": []},
        "full": {"kinds": {"event", "relation", "feedback"}, "factors": [], "rows": [], "features": {}, "training": []},
    }
    policy = report["factor_parameters"].get("execution_policy", "factor_margin_5")
    for index, record in enumerate(report["records"]):
        entry_at = datetime.fromisoformat(schedule[index]["entry_at"])
        exit_at = datetime.fromisoformat(schedule[index]["exit_at"])
        for name, method in methods.items():
            for source_index, entry, exit_index in eligible:
                if exit_index != index:
                    continue
                actual = prices[exit_index] / prices[entry] - 1
                method["training"].append({"source_round": source_index + 1, "exit_index": index,
                                            "features": method["features"][source_index],
                                            "target": (actual - np.mean(actual)) / 0.01})
            method["factors"].extend(factor for factor in record["new_factors"] if factor["kind"] in method["kinds"])
            predicted, arithmetic = forecast_components(method["factors"], entry_at, exit_at)
            decision, _ = execution_decision(policy, predicted, arithmetic, prices[:index + 1],
                                             schedule[:index + 1], index, method["training"])
            method["features"][index] = policy_features(arithmetic, prices[:index + 1], schedule[:index + 1], index)
            if name == "full":
                assert decision == record["decision"]
            if index not in eligible_by_index:
                continue
            entry, exit_index = eligible_by_index[index]
            long_ticker = decision["stock1"]
            short_ticker = decision["stock2"]
            if decision["action"] == "F":
                value = 0.0
            else:
                actual = prices[exit_index] / prices[entry] - 1
                value = float((actual[TICKERS.index(long_ticker)] - actual[TICKERS.index(short_ticker)]) / 2 - TRADE_COST)
            method["rows"].append({"round": index + 1,
                                   "date": datetime.fromisoformat(schedule[index]["timestamp"]).date().isoformat(),
                                   "action": decision["action"], "stock1": long_ticker, "stock2": short_ticker,
                                   "net_return": value})
    result = {}
    for name, method in methods.items():
        rows = method["rows"]
        assert len(rows) == len(eligible)
        result[name] = {
            "development": metrics([row["net_return"] for row in rows if row["date"] in DEVELOPMENT_DATES]),
            "validation": metrics([row["net_return"] for row in rows if row["date"] in VALIDATION_DATES]),
            "all": metrics([row["net_return"] for row in rows]),
            "rows": rows,
        }
    return result


def news_source_overlap(root):
    manifest = json.loads((root / "releases.json").read_text(encoding="utf-8"))
    previous = {}
    repeated = 0
    total = 0
    records = 0
    for release in manifest["published_rounds"]:
        folder = root / "data" / release["folder_url"].split("/")[-1]
        news = pd.read_parquet(folder / "news.parquet").set_index("ticker")
        for ticker in TICKERS:
            links = set(news.loc[ticker, "source_links"])
            total += len(links)
            repeated += len(links & previous.get(ticker, set()))
            records += int(bool(links & previous.get(ticker, set())))
            previous[ticker] = links
    return {"source_links": total, "repeated_from_previous_round": repeated,
            "ticker_rounds_with_repeated_sources": records}


def decision_objective_experiment(root, state_path, report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _, schedule, prices, _, eligible = read_data(root, state_path)
    eligible_by_index = {index: (entry, exit_index) for index, entry, exit_index in eligible}
    factors = []
    rows = []
    for index, record in enumerate(report["records"]):
        factors.extend(record["new_factors"])
        entry_at = datetime.fromisoformat(schedule[index]["entry_at"])
        exit_at = datetime.fromisoformat(schedule[index]["exit_at"])
        positive = np.zeros(len(TICKERS))
        negative = np.zeros(len(TICKERS))
        for factor in factors:
            value = factor_return(factor, entry_at, exit_at)
            ticker_index = TICKERS.index(factor["ticker"])
            if value >= 0:
                positive[ticker_index] += value
            else:
                negative[ticker_index] -= value
        geometric = forecast_components(factors, entry_at, exit_at)[0]
        long_index = int(np.argmax(geometric))
        short_index = int(np.argmin(geometric))
        arithmetic = (positive - negative) / 2
        predicted_net = (arithmetic[long_index] - arithmetic[short_index]) / 2 - TRADE_COST
        action = "LS" if predicted_net > TRADE_MARGIN else "F"
        if index not in eligible_by_index:
            continue
        entry, exit_index = eligible_by_index[index]
        if action == "F":
            actual_net = 0.0
        else:
            actual = prices[exit_index] / prices[entry] - 1
            actual_net = float((actual[long_index] - actual[short_index]) / 2 - TRADE_COST)
        rows.append({"round": index + 1,
                     "date": datetime.fromisoformat(schedule[index]["timestamp"]).date().isoformat(),
                     "action": action, "stock1": TICKERS[long_index] if action == "LS" else None,
                     "stock2": TICKERS[short_index] if action == "LS" else None,
                     "predicted_net_bps": float(predicted_net * 10000),
                     "net_return": actual_net})
    return {"development": metrics([row["net_return"] for row in rows if row["date"] in DEVELOPMENT_DATES]),
            "validation": metrics([row["net_return"] for row in rows if row["date"] in VALIDATION_DATES]),
            "all": metrics([row["net_return"] for row in rows]), "rows": rows}


def event_parameter_experiment(root, state_path, report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _, schedule, prices, _, eligible = read_data(root, state_path)
    eligible_by_index = {index: (entry, exit_index) for index, entry, exit_index in eligible}
    results = {}
    for amplitude_scale in (0.5, 1.0, 2.0):
        for time_scale in (0.5, 1.0, 2.0):
            factors = []
            rows = []
            for index, record in enumerate(report["records"]):
                for factor in record["new_factors"]:
                    if factor["kind"] != "event":
                        continue
                    adjusted = factor.copy()
                    adjusted["amplitude"] *= amplitude_scale
                    adjusted["time_scale_hours"] *= time_scale
                    factors.append(adjusted)
                entry_at = datetime.fromisoformat(schedule[index]["entry_at"])
                exit_at = datetime.fromisoformat(schedule[index]["exit_at"])
                geometric, arithmetic = forecast_components(factors, entry_at, exit_at)
                decision = choose_action(geometric, arithmetic)
                if index not in eligible_by_index:
                    continue
                entry, exit_index = eligible_by_index[index]
                if decision["action"] == "F":
                    net = 0.0
                else:
                    actual = prices[exit_index] / prices[entry] - 1
                    net = float((actual[TICKERS.index(decision["stock1"])] -
                                 actual[TICKERS.index(decision["stock2"])]) / 2 - TRADE_COST)
                rows.append({"date": datetime.fromisoformat(schedule[index]["timestamp"]).date().isoformat(),
                             "net_return": net})
            key = f"amplitude_{amplitude_scale}_time_{time_scale}"
            results[key] = {
                "development": metrics([row["net_return"] for row in rows if row["date"] in DEVELOPMENT_DATES]),
                "validation": metrics([row["net_return"] for row in rows if row["date"] in VALIDATION_DATES]),
            }
    selected = max(results, key=lambda name: results[name]["development"]["cumulative_return"])
    return {"candidates": results, "selected_by_development": selected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("research_import/task2_dataset"))
    parser.add_argument("--state", type=Path, default=Path("research_import/task2_live_state.json"))
    parser.add_argument("--hourly-cache", type=Path, default=Path("research_import/task2_yahoo_1h"))
    parser.add_argument("--report", type=Path, default=Path("outputs/task2_factor_agent_original.json"))
    parser.add_argument("--source-report", type=Path, default=Path("outputs/task2_factor_source_agent.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/task2_factor_experiment.json"))
    args = parser.parse_args()
    result = {"historical_relation": historical_relation(args.hourly_cache),
              "competition_ablation": competition_ablation(args.data, args.state, args.report),
              "news_source_overlap": news_source_overlap(args.data),
              "decision_objective": decision_objective_experiment(args.data, args.state, args.report),
              "event_parameter_experiment": event_parameter_experiment(args.data, args.state, args.report),
              "source_grounded": json.loads(args.source_report.read_text(encoding="utf-8"))["factor_metrics"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"historical_relation": {key: value for key, value in result["historical_relation"].items() if key != "results"},
                      "relation_results": {name: {key: value[key] for key in ("triggers", "trigger_days", "mean_gross_bps_all_triggers", "selected_trades", "selected_mean_net_bps", "selected_metrics")}
                                           for name, value in result["historical_relation"]["results"].items()},
                      "competition_ablation": {name: {key: value[key] for key in ("development", "validation", "all")}
                                               for name, value in result["competition_ablation"].items()},
                      "news_source_overlap": result["news_source_overlap"],
                      "decision_objective": {key: result["decision_objective"][key] for key in ("development", "validation", "all")},
                      "event_parameter_experiment": result["event_parameter_experiment"],
                      "source_grounded": result["source_grounded"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
