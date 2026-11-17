import argparse
import json
import platform
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from task2_experiment import TICKERS, metrics, read_data
from task2_factor_agent import forecast_components, geometric_return, interval_weight
from task2_factor_agent_audit import audit
from task2_factor_experiment import DEVELOPMENT_DATES, VALIDATION_DATES, competition_ablation


def reference_forecast(factors, entry_at, exit_at):
    positive = np.zeros(len(TICKERS))
    negative = np.zeros(len(TICKERS))
    for factor in factors:
        amplitude = factor["amplitude"]
        if amplitude == 0:
            continue
        created = datetime.fromisoformat(factor["created_at"])
        start = (entry_at - created).total_seconds() / 3600
        end = (exit_at - created).total_seconds() / 3600
        scale = factor["time_scale_hours"]
        assert 0 <= start < end and scale > 0
        if factor["kind"] == "event":
            value = amplitude * (np.exp(-((start / scale) ** 0.75)) - np.exp(-((end / scale) ** 0.75)))
        elif factor["kind"] == "relation":
            value = amplitude * (np.exp(-start / scale) - np.exp(-end / scale))
        else:
            assert factor["kind"] == "feedback"
            value = amplitude * ((1 + start / scale) ** -1.5 - (1 + end / scale) ** -1.5)
        position = TICKERS.index(factor["ticker"])
        if value >= 0:
            positive[position] += value
        else:
            negative[position] -= value
    return (np.asarray([geometric_return(a, b) for a, b in zip(positive, negative)]),
            (positive - negative) / 2)


def benchmark(report, schedule, repeats=31):
    cases = []
    factors = []
    for record, timing in zip(report["records"], schedule):
        factors.extend(record["new_factors"])
        case = (factors.copy(), datetime.fromisoformat(timing["entry_at"]),
                datetime.fromisoformat(timing["exit_at"]))
        expected = reference_forecast(*case)
        actual = forecast_components(*case)
        for left, right in zip(actual, expected):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(actual[0], record["predicted_stock_returns"])
        cases.append(case)
    timings = {"reference": [], "cached": []}
    for repetition in range(repeats):
        order = ("reference", "cached") if repetition % 2 == 0 else ("cached", "reference")
        for name in order:
            interval_weight.cache_clear()
            function = reference_forecast if name == "reference" else forecast_components
            started = perf_counter()
            for case in cases:
                function(*case)
            timings[name].append((perf_counter() - started) * 1000)
    medians = {name: float(np.median(values)) for name, values in timings.items()}
    interval_weight.cache_clear()
    for case in cases:
        forecast_components(*case)
    return {"repetitions": repeats, "rounds_per_repetition": len(cases),
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "median_ms": medians,
            "speedup": medians["reference"] / medians["cached"],
            "exact_prediction_equality": True, "cache": interval_weight.cache_info()._asdict(),
            "timings_ms": timings}


def summarize(report):
    rows = [row for row in report["records"] if "net_return" in row]
    dates = [row["published_at"][:10] for row in rows]
    daily = {date: metrics([row["net_return"] for row in rows if row["published_at"][:10] == date])
             for date in sorted(set(dates))}
    result = {"all": report["factor_metrics"],
              "development": metrics([row["net_return"] for row in rows if row["published_at"][:10] in DEVELOPMENT_DATES]),
              "later_dates": metrics([row["net_return"] for row in rows if row["published_at"][:10] in VALIDATION_DATES]),
              "daily": daily,
              "cost_sensitivity_bps": {str(cost): metrics([
                  row["gross_return"] - (cost / 10000 if row["pair"] is not None else 0) for row in rows
              ]) for cost in (0, 5, 10, 20)},
              "leave_one_date_out": {date: metrics([
                  row["net_return"] for row in rows if row["published_at"][:10] != date
              ]) for date in sorted(set(dates))}}
    usages = [usage for row in report["records"] for usage in row["usage"].values() if usage is not None]
    result["api"] = {"requests": len(usages), "feedback_requests": len(usages) - len(report["records"]),
                     "prompt_tokens": sum(row["prompt_tokens"] for row in usages),
                     "completion_tokens": sum(row["completion_tokens"] for row in usages)}
    return result


def baseline_comparison(root, state_path):
    _, schedule, prices, _, eligible = read_data(root, state_path)
    saved = json.loads(Path("outputs/task2_sentiment.json").read_text(encoding="utf-8"))
    baseline = saved["methods"]["finbert_plus_momentum"]
    rows = baseline["records"]
    assert len(rows) == len(eligible)
    net = []
    for row, (index, entry, exit_index) in zip(rows, eligible):
        assert row["round"] == index + 1 and row["date"] == schedule[index]["timestamp"][:10]
        if row["pair"] is None:
            gross = 0.0
        else:
            actual = prices[exit_index] / prices[entry] - 1
            first, second = [TICKERS.index(ticker) for ticker in row["pair"]]
            gross = float((actual[first] - actual[second]) / 2)
        assert np.isclose(gross, row["return"], rtol=0, atol=1e-15)
        net.append(gross - (0.001 if row["pair"] else 0))
    return {"finbert_plus_momentum": {
        "all": metrics(net),
        "development": metrics([value for row, value in zip(rows, net) if row["date"] in DEVELOPMENT_DATES]),
        "later_dates": metrics([value for row, value in zip(rows, net) if row["date"] in VALIDATION_DATES]),
        "model_revision": saved["model_revision"], "summary_sha256": saved["summary_sha256"]},
        "flat": metrics([0.0] * len(rows))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("research_import/task2_dataset"))
    parser.add_argument("--state", type=Path, default=Path("research_import/task2_live_state.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/task2_factor_research.json"))
    args = parser.parse_args()
    paths = {"geometric": Path("outputs/task2_factor_agent_before_return_optimization.json"),
             "arithmetic": Path("outputs/task2_factor_arithmetic.json"),
             "relative": Path("outputs/task2_factor_relative.json")}
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    results = {name: summarize(report) for name, report in reports.items()}
    selected = max(results, key=lambda name: results[name]["development"]["cumulative_return"])
    _, schedule, _, _, _ = read_data(args.data, args.state)
    result = {"selection_rule": "highest cumulative return on the first four dates; all dates previously explored",
              "selected_feedback_basis": selected, "methods": results,
              "audits": {name: audit(args.data, args.state, path) for name, path in paths.items()},
              "benchmark": benchmark(reports["geometric"], schedule),
              "baselines": baseline_comparison(args.data, args.state),
              "factor_ablation": competition_ablation(args.data, args.state, paths["geometric"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected,
                      "methods": {name: {key: value[key] for key in ("all", "development", "later_dates", "api")}
                                  for name, value in results.items()},
                      "benchmark": {key: value for key, value in result["benchmark"].items() if key != "timings_ms"}}, indent=2))


if __name__ == "__main__":
    main()
