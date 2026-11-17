import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from task2_experiment import TICKERS, read_data
from task2_execution_policy import execution_decision, policy_features
from task2_factor_agent import (EVENT_LEVEL, FEEDBACK_STRENGTH, RELATION_MIN_CORRELATION,
                                TRADE_COST, feedback_candidates, forecast_components)


def audit(root, state_path, report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    state, schedule, prices, _, eligible = read_data(root, state_path)
    manifest = json.loads((root / "releases.json").read_text(encoding="utf-8"))
    assert report["manifest_sha256"] == hashlib.sha256((root / "releases.json").read_bytes()).hexdigest()
    assert report["state_sha256"] == hashlib.sha256(state_path.read_bytes()).hexdigest()
    assert len(report["records"]) == report["rounds_processed"] == len(schedule)
    eligible_by_index = {index: (entry, exit_index) for index, entry, exit_index in eligible}
    factors = []
    returns = []
    decision_counts = Counter()
    basis = report["factor_parameters"].get("feedback_basis", "geometric")
    policy = report["factor_parameters"].get("execution_policy", "factor_margin_5")
    training = []
    parameters = {key: np.asarray(value) for key, value in report["historical_parameters"].items()
                  if key in ("volatility", "relative_volatility")}
    for index, record in enumerate(report["records"]):
        release = manifest["published_rounds"][index]
        assert record["round"] == index + 1 and record["round_id"] == release["round_id"]
        published_at = datetime.fromisoformat(record["published_at"])
        assert published_at <= datetime.fromisoformat(record["action_due"])
        new_factors = record["new_factors"]
        settled = [row for row in report["records"][:index] if row.get("exit_index") == index]
        for row in settled:
            actual = np.asarray(row["actual_stock_returns"])
            source_index = row["round"] - 1
            training.append({"source_round": row["round"], "exit_index": index,
                             "features": policy_features(np.asarray(row["predicted_stock_arithmetic_returns"]),
                                                         prices[:source_index + 1], schedule[:source_index + 1], source_index),
                             "target": (actual - np.mean(actual)) / 0.01})
        candidates = feedback_candidates(settled, schedule, parameters, basis)
        if "feedback_candidates" in record:
            assert candidates == record["feedback_candidates"]
        candidates_by_ticker = {row["ticker"]: row for row in candidates}
        event_factors = [factor for factor in new_factors if factor["kind"] == "event"]
        assert set(factor["ticker"] for factor in event_factors) == set(TICKERS)
        assert len(record["event_analysis"]) == len(event_factors)
        for event, factor in zip(record["event_analysis"], event_factors):
            assert factor["ticker"] == event["ticker"]
            assert factor["level"] == event["level"]
            assert np.sign(factor["amplitude"]) == np.sign(EVENT_LEVEL[event["level"]])
        for factor in new_factors:
            assert factor["id"] == len(factors) + 1
            assert factor["ticker"] in TICKERS
            assert datetime.fromisoformat(factor["created_at"]) == published_at
            assert np.isfinite(factor["amplitude"])
            assert factor["time_scale_hours"] > 0
            if factor["kind"] == "event":
                assert factor["source_round"] == index + 1
            elif factor["kind"] == "relation":
                assert factor["correlation"] >= RELATION_MIN_CORRELATION
                assert factor["ticker"] in factor["pair"]
            else:
                assert factor["kind"] == "feedback"
                source_index = factor["source_round"] - 1
                assert source_index < index
                assert report["records"][source_index]["exit_index"] == index
                candidate = candidates_by_ticker[factor["ticker"]]
                assert candidate["source_round"] == factor["source_round"]
                assert factor["amplitude"] == candidate["error"] * FEEDBACK_STRENGTH[factor["strength"]]
            factors.append(factor)
        assert record["active_factor_count"] == len(factors)
        entry_at = datetime.fromisoformat(schedule[index]["entry_at"])
        exit_at = datetime.fromisoformat(schedule[index]["exit_at"])
        predicted, arithmetic = forecast_components(factors, entry_at, exit_at)
        np.testing.assert_allclose(record["predicted_stock_returns"], predicted, rtol=0, atol=1e-15)
        np.testing.assert_allclose(record["predicted_stock_arithmetic_returns"], arithmetic, rtol=0, atol=1e-15)
        expected_decision, details = execution_decision(policy, predicted, arithmetic,
                                                       prices[:index + 1], schedule[:index + 1], index, training)
        assert record["decision"] == expected_decision
        if "execution_details" in record:
            assert record["execution_details"] == details
            np.testing.assert_array_equal(record["policy_features"], policy_features(
                arithmetic, prices[:index + 1], schedule[:index + 1], index))
        decision_counts[record["decision"]["action"]] += 1
        if index not in eligible_by_index:
            assert "net_return" not in record
            continue
        entry_index, exit_index = eligible_by_index[index]
        assert (record["entry_index"], record["exit_index"]) == (entry_index, exit_index)
        if record["decision"]["action"] == "F":
            assert record["pair"] is None
        else:
            assert record["pair"] == [TICKERS.index(record["decision"]["stock1"]),
                                      TICKERS.index(record["decision"]["stock2"])]
        if exit_index >= len(report["records"]):
            assert "net_return" not in record
            continue
        assert datetime.fromisoformat(manifest["published_rounds"][exit_index]["published_at"]) > published_at
        entry_prices = prices[entry_index]
        actual_stock_returns = prices[exit_index] / entry_prices - 1
        np.testing.assert_allclose(record["actual_stock_returns"], actual_stock_returns, rtol=0, atol=1e-15)
        if record["pair"] is None:
            gross = 0.0
        else:
            long_index, short_index = record["pair"]
            gross = (actual_stock_returns[long_index] - actual_stock_returns[short_index]) / 2
        net = gross - (TRADE_COST if record["pair"] is not None else 0)
        assert np.isclose(record["gross_return"], gross, rtol=0, atol=1e-15)
        assert np.isclose(record["net_return"], net, rtol=0, atol=1e-15)
        returns.append(net)
    assert len(returns) == report["rounds_evaluable"] == len(eligible)
    assert np.isclose(report["factor_metrics"]["cumulative_return"], np.prod(1 + np.asarray(returns)) - 1)
    assert report["factor_metrics"]["active_rounds"] == sum(value != 0 for value in returns)
    return {"audit": "passed", "rounds": len(report["records"]),
            "evaluable_rounds": len(returns), "factors": len(factors),
            "decisions": dict(decision_counts),
            "cumulative_net_return": report["factor_metrics"]["cumulative_return"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("research_import/task2_dataset"))
    parser.add_argument("--state", type=Path, default=Path("research_import/task2_live_state.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/task2_factor_agent.json"))
    args = parser.parse_args()
    print(json.dumps(audit(args.data, args.state, args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
