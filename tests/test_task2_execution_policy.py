import json
from pathlib import Path

import numpy as np

from task2_experiment import TICKERS, read_data


ROOT = Path(__file__).resolve().parents[1]


def read_report(name):
    return json.loads((ROOT / "outputs" / name).read_text(encoding="utf-8"))


def test_complete_agent_matches_selected_policy_experiment():
    before = read_report("task2_factor_agent_before_return_optimization.json")
    after = read_report("task2_factor_agent.json")
    experiment = read_report("task2_return_experiment.json")
    selected = experiment["selected"]
    assert after["factor_parameters"]["execution_policy"] == selected
    expected = experiment["methods"][selected]["records"]
    assert len(before["records"]) == len(after["records"]) == len(expected) == 41
    for original, current, row in zip(before["records"], after["records"], expected):
        for key in ("new_factors", "predicted_stock_returns", "predicted_stock_arithmetic_returns",
                    "event_analysis", "feedback_analysis", "usage"):
            assert original[key] == current[key]
        assert current["decision"] == row["decision"]
        if "net_return" in row:
            assert current["net_return"] == row["net_return"]


def test_actual_prefix_execution_matches_complete_execution():
    complete = read_report("task2_factor_agent.json")
    prefix = read_report("task2_factor_optimized_prefix.json")
    assert prefix["rounds_processed"] == 25
    for observed, expected in zip(prefix["records"], complete["records"]):
        for key in ("round", "decision", "predicted_stock_returns", "predicted_stock_arithmetic_returns",
                    "policy_features", "execution_details", "new_factors", "feedback_candidates"):
            assert observed[key] == expected[key]


def test_selected_returns_reconcile_with_public_entry_and_exit_prices():
    report = read_report("task2_factor_agent.json")
    _, _, prices, _, eligible = read_data(ROOT / "research_import/task2_dataset",
                                          ROOT / "research_import/task2_live_state.json")
    values = []
    for index, entry, exit_index in eligible:
        record = report["records"][index]
        decision = record["decision"]
        if decision["action"] == "F":
            actual = 0.0
        else:
            long_index = TICKERS.index(decision["stock1"])
            short_index = TICKERS.index(decision["stock2"])
            actual = ((prices[exit_index, long_index] - prices[entry, long_index]) / prices[entry, long_index] -
                      (prices[exit_index, short_index] - prices[entry, short_index]) / prices[entry, short_index]) / 2 - 0.001
        np.testing.assert_allclose(actual, record["net_return"], rtol=0, atol=1e-15)
        values.append(actual)
    np.testing.assert_allclose(np.prod(1 + np.asarray(values)) - 1,
                               report["factor_metrics"]["cumulative_return"], rtol=0, atol=1e-14)
