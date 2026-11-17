import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from task2_execution_policy import POLICIES, execution_decision, policy_features
from task2_experiment import TICKERS, metrics, read_data
from task2_factor_experiment import DEVELOPMENT_DATES, VALIDATION_DATES
from task2_factor_agent import forecast_components


def evaluate(root, state_path, report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _, schedule, prices, _, eligible = read_data(root, state_path)
    reports = {name: [] for name in POLICIES}
    training = []
    features = {}
    eligible_by_index = {index: (entry, exit_index) for index, entry, exit_index in eligible}
    for index, record in enumerate(report["records"]):
        for source_index, entry, exit_index in eligible:
            if exit_index != index:
                continue
            target = prices[exit_index] / prices[entry] - 1
            training.append({"source_round": source_index + 1, "exit_index": exit_index,
                             "features": features[source_index],
                             "target": (target - np.mean(target)) / 0.01})
        geometric = np.asarray(record["predicted_stock_returns"])
        arithmetic = np.asarray(record["predicted_stock_arithmetic_returns"])
        features[index] = policy_features(arithmetic, prices[:index + 1], schedule[:index + 1], index)
        assert all(row["exit_index"] <= index for row in training)
        for name in POLICIES:
            decision, details = execution_decision(name, geometric, arithmetic,
                                                   prices[:index + 1], schedule[:index + 1], index, training)
            prefix_decision, prefix_details = execution_decision(name, geometric, arithmetic,
                                                                 prices, schedule, index, training)
            assert decision == prefix_decision and details == prefix_details
            if name == "factor_margin_5":
                assert decision == record["decision"]
            row = {"round": index + 1, "date": schedule[index]["timestamp"][:10],
                   "decision": decision, "details": details,
                   "training_source_rounds": [item["source_round"] for item in training]}
            if index in eligible_by_index:
                entry, exit_index = eligible_by_index[index]
                if decision["action"] == "F":
                    gross = 0.0
                else:
                    actual = prices[exit_index] / prices[entry] - 1
                    first, second = TICKERS.index(decision["stock1"]), TICKERS.index(decision["stock2"])
                    gross = float((actual[first] - actual[second]) / 2)
                row["gross_return"] = gross
                row["net_return"] = gross - (0.001 if decision["action"] != "F" else 0.0)
            reports[name].append(row)
    results = {}
    for name, rows in reports.items():
        settled = [row for row in rows if "net_return" in row]
        results[name] = {
            "development": metrics([row["net_return"] for row in settled if row["date"] in DEVELOPMENT_DATES]),
            "later_dates": metrics([row["net_return"] for row in settled if row["date"] in VALIDATION_DATES]),
            "all": metrics([row["net_return"] for row in settled]),
            "daily": {date: metrics([row["net_return"] for row in settled if row["date"] == date])
                      for date in sorted(DEVELOPMENT_DATES | VALIDATION_DATES)},
            "records": rows,
        }
    selected = max(results, key=lambda name: results[name]["development"]["cumulative_return"])
    return {"selection_rule": "maximum development net cumulative return; ties follow declaration order",
            "search_stages": {"initial": [name for name in POLICIES if name not in {
                "momentum_weight_1_5", "momentum_weight_2", "momentum_margin_0", "momentum_margin_2_5", "momentum_margin_10"}],
                "momentum_refinement": ["momentum_weight_1_5", "momentum_weight_2", "momentum_margin_0", "momentum_margin_2_5", "momentum_margin_10"]},
            "source_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "policies": POLICIES, "selected": selected, "methods": results,
            "prefix_audit": "passed", "feedback_target": "unchanged raw factor forecasts; all settled stocks"}


def component_ablation(root, state_path, report_path, policy):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _, schedule, prices, _, eligible = read_data(root, state_path)
    eligible_by_index = {index: (entry, exit_index) for index, entry, exit_index in eligible}
    kinds = {"price_only": set(), "event_price": {"event"},
             "event_relation_price": {"event", "relation"},
             "full": {"event", "relation", "feedback"}}
    factors = []
    results = {name: [] for name in kinds}
    for index, record in enumerate(report["records"]):
        factors.extend(record["new_factors"])
        for name, accepted in kinds.items():
            selected = [factor for factor in factors if factor["kind"] in accepted]
            geometric, arithmetic = forecast_components(selected, datetime.fromisoformat(schedule[index]["entry_at"]),
                                                        datetime.fromisoformat(schedule[index]["exit_at"]))
            decision, _ = execution_decision(policy, geometric, arithmetic, prices[:index + 1],
                                             schedule[:index + 1], index, [])
            if index not in eligible_by_index:
                continue
            entry, exit_index = eligible_by_index[index]
            actual = prices[exit_index] / prices[entry] - 1
            if decision["action"] == "F":
                value = 0.0
            else:
                value = float((actual[TICKERS.index(decision["stock1"])] -
                               actual[TICKERS.index(decision["stock2"])]) / 2 - 0.001)
            results[name].append({"round": index + 1, "date": schedule[index]["timestamp"][:10],
                                  "decision": decision, "net_return": value})
    return {name: {"development": metrics([row["net_return"] for row in rows if row["date"] in DEVELOPMENT_DATES]),
                   "later_dates": metrics([row["net_return"] for row in rows if row["date"] in VALIDATION_DATES]),
                   "all": metrics([row["net_return"] for row in rows]), "records": rows}
            for name, rows in results.items()}


def selection_diagnostics(result):
    rows = [row for row in result["methods"][result["selected"]]["records"] if "net_return" in row]
    reference = [row for row in result["methods"]["factor_margin_5"]["records"] if "net_return" in row]
    changes = []
    for row, before in zip(rows, reference):
        assert row["round"] == before["round"]
        fields = ("action", "stock1", "stock2")
        if any(row["decision"][key] != before["decision"][key] for key in fields):
            changes.append({"round": row["round"], "date": row["date"],
                            "before": {key: before["decision"][key] for key in fields},
                            "after": {key: row["decision"][key] for key in fields},
                            "before_net_return": before["net_return"], "after_net_return": row["net_return"]})
    return {"cost_sensitivity_bps": {str(cost): metrics([
                row["gross_return"] - (cost / 10000 if row["decision"]["action"] != "F" else 0.0)
                for row in rows]) for cost in (0, 5, 10, 20)},
            "changed_rounds": changes,
            "leave_one_date_out": {date: metrics([row["net_return"] for row in rows if row["date"] != date])
                                   for date in sorted(DEVELOPMENT_DATES | VALIDATION_DATES)},
            "daily_net_return_difference": {date: result["methods"][result["selected"]]["daily"][date]["cumulative_return"] -
                                             result["methods"]["factor_margin_5"]["daily"][date]["cumulative_return"]
                                             for date in sorted(DEVELOPMENT_DATES | VALIDATION_DATES)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("research_import/task2_dataset"))
    parser.add_argument("--state", type=Path, default=Path("research_import/task2_live_state.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/task2_factor_agent_before_return_optimization.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/task2_return_experiment.json"))
    args = parser.parse_args()
    result = evaluate(args.data, args.state, args.report)
    result["component_ablation"] = component_ablation(args.data, args.state, args.report, result["selected"])
    result["selection_diagnostics"] = selection_diagnostics(result)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, method in result["methods"].items():
        print(f"{name:25} dev={method['development']['cumulative_return']:+.4%} later={method['later_dates']['cumulative_return']:+.4%} all={method['all']['cumulative_return']:+.4%} trades={method['all']['active_rounds']} mdd={method['all']['max_drawdown']:.4%}")
    print("Selected:", result["selected"])
    for name, method in result["component_ablation"].items():
        print(f"ablation {name:22} dev={method['development']['cumulative_return']:+.4%} later={method['later_dates']['cumulative_return']:+.4%} all={method['all']['cumulative_return']:+.4%} trades={method['all']['active_rounds']}")


if __name__ == "__main__":
    main()
