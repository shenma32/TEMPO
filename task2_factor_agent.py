import argparse
import hashlib
import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.covariance import LedoitWolf

from task2_experiment import TICKERS, metrics, pair_return, read_data, recent_returns
from task2_execution_policy import (DEFAULT_EXECUTION_POLICY, POLICIES, TRADE_COST,
                                    execution_decision, policy_features)
from task2_hourly_validation import prepare


MODEL = "deepseek-flash"
PROMPTS = {
    "event": (
        "你是股票新闻事件分析员。只根据当轮公开摘要识别未来数小时可能影响价格的独立事件。"
        "逐只股票输出方向和幅度等级：unchanged、small_up、large_up、small_down、large_down；"
        "响应速度只可为 slow、medium、fast。没有明确方向性事件时使用 unchanged。"
        "每只股票输出一到三个事件；不同事件分别占一行，不要把几个事件合并。"
        "没有明确事件时只输出一行 unchanged。按输入的股票顺序排列，单只股票的事件连续排列。"
        "必须输出 JSON 对象，格式为 {\"events\":[{\"ticker\":\"NVDA\","
        "\"level\":\"unchanged\",\"speed\":\"medium\",\"reason\":\"简短依据\"}]}。"
        "不得使用外部信息或事后价格。"
    ),
    "feedback": (
        "你是股票预期误差分析员。只根据已经结算的预测、实际价格变化和当前公开摘要，"
        "判断是否需要记录一个持续衰减的误差反馈因子。方向由实际收益减去原预测收益确定，"
        "你只决定是否加入以及强度 low、medium、high。避免把普通随机波动当成持续影响。"
        "必须输出 JSON 对象，格式为 {\"feedback\":[{\"ticker\":\"NVDA\","
        "\"add\":false,\"strength\":\"low\",\"reason\":\"简短依据\"}]}。"
        "只为 candidates 数组中的股票输出结果，严格按照 candidates 的顺序排列，"
        "每只候选股票恰好出现一次。current_summaries 中的其他股票仅供参考，不能输出。"
        "不得使用尚未公布的数据。"
    ),
    "feedback_relative": (
        "你是股票相对收益误差分析员。只根据已经结算的预测、实际价格变化和当前公开摘要，"
        "判断是否需要记录一个持续衰减的误差反馈因子。输入 error 已扣除十只股票的平均预测误差，"
        "代表个股相对共同市场变化的误差，反馈方向由 error 确定。"
        "你只决定是否加入以及强度 low、medium、high。避免把普通随机波动当成持续影响。"
        "必须输出 JSON 对象，格式为 {\"feedback\":[{\"ticker\":\"NVDA\","
        "\"add\":false,\"strength\":\"low\",\"reason\":\"简短依据\"}]}。"
        "只为 candidates 数组中的股票输出结果，严格按照 candidates 的顺序排列，"
        "每只候选股票恰好出现一次。current_summaries 中的其他股票仅供参考，不能输出。"
        "不得使用尚未公布的数据。"
    ),
}
FLASH_FEEDBACK_PROMPTS = {
    "feedback": (
        "你是股票预期误差分析员。只根据已经结算的预测、实际价格变化和当前公开摘要，"
        "判断是否需要记录一个持续衰减的误差反馈因子。方向由实际收益减去原预测收益确定，"
        "你只决定是否加入以及强度 low、medium、high。避免把普通随机波动当成持续影响。"
        "必须输出 JSON 对象，格式为 {\"feedback\":[{\"ticker\":\"NVDA\","
        "\"add\":false,\"strength\":\"low\",\"reason\":\"简短依据\"}]}。"
        "输入中的每只股票恰好出现一次。不得使用尚未公布的数据。"
    ),
    "feedback_relative": (
        "你是股票相对收益误差分析员。只根据已经结算的预测、实际价格变化和当前公开摘要，"
        "判断是否需要记录一个持续衰减的误差反馈因子。输入 error 已扣除十只股票的平均预测误差，"
        "代表个股相对共同市场变化的误差，反馈方向由 error 确定。"
        "你只决定是否加入以及强度 low、medium、high。避免把普通随机波动当成持续影响。"
        "必须输出 JSON 对象，格式为 {\"feedback\":[{\"ticker\":\"NVDA\","
        "\"add\":false,\"strength\":\"low\",\"reason\":\"简短依据\"}]}。"
        "输入中的每只股票恰好出现一次。不得使用尚未公布的数据。"
    ),
}
EVENT_LEVEL = {"unchanged": 0.0, "small_up": 0.75, "large_up": 2.0, "small_down": -0.75, "large_down": -2.0}
EVENT_SPEED_HOURS = {"fast": 1.0, "medium": 3.0, "slow": 8.0}
EVENT_TIME_MULTIPLIER = 0.5
FEEDBACK_STRENGTH = {"low": 0.25, "medium": 0.5, "high": 1.0}
TRADE_MARGIN = 0.0005
RELATION_MIN_CORRELATION = 0.45
RELATION_Z_THRESHOLD = 2.0
FEEDBACK_ERROR_VOLATILITY = 2.0
FEEDBACK_BASES = ("geometric", "arithmetic", "relative")
QUADRATURE_NODES, QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(16)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def api_json(client, cache_dir, index, role, payload, model, thinking, reasoning_effort):
    prompt = (FLASH_FEEDBACK_PROMPTS[role]
              if model == "deepseek-flash" and thinking == "disabled" and role in FLASH_FEEDBACK_PROMPTS
              else PROMPTS[role])
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]
    request = {"model": model, "messages": messages}
    if thinking == "enabled":
        request.update({"thinking": thinking, "reasoning_effort": reasoning_effort})
    request_hash = digest(request)
    path = cache_dir / f"round_{index + 1:02d}_{role}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["request_sha256"] == request_hash:
            return saved["parsed"], saved["usage"]
        path = cache_dir / f"round_{index + 1:02d}_{role}_{request_hash[:16]}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            assert saved["request_sha256"] == request_hash, path
            return saved["parsed"], saved["usage"]
    if client is None:
        if "DEEPSEEK_API_KEY" not in os.environ:
            raise FileNotFoundError(f"Cached response missing for round {index + 1}, {role}, request_sha256={request_hash}")
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com", max_retries=0, timeout=300)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=(16000 if role == "event" else 10000) if thinking == "enabled" else
                   (3200 if role == "event" else 1600),
        extra_body={"thinking": {"type": thinking}},
        **({"reasoning_effort": reasoning_effort} if thinking == "enabled" else {}),
    )
    choice = response.choices[0]
    assert choice.finish_reason == "stop" and choice.message.content
    parsed = json.loads(choice.message.content)
    assert isinstance(parsed, dict)
    usage = response.usage.model_dump() if response.usage else None
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "request_sha256": request_hash,
        "response_id": response.id,
        "response_model": response.model,
        "thinking": thinking,
        "reasoning_effort": reasoning_effort if thinking == "enabled" else None,
        "system_fingerprint": response.system_fingerprint,
        "parsed": parsed,
        "usage": usage,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return parsed, usage


def validate_rows(value, key, tickers):
    rows = value[key]
    assert isinstance(rows, list)
    if key == "events":
        assert len(tickers) <= len(rows) <= 3 * len(tickers)
        assert set(row["ticker"] for row in rows) == set(tickers)
        order = [tickers.index(row["ticker"]) for row in rows]
        assert order == sorted(order)
        for ticker in tickers:
            group = [row for row in rows if row["ticker"] == ticker]
            assert 1 <= len(group) <= 3
    else:
        assert len(rows) == len(tickers), (key, list(tickers), [row["ticker"] for row in rows])
        assert [row["ticker"] for row in rows] == list(tickers), (key, list(tickers), [row["ticker"] for row in rows])
    for row in rows:
        assert isinstance(row["reason"], str)
        if key == "events":
            assert row["level"] in EVENT_LEVEL and row["speed"] in EVENT_SPEED_HOURS
        else:
            assert type(row["add"]) is bool and row["strength"] in FEEDBACK_STRENGTH
    return rows


def historical_parameters(hourly_cache):
    schedule, prices, _, _, dates = prepare(hourly_cache)
    returns = np.log(prices[1:] / prices[:-1])
    durations = np.asarray([
        (datetime.fromisoformat(schedule[index]["timestamp"]) -
         datetime.fromisoformat(schedule[index - 1]["timestamp"])).total_seconds() / 3600
        for index in range(1, len(schedule))
    ])
    same_day = np.asarray([
        schedule[index]["timestamp"][:10] == schedule[index - 1]["timestamp"][:10]
        for index in range(1, len(schedule))
    ])
    assert np.all(durations[same_day] > 0)
    returns = returns[same_day] / np.sqrt(durations[same_day, None])
    assert len(dates) >= 400 and len(returns) >= 1000
    covariance = LedoitWolf().fit(returns).covariance_
    vol = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(vol, vol)
    beta = covariance / np.diag(covariance)[None, :]
    residual_vol = np.sqrt(np.maximum(
        np.diag(covariance)[:, None] - covariance * beta, 0
    ))
    centering = np.eye(len(TICKERS)) - np.ones((len(TICKERS), len(TICKERS))) / len(TICKERS)
    relative_vol = np.sqrt(np.diag(centering @ covariance @ centering.T))
    return {
        "volatility": vol,
        "correlation": correlation,
        "beta": beta,
        "residual_volatility": residual_vol,
        "relative_volatility": relative_vol,
        "source_sha256": {ticker: hashlib.sha256((hourly_cache / f"{ticker}.json").read_bytes()).hexdigest() for ticker in TICKERS},
        "sample_count": len(returns),
        "first_date": dates[0], "last_date": dates[-1], "trading_dates": len(dates),
    }


def hours(start, end):
    value = (end - start).total_seconds() / 3600
    assert value >= 0
    return value


@lru_cache(maxsize=4096)
def interval_weight(kind, created_at, scale, entry_at, exit_at):
    start = datetime.fromisoformat(created_at)
    entry = hours(start, entry_at)
    exit_time = hours(start, exit_at)
    assert entry < exit_time
    assert scale > 0
    if kind == "event":
        return np.exp(-((entry / scale) ** 0.75)) - np.exp(-((exit_time / scale) ** 0.75))
    if kind == "relation":
        return np.exp(-entry / scale) - np.exp(-exit_time / scale)
    assert kind == "feedback"
    return (1 + entry / scale) ** -1.5 - (1 + exit_time / scale) ** -1.5


def factor_return(factor, entry_at, exit_at):
    return factor["amplitude"] * interval_weight(factor["kind"], factor["created_at"],
                                                factor["time_scale_hours"], entry_at, exit_at)


def geometric_return(positive, negative):
    assert positive >= 0 and negative >= 0 and negative < 1
    midpoint = (positive - negative) / 2
    half_width = (positive + negative) / 2
    samples = midpoint + half_width * QUADRATURE_NODES
    assert np.all(samples > -1)
    return float(np.expm1(np.dot(QUADRATURE_WEIGHTS, np.log1p(samples)) / 2))


def forecast_components(factors, entry_at, exit_at):
    positives = np.zeros(len(TICKERS))
    negatives = np.zeros(len(TICKERS))
    for factor in factors:
        if factor["amplitude"] == 0:
            continue
        value = factor_return(factor, entry_at, exit_at)
        ticker_index = TICKERS.index(factor["ticker"])
        if value >= 0:
            positives[ticker_index] += value
        else:
            negatives[ticker_index] -= value
    assert np.max(negatives) < 1
    geometric = np.asarray([geometric_return(a, b) for a, b in zip(positives, negatives)])
    arithmetic = (positives - negatives) / 2
    return geometric, arithmetic


def forecast(factors, entry_at, exit_at):
    return forecast_components(factors, entry_at, exit_at)[0]


def relation_factors(prices, schedule, index, parameters, created_at):
    current = recent_returns(prices, schedule, index)
    if current is None:
        return []
    interval_hours = hours(datetime.fromisoformat(schedule[index - 1]["timestamp"]),
                           datetime.fromisoformat(schedule[index]["timestamp"]))
    assert interval_hours > 0
    correlation = parameters["correlation"]
    beta = parameters["beta"]
    residual_vol = parameters["residual_volatility"]
    factors = []
    for first in range(len(TICKERS)):
        for second in range(first + 1, len(TICKERS)):
            strength = float(correlation[first, second])
            if strength < RELATION_MIN_CORRELATION:
                continue
            residual = float(current[first] - beta[first, second] * current[second])
            scale = float(residual_vol[first, second] * np.sqrt(interval_hours))
            if scale <= 0 or abs(residual) <= RELATION_Z_THRESHOLD * scale:
                continue
            excess = min(abs(residual) - RELATION_Z_THRESHOLD * scale, 3 * scale)
            amplitude = excess * strength
            time_scale = 1.0 + 4.0 * (1.0 - strength)
            for ticker, direction in ((TICKERS[first], -np.sign(residual)), (TICKERS[second], np.sign(residual))):
                factors.append({
                    "kind": "relation", "ticker": ticker, "created_at": created_at.isoformat(),
                    "amplitude": float(direction * amplitude / 2),
                    "time_scale_hours": time_scale,
                    "pair": [TICKERS[first], TICKERS[second]],
                    "correlation": strength, "residual_z": residual / scale,
                })
    return factors


def choose_action(geometric, arithmetic):
    long_index = int(np.argmax(geometric))
    short_index = int(np.argmin(geometric))
    geometric_spread = float(geometric[long_index] - geometric[short_index])
    predicted_net = float((arithmetic[long_index] - arithmetic[short_index]) / 2 - TRADE_COST)
    action = "LS" if long_index != short_index and predicted_net > TRADE_MARGIN else "F"
    return {"action": action, "stock1": TICKERS[long_index] if action == "LS" else None,
            "stock2": TICKERS[short_index] if action == "LS" else None,
            "predicted_spread_bps": geometric_spread * 10000,
            "predicted_net_bps": predicted_net * 10000}


def news_rows(root, manifest, index):
    release = manifest["published_rounds"][index]
    folder = root / "data" / release["folder_url"].split("/")[-1]
    news = pd.read_parquet(folder / "news.parquet").set_index("ticker")
    return [{"ticker": ticker, "summary": str(news.loc[ticker, "summary"]),
             "news_count": int(news.loc[ticker, "news_count"])} for ticker in TICKERS]


def feedback_candidates(settled, schedule, parameters, basis):
    assert basis in FEEDBACK_BASES
    candidates = []
    prediction_key = "predicted_stock_returns" if basis == "geometric" else "predicted_stock_arithmetic_returns"
    for record in settled:
        predicted = np.asarray(record[prediction_key])
        actual = np.asarray(record["actual_stock_returns"])
        errors = actual - predicted
        common_error = float(np.mean(errors)) if basis == "relative" else 0.0
        errors = errors - common_error
        interval_hours = hours(datetime.fromisoformat(schedule[record["entry_index"]]["timestamp"]),
                               datetime.fromisoformat(schedule[record["exit_index"]]["timestamp"]))
        volatility_key = "relative_volatility" if basis == "relative" else "volatility"
        volatility = parameters[volatility_key] * np.sqrt(interval_hours)
        for ticker_index, ticker in enumerate(TICKERS):
            if abs(errors[ticker_index]) <= FEEDBACK_ERROR_VOLATILITY * volatility[ticker_index]:
                continue
            candidate = {"ticker": ticker, "source_round": record["round"],
                         "predicted_return": float(predicted[ticker_index]),
                         "actual_return": float(actual[ticker_index]),
                         "error": float(errors[ticker_index])}
            if basis == "relative":
                candidate["common_market_error"] = common_error
                candidate["error_definition"] = "actual_return - predicted_return - common_market_error"
            candidates.append(candidate)
    return candidates


def run(root, state_path, hourly_cache, call_cache, limit, feedback_basis="geometric",
        execution_policy=DEFAULT_EXECUTION_POLICY, model=MODEL, thinking="disabled",
        reasoning_effort="high", verbose=True):
    state, schedule, prices, _, eligible = read_data(root, state_path)
    manifest = json.loads((root / "releases.json").read_text(encoding="utf-8"))
    assert 1 <= limit <= len(schedule)
    parameters = historical_parameters(hourly_cache)
    assert parameters["last_date"] < schedule[0]["timestamp"][:10]
    client = None
    eligible_by_index = {index: (entry, exit_index) for index, entry, exit_index in eligible}
    pending = []
    factors = []
    records = []
    training = []
    for index in range(limit):
        release = manifest["published_rounds"][index]
        now = datetime.fromisoformat(release["published_at"])
        assert now <= datetime.fromisoformat(schedule[index]["action_due"])
        current_news = news_rows(root, manifest, index)
        new_factors = []
        settled = []
        for record in pending[:]:
            if record["exit_index"] != index:
                continue
            assert datetime.fromisoformat(record["published_at"]) < now
            entry_prices = prices[record["entry_index"]]
            actual = prices[index] / entry_prices - 1
            pair = record["pair"]
            gross = pair_return(pair, entry_prices, prices[index])
            record["gross_return"] = gross
            record["net_return"] = gross - (TRADE_COST if pair is not None else 0)
            record["actual_stock_returns"] = actual.tolist()
            settled.append(record)
            training.append({"source_round": record["round"], "exit_index": index,
                             "features": np.asarray(record["policy_features"]),
                             "target": (actual - np.mean(actual)) / 0.01})
            pending.remove(record)
        feedback_inputs = feedback_candidates(settled, schedule, parameters, feedback_basis)
        feedback_usage = None
        if feedback_inputs:
            candidates_by_ticker = {row["ticker"]: row for row in feedback_inputs}
            candidates = [candidates_by_ticker[ticker] for ticker in TICKERS if ticker in candidates_by_ticker]
            payload = {
                "round": index + 1, "candidates": candidates, "current_summaries": current_news,
            }
            if feedback_basis == "relative":
                payload["error_basis"] = "共同市场误差已被扣除；输入 error 决定反馈方向，只评价股票相对市场的持续影响。"
                payload["current_summaries"] = [row for row in current_news if row["ticker"] in candidates_by_ticker]
            role = "feedback_relative" if feedback_basis == "relative" else "feedback"
            feedback_result, feedback_usage = api_json(
                client, call_cache, index, role, payload, model, thinking, reasoning_effort)
            feedback_rows = validate_rows(feedback_result, "feedback", [row["ticker"] for row in candidates])
            for row in feedback_rows:
                if not row["add"]:
                    continue
                candidate = candidates_by_ticker[row["ticker"]]
                amplitude = candidate["error"] * FEEDBACK_STRENGTH[row["strength"]]
                new_factors.append({"kind": "feedback", "ticker": row["ticker"],
                                    "created_at": now.isoformat(), "amplitude": amplitude,
                                    "time_scale_hours": 3.0, "source_round": candidate["source_round"],
                                    "strength": row["strength"]})
        else:
            feedback_rows = []
        relation_new = relation_factors(prices[:index + 1], schedule[:index + 1], index, parameters, now)
        new_factors.extend(relation_new)
        event_result, event_usage = api_json(client, call_cache, index, "event", {
            "round": index + 1, "summaries": current_news,
        }, model, thinking, reasoning_effort)
        event_rows = validate_rows(event_result, "events", TICKERS)
        event_numbers = {ticker: 0 for ticker in TICKERS}
        for row in event_rows:
            event_numbers[row["ticker"]] += 1
            amplitude = EVENT_LEVEL[row["level"]] * parameters["volatility"][TICKERS.index(row["ticker"])]
            new_factors.append({"kind": "event", "ticker": row["ticker"],
                                "created_at": now.isoformat(), "amplitude": float(amplitude),
                                "time_scale_hours": EVENT_SPEED_HOURS[row["speed"]] * EVENT_TIME_MULTIPLIER,
                                "level": row["level"], "speed": row["speed"],
                                "event_number": event_numbers[row["ticker"]],
                                "reason": row["reason"], "source_round": index + 1})
        for factor in new_factors:
            factor["id"] = len(factors) + 1
            factors.append(factor)
        entry_at = datetime.fromisoformat(schedule[index]["entry_at"])
        exit_at = datetime.fromisoformat(schedule[index]["exit_at"])
        predicted, predicted_arithmetic = forecast_components(factors, entry_at, exit_at)
        decision, execution_details = execution_decision(
            execution_policy, predicted, predicted_arithmetic, prices[:index + 1],
            schedule[:index + 1], index, training)
        pair = None if decision["action"] == "F" else (TICKERS.index(decision["stock1"]), TICKERS.index(decision["stock2"]))
        record = {"round": index + 1, "round_id": release["round_id"], "published_at": release["published_at"],
                  "action_due": schedule[index]["action_due"], "decision": decision, "pair": pair,
                  "predicted_stock_returns": predicted.tolist(),
                  "predicted_stock_arithmetic_returns": predicted_arithmetic.tolist(),
                  "execution_details": execution_details,
                  "policy_features": policy_features(predicted_arithmetic, prices[:index + 1],
                                                       schedule[:index + 1], index).tolist(),
                  "event_analysis": event_rows,
                  "feedback_candidates": feedback_inputs,
                  "feedback_analysis": feedback_rows, "new_factors": new_factors,
                  "active_factor_count": len(factors),
                  "usage": {"event": event_usage, "feedback": feedback_usage}}
        if index in eligible_by_index:
            record["entry_index"], record["exit_index"] = eligible_by_index[index]
            pending.append(record)
        records.append(record)
        if verbose:
            print(f"round={index + 1:02d} action={decision['action']} stock1={decision['stock1']} stock2={decision['stock2']} factors={len(factors)}", flush=True)
    evaluable = [record for record in records if "net_return" in record]
    return {"model": model, "thinking": thinking,
            "reasoning_effort": reasoning_effort if thinking == "enabled" else None,
            "rounds_processed": limit, "rounds_evaluable": len(evaluable),
            "manifest_sha256": hashlib.sha256((root / "releases.json").read_bytes()).hexdigest(),
            "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
            "historical_parameters": {"sample_count": parameters["sample_count"],
                                      "first_date": parameters["first_date"],
                                      "last_date": parameters["last_date"],
                                      "trading_dates": parameters["trading_dates"],
                                      "source_sha256": parameters["source_sha256"],
                                      "volatility": parameters["volatility"].tolist(),
                                      "relative_volatility": parameters["relative_volatility"].tolist(),
                                      "correlation": parameters["correlation"].tolist()},
            "factor_parameters": {"event_levels": EVENT_LEVEL, "event_speed_hours": EVENT_SPEED_HOURS,
                                  "event_time_multiplier": EVENT_TIME_MULTIPLIER,
                                  "feedback_strength": FEEDBACK_STRENGTH,
                                  "feedback_basis": feedback_basis,
                                  "execution_policy": execution_policy,
                                  "execution_config": POLICIES[execution_policy],
                                  "trade_cost": TRADE_COST, "trade_margin": POLICIES[execution_policy]["margin"],
                                  "relation_min_correlation": RELATION_MIN_CORRELATION,
                                  "relation_z_threshold": RELATION_Z_THRESHOLD,
                                  "feedback_error_volatility": FEEDBACK_ERROR_VOLATILITY},
            "factor_metrics": metrics([record["net_return"] for record in evaluable]) if len(evaluable) >= 2 else None,
            "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("research_import/task2_dataset"))
    parser.add_argument("--state", type=Path, default=Path("research_import/task2_live_state.json"))
    parser.add_argument("--hourly-cache", type=Path, default=Path("research_import/task2_yahoo_1h"))
    parser.add_argument("--call-cache", type=Path, default=Path("outputs/task2_factor_calls"))
    parser.add_argument("--output", type=Path, default=Path("outputs/task2_factor_agent.json"))
    parser.add_argument("--limit", type=int, default=41)
    parser.add_argument("--feedback-basis", choices=FEEDBACK_BASES, default="geometric")
    parser.add_argument("--execution-policy", choices=POLICIES, default=DEFAULT_EXECUTION_POLICY)
    parser.add_argument("--model", choices=("deepseek-flash", "deepseek-v4-pro"), default=MODEL)
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled")
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), default="high")
    args = parser.parse_args()
    report = run(args.data, args.state, args.hourly_cache, args.call_cache, args.limit,
                 args.feedback_basis, args.execution_policy, args.model, args.thinking,
                 args.reasoning_effort)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Report:", args.output)
    print("Metrics:", report["factor_metrics"])


if __name__ == "__main__":
    main()
