import numpy as np
from sklearn.linear_model import Ridge

from task2_experiment import TICKERS, recent_returns


TRADE_COST = 0.001
DEFAULT_EXECUTION_POLICY = "momentum_weight_1_5"


POLICIES = {
    "factor_margin_5": {"mode": "factor", "margin": 0.0005},
    "factor_margin_0": {"mode": "factor", "margin": 0.0},
    "factor_margin_2_5": {"mode": "factor", "margin": 0.00025},
    "factor_margin_10": {"mode": "factor", "margin": 0.001},
    "momentum_weight_0_25": {"mode": "momentum", "weight": 0.25, "margin": 0.0005},
    "momentum_weight_0_5": {"mode": "momentum", "weight": 0.5, "margin": 0.0005},
    "momentum_weight_1": {"mode": "momentum", "weight": 1.0, "margin": 0.0005},
    "momentum_weight_1_5": {"mode": "momentum", "weight": 1.5, "margin": 0.0005},
    "momentum_weight_2": {"mode": "momentum", "weight": 2.0, "margin": 0.0005},
    "momentum_margin_0": {"mode": "momentum", "weight": 1.0, "margin": 0.0},
    "momentum_margin_2_5": {"mode": "momentum", "weight": 1.0, "margin": 0.00025},
    "momentum_margin_10": {"mode": "momentum", "weight": 1.0, "margin": 0.001},
    "online_ridge_1": {"mode": "online", "alpha": 1.0, "margin": 0.0005},
    "online_ridge_10": {"mode": "online", "alpha": 10.0, "margin": 0.0005},
    "online_ridge_100": {"mode": "online", "alpha": 100.0, "margin": 0.0005},
}


def price_feature(prices, schedule, index):
    recent = recent_returns(prices, schedule, index)
    if recent is None:
        return np.zeros(len(TICKERS))
    return recent - np.mean(recent)


def policy_features(arithmetic, prices, schedule, index):
    centered = np.asarray(arithmetic) - np.mean(arithmetic)
    return np.column_stack((centered, price_feature(prices, schedule, index))) / 0.01


def execution_decision(name, geometric, arithmetic, prices, schedule, index, training):
    config = POLICIES[name]
    geometric = np.asarray(geometric)
    arithmetic = np.asarray(arithmetic)
    ranking = geometric.copy()
    expected = arithmetic.copy()
    coefficients = [1.0, 0.0]
    if config["mode"] == "momentum":
        addition = config["weight"] * price_feature(prices, schedule, index)
        ranking += addition
        expected += addition
        coefficients[1] = config["weight"]
    elif config["mode"] == "online":
        prior = np.asarray(coefficients)
        if training:
            x = np.concatenate([row["features"] for row in training])
            y = np.concatenate([row["target"] for row in training])
            model = Ridge(alpha=config["alpha"], fit_intercept=False)
            model.fit(x, y - x @ prior)
            coefficients = (prior + model.coef_).tolist()
        features = policy_features(arithmetic, prices, schedule, index)
        expected = features @ np.asarray(coefficients) * 0.01
        ranking = expected
    else:
        assert config["mode"] == "factor"
    long_index = int(np.argmax(ranking))
    short_index = int(np.argmin(ranking))
    gross = float((expected[long_index] - expected[short_index]) / 2)
    active = long_index != short_index and gross - TRADE_COST > config["margin"]
    decision = {
        "action": "LS" if active else "F",
        "stock1": TICKERS[long_index] if active else None,
        "stock2": TICKERS[short_index] if active else None,
        "predicted_spread_bps": float((ranking[long_index] - ranking[short_index]) * 10000),
        "predicted_net_bps": float((gross - TRADE_COST) * 10000),
    }
    return decision, {"coefficients": coefficients, "settled_training_rounds": len(training),
                      "expected_stock_returns": expected.tolist()}
