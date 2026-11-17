from datetime import datetime, timedelta, timezone

import numpy as np

from task2_factor_agent import factor_return, geometric_return


def test_geometric_return_matches_uniform_log_integral():
    positive = 0.025
    negative = 0.012

    def antiderivative(value):
        return (1 + value) * np.log1p(value) - value

    expected = np.expm1((antiderivative(positive) - antiderivative(-negative)) / (positive + negative))
    assert np.isclose(geometric_return(positive, negative), expected, rtol=0, atol=1e-14)
    assert geometric_return(0, 0) == 0


def test_factor_kernels_keep_direction_and_decay():
    created = datetime(2026, 10, 14, 13, 30, tzinfo=timezone.utc)
    for kind in ("event", "relation", "feedback"):
        factor = {"kind": kind, "created_at": created.isoformat(),
                  "amplitude": 0.02, "time_scale_hours": 3.0}
        early = factor_return(factor, created + timedelta(hours=1), created + timedelta(hours=2))
        late = factor_return(factor, created + timedelta(hours=5), created + timedelta(hours=6))
        assert 0 < late < early
        factor["amplitude"] *= -1
        assert np.isclose(factor_return(factor, created + timedelta(hours=1), created + timedelta(hours=2)), -early)
