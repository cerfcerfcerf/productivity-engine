from productivity_engine.evaluator import compare


def test_compare_deltas():
    baseline = {
        "expected_completion_rate": 0.5,
        "expected_delay_hours": 10.0,
        "expected_ignore_rate": 0.2,
    }
    adaptive = {
        "expected_completion_rate": 0.6,
        "expected_delay_hours": 8.0,
        "expected_ignore_rate": 0.1,
    }
    result = compare(baseline, adaptive)
    assert round(result["completion_improvement_pct"], 2) == 20.0
    assert round(result["delay_reduction_pct"], 2) == 20.0
    assert round(result["ignore_reduction_pct"], 2) == 50.0
