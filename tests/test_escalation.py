from productivity_engine.escalation import deadline_schedule


def test_escalation_rules():
    assert deadline_schedule(100) == [24]
    assert deadline_schedule(48) == [12, 24]
    assert deadline_schedule(12) == [3.0, 6.0, 9.0, 12.0]
    assert deadline_schedule(4) == [1.0, 2.0, 3.0, 4.0]
