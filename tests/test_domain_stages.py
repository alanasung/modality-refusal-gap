from vlmrefusal.stages import STAGES
def test_names():
    assert "render" in STAGES and "utility" in STAGES
def test_no_ni():
    import inspect
    import vlmrefusal.stages as s
    assert "NotImplementedError" not in inspect.getsource(s)
def test_callable():
    assert all(callable(v) for v in STAGES.values())
