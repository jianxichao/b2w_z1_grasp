from rl2_hardware.unitree_slam_client import _as_bool


def test_official_boolean_values_are_parsed_strictly():
    assert _as_bool(True)
    assert _as_bool(1)
    assert _as_bool("true")
    assert not _as_bool(False)
    assert not _as_bool(0)
    assert not _as_bool("false")
