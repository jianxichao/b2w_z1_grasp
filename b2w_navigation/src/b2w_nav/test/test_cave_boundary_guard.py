from b2w_nav.cave_boundary_guard_node import gate_midpoint, overlay_segment_wall


def test_gate_midpoint():
    assert gate_midpoint((1.0, 2.0), (3.0, 6.0)) == (2.0, 4.0)


def test_wall_marks_segment_but_not_infinite_line():
    result = overlay_segment_wall(
        [0] * 100, 10, 10, 1.0, 0.0, 0.0, (2.5, 5.5), (6.5, 5.5), 0.1
    )
    assert result[5 * 10 + 2] == 100
    assert result[5 * 10 + 6] == 100
    assert result[5 * 10 + 0] == 0
    assert result[5 * 10 + 9] == 0
