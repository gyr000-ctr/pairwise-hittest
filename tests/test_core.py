import math

import pytest

import hittest

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _h():
    return hittest.HitTest()


def _check_cc_hit():
    return _h().circle_circle(((0, 0), 1), ((1, 0), 1)) is True


def _check_cc_miss():
    return _h().circle_circle(((0, 0), 1), ((3, 0), 1)) is False


def _check_cc_touch():
    return _h().circle_circle(((0, 0), 1), ((2, 0), 1)) is True


def _check_cr_hit():
    return _h().circle_rect(((0, 0), 1), (0, 0, 2, 2)) is True


def _check_cr_miss():
    return _h().circle_rect(((0, 0), 1), (3, 3, 2, 2)) is False


def _check_cr_touch():
    return _h().circle_rect(((2, 0), 1), (2, 0, 2, 2)) is True


def _check_rr_hit():
    return _h().rect_rect((0, 0, 2, 2), (1, 1, 2, 2)) is True


def _check_rr_miss():
    return _h().rect_rect((0, 0, 2, 2), (3, 3, 2, 2)) is False


def _check_rr_touch():
    return _h().rect_rect((0, 0, 2, 2), (2, 0, 2, 2)) is True


def _check_bad_radius():
    return _raises(lambda: _h().circle_circle(((0, 0), -1), ((0, 0), 1)))


def _check_det():
    return _h().rect_rect((0, 0, 1, 1), (0, 0, 1, 1)) == _h().rect_rect((0, 0, 1, 1), (0, 0, 1, 1))


def _check_type():
    return isinstance(_h().circle_circle(((0, 0), 1), ((0, 0), 1)), bool)


def _check_inside():
    return _h().circle_rect(((1, 1), 0.5), (0, 0, 3, 3)) is True


def _check_symmetry():
    return _h().circle_circle(((0, 0), 1), ((1, 0), 1)) == _h().circle_circle(((1, 0), 1), ((0, 0), 1))


def _check_rr_symmetry():
    return _h().rect_rect((0, 0, 2, 2), (1, 1, 2, 2)) == _h().rect_rect((1, 1, 2, 2), (0, 0, 2, 2))

CASES = [("cc_hit", "_check_cc_hit"), ("cc_miss", "_check_cc_miss"), ("cc_touch", "_check_cc_touch"), ("cr_hit", "_check_cr_hit"), ("cr_miss", "_check_cr_miss"), ("cr_touch", "_check_cr_touch"), ("rr_hit", "_check_rr_hit"), ("rr_miss", "_check_rr_miss"), ("rr_touch", "_check_rr_touch"), ("bad_radius", "_check_bad_radius"), ("det", "_check_det"), ("type", "_check_type"), ("inside", "_check_inside"), ("symmetry", "_check_symmetry"), ("rr_symmetry", "_check_rr_symmetry")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
