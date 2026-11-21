"""新增测试：边界、错误可区分性、空间加速性能、并发安全、顺序无关。

不修改既有 tests/test_core.py。运行：
    python -m pytest tests -q -k hit
"""

import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import hittest
from hittest import (
    DegenerateRectError,
    InvalidRadiusError,
    NonFiniteError,
)


def _h():
    return hittest.HitTest()


# ---------- 基础几何：相切算碰撞（<= 语义） ----------

def test_hit_circle_circle_tangent_is_hit():
    h = _h()
    assert h.circle_circle(((0, 0), 1), ((2, 0), 1)) is True
    # 相距只比 2 大一个最小浮点步进 -> 不碰撞
    assert h.circle_circle(((0, 0), 1), ((2 + 1e-12, 0), 1)) is False


def test_hit_circle_rect_nearest_point_all_regions():
    """最近点法：圆心在矩形外四个方向 + 角部。"""
    h = _h()
    rect = (0, 0, 2, 2)
    # 右侧最近点 (2,1)，距离 1，相切
    assert h.circle_rect(((3, 1), 1), rect) is True
    assert h.circle_rect(((3 + 1e-12, 1), 1), rect) is False
    # 上方最近点 (1,2)
    assert h.circle_rect(((1, 3), 1), rect) is True
    # 角部最近点 (2,2)，圆心 (2+d,2+d)，相切距离 sqrt(2)
    d = math.sqrt(0.5)
    assert h.circle_rect(((2 + d, 2 + d), 1), rect) is True
    assert h.circle_rect(((2 + d + 1e-12, 2 + d), 1), rect) is False
    # 圆心在矩形内部必碰撞（最近点就是圆心，距离 0）
    assert h.circle_rect(((1, 1), 0.1), rect) is True
    # 零半径虽非法，但半径很小的圆在矩形外
    assert h.circle_rect(((5, 5), 0.5), rect) is False


def test_hit_rect_rect_sat_tangent_hit():
    h = _h()
    # x 轴相切（投影端点重合）算碰撞
    assert h.rect_rect((0, 0, 2, 2), (2, 0, 2, 2)) is True
    # y 轴严格分离
    assert h.rect_rect((0, 0, 2, 2), (0, 2 + 1e-12, 2, 2)) is False
    # x、y 都分离
    assert h.rect_rect((0, 0, 2, 2), (3, 3, 2, 2)) is False
    # 部分重叠
    assert h.rect_rect((0, 0, 2, 2), (1, 1, 2, 2)) is True
    # 包含
    assert h.rect_rect((0, 0, 10, 10), (1, 1, 2, 2)) is True


def test_hit_return_type_is_bool():
    h = _h()
    assert isinstance(h.circle_circle(((0, 0), 1), ((1, 0), 1)), bool)
    assert isinstance(h.circle_rect(((0, 0), 1), (0, 0, 2, 2)), bool)
    assert isinstance(h.rect_rect((0, 0, 1, 1), (1, 1, 1, 1)), bool)


# ---------- 顺序无关性 ----------

def test_hit_order_independent():
    h = _h()
    c1, c2 = ((0, 0), 1), ((1.5, 0), 1)
    assert h.circle_circle(c1, c2) == h.circle_circle(c2, c1)
    c, r = ((3, 1), 1), (0, 0, 2, 2)
    assert h.circle_rect(c, r) == h.circle_rect(c, r)  # 单方向固定
    r1, r2 = (0, 0, 2, 2), (1.5, 1.5, 2, 2)
    assert h.rect_rect(r1, r2) == h.rect_rect(r2, r1)
    assert h.rect_rect(r1, r2) is True


def test_hit_pairwise_order_independent():
    rng = random.Random(42)
    objs = []
    for _ in range(300):
        if rng.random() < 0.5:
            objs.append(((rng.uniform(0, 100), rng.uniform(0, 100)),
                         rng.uniform(0.5, 3)))
        else:
            objs.append((rng.uniform(0, 100), rng.uniform(0, 100),
                         rng.uniform(0.5, 3), rng.uniform(0.5, 3)))
    h = _h()
    base = h.pairwise(objs)
    shuffled = objs[:]
    rng.shuffle(shuffled)
    reordered = h.pairwise(shuffled)
    # 下标对随排列变化，但映射回物体后碰撞集合必须一致
    def to_set(pairs, seq):
        return {tuple(sorted((id(seq[i]), id(seq[j])))) for i, j in pairs}
    assert to_set(base, objs) == to_set(reordered, shuffled)
    # 直接结果满足 i < j 且有序
    assert all(i < j for i, j in base)
    assert base == sorted(base)


def test_hit_pairwise_matches_bruteforce():
    """空间加速结果与纯暴力精确判定逐一对照（混合三种组合）。"""
    rng = random.Random(7)
    objs = []
    for k in range(120):
        x, y = rng.uniform(0, 20), rng.uniform(0, 20)
        if k % 3 == 0:
            objs.append(((x, y), rng.uniform(0.5, 2.0)))
        elif k % 3 == 1:
            objs.append((x, y, rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0)))
        else:
            objs.append(((x, y), rng.uniform(0.2, 1.0)))
    h = _h()
    got = set(h.pairwise(objs))
    expected = set()
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            if isinstance(a[0], (tuple, list)):
                if isinstance(b[0], (tuple, list)):
                    ok = h.circle_circle(a, b)
                else:
                    ok = h.circle_rect(a, b)
            else:
                if isinstance(b[0], (tuple, list)):
                    ok = h.circle_rect(b, a)
                else:
                    ok = h.rect_rect(a, b)
            if ok:
                expected.add((i, j))
    assert got == expected


# ---------- 可区分的错误 ----------

def test_hit_nonfinite_inputs_distinguishable():
    h = _h()
    bad_floats = (float("nan"), float("inf"), float("-inf"))
    for v in bad_floats:
        with pytest.raises(NonFiniteError):
            h.circle_circle(((v, 0), 1), ((0, 0), 1))
        with pytest.raises(NonFiniteError):
            h.circle_circle(((0, 0), 1), ((0, 0), v))
        with pytest.raises(NonFiniteError):
            h.circle_rect(((0, 0), 1), (v, 0, 1, 1))
        with pytest.raises(NonFiniteError):
            h.rect_rect((0, 0, 1, 1), (0, v, 1, 1))


def test_hit_invalid_radius_distinguishable():
    h = _h()
    for bad in (0, -1, -1e9):
        with pytest.raises(InvalidRadiusError):
            h.circle_circle(((0, 0), bad), ((0, 0), 1))
        with pytest.raises(InvalidRadiusError):
            h.circle_rect(((0, 0), bad), (0, 0, 1, 1))
    # 仍是 ValueError，兼容既有测试的捕获方式
    with pytest.raises(ValueError):
        h.circle_circle(((0, 0), -1), ((0, 0), 1))


def test_hit_degenerate_rect_distinguishable():
    h = _h()
    for bad_rect in ((0, 0, 0, 1), (0, 0, 1, 0), (0, 0, -1, 1),
                     (0, 0, 1, -1e-9)):
        with pytest.raises(DegenerateRectError):
            h.rect_rect(bad_rect, (0, 0, 1, 1))
        with pytest.raises(DegenerateRectError):
            h.circle_rect(((0, 0), 1), bad_rect)
    # 仍是 ValueError
    with pytest.raises(ValueError):
        h.rect_rect((0, 0, 0, 1), (0, 0, 1, 1))


def test_hit_error_classes_are_distinct():
    assert issubclass(InvalidRadiusError, ValueError)
    assert issubclass(DegenerateRectError, ValueError)
    assert issubclass(NonFiniteError, ValueError)
    assert InvalidRadiusError is not DegenerateRectError
    assert NonFiniteError is not InvalidRadiusError


def test_hit_pairwise_grid_same_cell_non_overlapping_rects():
    """同处一个网格单元但包围盒不相交的矩形对，不能因同格误报碰撞。

    场景跨度足够大（> 4 倍最大边长），确保走均匀网格主路径而非
    SAP 回退；四个矩形挤在同一格附近，只有 r0/r3 相切。
    """
    h = _h()
    objs = [
        (0.0, 0.0, 1.0, 1.0),      # 原点，x 区间 [0,1]
        (0.0, 2.5, 1.0, 1.0),      # 正上方，y 区间 [2.5,3.5]，严格分离
        (3.0, 0.0, 1.0, 1.0),      # 正右方，x 区间 [3,4]，严格分离
        (1.0, 0.0, 1.0, 1.0),      # x 区间 [1,2]，与 r0 在 x=1 相切
        (40.0, 40.0, 1.0, 1.0),    # 远处矩形，撑大跨度启用网格
    ]
    # 同格但 x/y 严格分离的 (0,1)、(0,2) 等不得误报；只有 (0,3) 相切碰撞
    assert h.pairwise(objs) == [(0, 3)]
    shuffled = [objs[3], objs[1], objs[0], objs[2], objs[4]]
    assert h.pairwise(shuffled) == [(0, 2)]


def test_hit_pairwise_rejects_invalid_input():
    h = _h()
    with pytest.raises(InvalidRadiusError):
        h.pairwise([((0, 0), 1), ((1, 1), -2)])
    with pytest.raises(DegenerateRectError):
        h.pairwise([(0, 0, 1, 1), (0, 0, 0, 1)])
    with pytest.raises(NonFiniteError):
        h.pairwise([((0, 0), float("inf"))])


# ---------- 极大坐标：不能因平方溢出而误判 ----------

def test_hit_huge_coordinates_no_overflow():
    h = _h()
    # 1e12 量级：坐标平方 1e24 在 double 内安全，且小偏移仍可表示，
    # 验证相切 <= 语义在大坐标下不被舍入破坏。
    big = 1e12
    assert h.circle_circle(((big, big), 1),
                           ((big + 2, big), 1)) is True
    assert h.circle_circle(((big, big), 1),
                           ((big + 2 + 1e-4, big), 1)) is False
    assert h.circle_circle(((big, big), 1),
                           ((big + 4, big), 1)) is False
    assert h.circle_rect(((big + 3, big + 1), 1),
                         (big, big, 2, 2)) is True
    assert h.rect_rect((big, big, 2, 2), (big + 2, big, 2, 2)) is True
    assert h.rect_rect((big, big, 2, 2),
                       (big + 3, big, 2, 2)) is False
    # 相切后错开 1e-3（大于该量级 ulp≈1.16e-4，可表示）-> miss
    assert h.rect_rect((big, big, 2, 2),
                       (big + 2 + 1e-3, big, 2, 2)) is False


def test_hit_extreme_scale_naive_squaring_would_overflow():
    """1e200 量级：朴素 d*d 比较会变 inf 而误判，hypot 实现必须正确。

    该量级下 double 的 ulp 约 1e184，小偏移不可表示，因此用
    显式不同的大坐标验证：距离远大于半径和时必须判 miss，
    不能因平方溢出（inf 比较恒假/恒真）而给出错误结论。
    """
    h = _h()
    huge = 1e200
    # 两圆心可表示地相距 1e196，半径和仅 2 -> 必 miss；
    # 朴素实现 (dx*dx + dy*dy) <= (r1+r2)^2 中 dx*dx 为 inf，逻辑失效。
    assert h.circle_circle(((huge, 0.0), 1.0),
                           ((huge + 1e196, 0.0), 1.0)) is False
    # 同坐标（偏移被舍入为 0）-> 必 hit，hypot 正常返回 0
    assert h.circle_circle(((huge, 0.0), 1.0),
                           ((huge + 1.0, 0.0), 1.0)) is True
    # 圆在矩形外，最近点距离 1e196 远大于半径 1 -> miss
    assert h.circle_rect(((huge + 1e196, huge), 1.0),
                         (huge, huge, 2.0, 2.0)) is False
    # 圆心坐标落在矩形边上（最近点距离 0）-> hit
    assert h.circle_rect(((huge, huge), 1.0),
                         (huge, huge, 2.0, 2.0)) is True


# ---------- 性能：10 万物体 < 500ms（空间加速，非暴力） ----------

def _make_sparse_objects(n=100_000, seed=1234):
    rng = random.Random(seed)
    objs = []
    # 散布在 0..4000 的稀疏场中，物体半径/边长 0.5~1.5，碰撞很少
    for k in range(n):
        x, y = rng.uniform(0, 4000), rng.uniform(0, 4000)
        if k % 2 == 0:
            objs.append(((x, y), rng.uniform(0.5, 1.5)))
        else:
            s = rng.uniform(0.5, 1.5)
            objs.append((x, y, s, s))
    return objs


def test_hit_pairwise_100k_under_500ms():
    objs = _make_sparse_objects()
    h = _h()
    start = time.perf_counter()
    hits = h.pairwise(objs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"\npairwise n={len(objs)} hits={len(hits)} 耗时={elapsed_ms:.1f} ms")
    assert elapsed_ms < 500.0, f"10 万物体耗时 {elapsed_ms:.1f} ms 超过 500 ms"


def test_hit_pairwise_dense_100k_under_500ms():
    """高密度场：10 万物体集中在较小区域，候选对多，加速结构仍需达标。"""
    rng = random.Random(99)
    objs = []
    for k in range(100_000):
        x, y = rng.uniform(0, 400), rng.uniform(0, 400)
        if k % 2 == 0:
            objs.append(((x, y), rng.uniform(0.5, 1.0)))
        else:
            s = rng.uniform(0.5, 1.0)
            objs.append((x, y, s, s))
    h = _h()
    start = time.perf_counter()
    hits = h.pairwise(objs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"\npairwise(dense) n=100000 hits={len(hits)} 耗时={elapsed_ms:.1f} ms")
    assert elapsed_ms < 500.0


# ---------- 并发安全 ----------

def test_hit_concurrent_calls_are_safe():
    h = _h()
    objs = _make_sparse_objects(n=5_000)
    errors = []

    def worker():
        try:
            for _ in range(3):
                h.pairwise(objs)
                h.circle_circle(((0, 0), 1), ((1.5, 0), 1))
                h.circle_rect(((0, 0), 1), (0, 0, 2, 2))
                h.rect_rect((0, 0, 2, 2), (1, 1, 2, 2))
        except Exception as exc:  # noqa: BLE001 - 并发测试需要捕获一切
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_hit_concurrent_results_consistent():
    h = _h()
    objs = _make_sparse_objects(n=8_000)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: h.pairwise(objs), range(8)))
    assert all(r == results[0] for r in results)
