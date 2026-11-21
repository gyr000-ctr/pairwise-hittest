"""hittest 核心 API。

碰撞约定
--------
- 圆-圆：圆心距 <= 半径和（按距离，相切算碰撞）。
- 圆-矩形：圆心到矩形最近点的距离 <= 半径（最近点法，相切算碰撞）。
- 矩形-矩形：分离轴定理（SAT），任一轴投影严格分离才不碰撞，
  相切（投影端点重合）算碰撞。当前矩形为轴对齐矩形 (x, y, w, h)，
  两条边法向轴即 x、y 轴。

批量两两检测 :meth:`HitTest.pairwise` 采用 sweep-and-prune（排序扫描）
空间加速，平均复杂度 O(n log n)，不是纯暴力 O(n^2)。

线程安全
--------
所有方法都是无状态纯函数：HitTest 实例不读写任何可变属性，
批量检测只使用函数内局部变量，因此多线程并发调用安全。

错误可区分
----------
- 非有限输入（NaN/Inf）-> :class:`NonFiniteError`
- 非法半径（r <= 0）-> :class:`InvalidRadiusError`
- 退化矩形（w <= 0 或 h <= 0）-> :class:`DegenerateRectError`
三者均继承 ValueError，既有代码捕获 ValueError 不受影响。
"""

import math

__all__ = [
    "HitTest",
    "HitTestError",
    "NonFiniteError",
    "InvalidRadiusError",
    "DegenerateRectError",
]

_KIND_CIRCLE = 0
_KIND_RECT = 1


class HitTestError(ValueError):
    """碰撞检测输入错误基类，本身仍是 ValueError。"""


class NonFiniteError(HitTestError):
    """输入中出现 NaN / Infinity 等非有限数值。"""


class InvalidRadiusError(HitTestError):
    """圆半径非法（有限但 r <= 0；NaN/Inf 归 NonFiniteError）。"""


class DegenerateRectError(HitTestError):
    """退化矩形：宽度或高度不大于 0。"""


def _validated(value, what):
    """单字段校验：必须为有限 int/float（bool 归一为 int）。

    常见路径（float/int）只做一次 isinstance + 一次 isfinite；
    bool 与非法类型是罕见情形，放到分支末端处理。
    """
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NonFiniteError(f"{what} 不是有限值: {value!r}")
        return value
    if isinstance(value, int):
        if not math.isfinite(value):  # int 恒有限，防御性保留
            raise NonFiniteError(f"{what} 不是有限值: {value!r}")
        return value
    if isinstance(value, bool):
        return int(value)
    raise TypeError(f"{what} 必须是实数，实际类型为 {type(value).__name__}")


def _parse_circle(c):
    """解析 ((x, y), r)，返回 (x, y, r)。"""
    try:
        center, r = c
        x, y = center
    except (TypeError, ValueError) as exc:
        raise TypeError("圆的格式应为 ((x, y), r)") from exc
    x = _validated(x, "圆心 x")
    y = _validated(y, "圆心 y")
    r = _validated(r, "圆半径")
    if r <= 0:
        raise InvalidRadiusError(f"圆半径必须为正数，得到 {r!r}")
    return x, y, r


def _parse_rect(rect):
    """解析 (x, y, w, h)，返回轴对齐边界 (x0, y0, x1, y1)。"""
    try:
        x, y, w, h = rect
    except (TypeError, ValueError) as exc:
        raise TypeError("矩形的格式应为 (x, y, w, h)") from exc
    x = _validated(x, "矩形 x")
    y = _validated(y, "矩形 y")
    w = _validated(w, "矩形宽 w")
    h = _validated(h, "矩形高 h")
    if w <= 0 or h <= 0:
        raise DegenerateRectError(
            f"矩形宽高必须为正数，得到 w={w!r}, h={h!r}"
        )
    return x, y, x + w, y + h


def _parse_object(obj):
    """批量输入的单物体解析，返回 (kind, params)。

    支持两种位置写法，与三个基础接口一致：
    - 圆：((x, y), r)
    - 矩形：(x, y, w, h)
    另外兼容带标签写法 ("circle", (x, y), r) / ("rect", x, y, w, h)。
    """
    if isinstance(obj, tuple) and len(obj) > 0 and isinstance(obj[0], str):
        tag = obj[0].lower()
        if tag in ("c", "circle"):
            return _KIND_CIRCLE, _parse_circle((obj[1], obj[2]))
        if tag in ("r", "rect"):
            return _KIND_RECT, _parse_rect((obj[1], obj[2], obj[3], obj[4]))
        raise TypeError(f"未知物体标签: {obj[0]!r}")
    try:
        head = obj[0]
    except (TypeError, IndexError, KeyError) as exc:
        raise TypeError("物体应为圆 ((x, y), r) 或矩形 (x, y, w, h)") from exc
    if isinstance(head, (tuple, list)):
        return _KIND_CIRCLE, _parse_circle(obj)
    return _KIND_RECT, _parse_rect(obj)


def _rects_hit(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
    """SAT：两条边法向轴（AABB 即 x/y 轴）上的投影区间检测。

    任一轴上两区间严格分离（无重叠且不相切）即可判定不碰撞；
    端点重合属于相切，不构成分离轴。
    """
    for a0, a1, b0, b1 in (
        (ax0, ax1, bx0, bx1),
        (ay0, ay1, by0, by1),
    ):
        if a1 < b0 or b1 < a0:
            return False
    return True


def _circle_rect_hit(cx, cy, r, x0, y0, x1, y1):
    """圆心到矩形最近点（clamp）距离 <= 半径。"""
    nearest_x = x0 if cx < x0 else (x1 if cx > x1 else cx)
    nearest_y = y0 if cy < y0 else (y1 if cy > y1 else cy)
    return math.hypot(cx - nearest_x, cy - nearest_y) <= r


def _exact_hit(e, f):
    """对包围盒已重叠的两个扫描条目做精确判定（SAP 回退路径用）。

    条目布局:
    (minx, maxx, miny, maxy, kind, idx, p1, p2, p3, p4)
    圆  : p1=x, p2=y, p3=r, p4=0.0
    矩形: p1=x0, p2=y0, p3=x1, p4=y1
    """
    if e[4] == _KIND_CIRCLE:
        if f[4] == _KIND_CIRCLE:
            # 圆-圆按距离；平方比较优先，溢出边缘由 hypot 兜底，
            # 相切 <= 算碰撞。
            return _circles_hit(e[6], e[7], e[8], f[6], f[7], f[8])
        return _circle_rect_hit(e[6], e[7], e[8], f[6], f[7], f[8], f[9])
    if f[4] == _KIND_CIRCLE:
        return _circle_rect_hit(f[6], f[7], f[8], e[6], e[7], e[8], e[9])
    # 矩形-矩形：能进入精确判定说明 SAT 两轴均无分离轴。
    return True


def _circles_hit(ax, ay, ar, bx, by, br):
    """圆-圆按距离：d <= r1+r2，相切算碰撞；大坐标防平方溢出。"""
    dx = ax - bx
    dy = ay - by
    rr = ar + br
    d2 = dx * dx + dy * dy
    r2 = rr * rr
    if math.isfinite(d2) and math.isfinite(r2):
        return d2 <= r2
    # 平方溢出（约 >1e154 的坐标差）时退回不会中间溢出的 hypot
    return math.hypot(dx, dy) <= rr


def _sweep_and_prune(entries):
    """x 轴排序扫描（sweep-and-prune），网格不适用时的回退路径。"""
    entries.sort()
    n = len(entries)
    hits = []
    hit_append = hits.append

    for i in range(n):
        e = entries[i]
        xi1 = e[1]
        yi0 = e[2]
        yi1 = e[3]
        j = i + 1
        while j < n:
            f = entries[j]
            if f[0] > xi1:  # x 轴已无交集，后续更远，直接停止扫描
                break
            if f[2] <= yi1 and yi0 <= f[3]:  # y 轴包围盒过滤
                if _exact_hit(e, f):
                    a = e[5]
                    b = f[5]
                    hit_append((a, b) if a < b else (b, a))
            j += 1

    hits.sort()
    return hits


def _uniform_grid(entries):
    """均匀空间网格主路径（单格分配 + 前向邻居，无去重集合）。

    每个物体只登记进其**中心点所在格**。cell 取全体最大包围盒边长，
    因此任意两个可能碰撞的物体，中心格的列/行偏移都在 {-1,0,1} 内
    （中心距 <= cell 才可能碰撞）。对每个格子只检查：

    - 同格内的物体两两配对；
    - 与 4 个“前向”邻居格 (1,-1)、(1,0)、(1,1)、(0,1) 的物体。

    9 个相邻格偏移被划分为“同格 + 4 个前向方向”，每个无序格对
    恰好枚举一次；而物体中心格唯一，故每个物体对也恰好检测一次，
    天然无重复，无需全局 pair 集合。候选数从 O(n^2) 降到约 O(n)。
    """
    gminx = min(e[0] for e in entries)
    gmaxx = max(e[1] for e in entries)
    gminy = min(e[2] for e in entries)
    gmaxy = max(e[3] for e in entries)
    spanx = gmaxx - gminx
    spany = gmaxy - gminy
    cell = max(max(e[1] - e[0], e[3] - e[2]) for e in entries)

    # 网格退化（单点场、跨度非有限、少数巨型物体占满场景）-> 回退 SAP
    if (
        cell <= 0
        or not math.isfinite(spanx)
        or not math.isfinite(spany)
        or cell * 4 >= min(spanx, spany)
    ):
        return _sweep_and_prune(entries)

    rows = int(spany // cell) + 1
    max_cx = int(spanx // cell)
    max_cy = rows - 1
    cells = {}
    isfinite = math.isfinite
    hypot = math.hypot
    KIND_C = _KIND_CIRCLE
    inv_cell = 1.0 / cell

    # 登记：每个物体进且只进中心格。中心坐标相对 gmin 恒 >= 0，
    # 下界 clamp 仅用于抵御浮点舍入产生的 -0/极小负值。
    for e in entries:
        ccx = int(((e[0] + e[1]) * 0.5 - gminx) * inv_cell)
        ccy = int(((e[2] + e[3]) * 0.5 - gminy) * inv_cell)
        if ccx < 0:
            ccx = 0
        elif ccx > max_cx:
            ccx = max_cx
        if ccy < 0:
            ccy = 0
        elif ccy > max_cy:
            ccy = max_cy
        key = ccx * rows + ccy
        bucket = cells.get(key)
        if bucket is None:
            cells[key] = [e]
        else:
            bucket.append(e)

    hits = []
    hit_append = hits.append
    neighbor_offsets = ((1, -1), (1, 0), (1, 1), (0, 1))

    for key, bucket in cells.items():
        cx = key // rows
        cy = key - cx * rows
        blen = len(bucket)

        # 同格内两两配对
        for a in range(blen):
            e = bucket[a]
            for b in range(a + 1, blen):
                f = bucket[b]
                # AABB 粗筛：任一轴严格分离即不可能碰撞。
                if e[1] < f[0] or f[1] < e[0] or e[3] < f[2] or f[3] < e[2]:
                    continue
                ei = e[5]
                fi = f[5]
                ek = e[4]
                fk = f[4]
                if ek == KIND_C:
                    if fk == KIND_C:
                        dx = e[6] - f[6]
                        dy = e[7] - f[7]
                        rr = e[8] + f[8]
                        d2 = dx * dx + dy * dy
                        collided = (d2 <= rr * rr) if isfinite(d2) \
                            else (hypot(dx, dy) <= rr)
                    else:
                        cx_, cy_, cr_ = e[6], e[7], e[8]
                        fx0, fy0, fx1, fy1 = f[6], f[7], f[8], f[9]
                        nx = fx0 if cx_ < fx0 else (fx1 if cx_ > fx1 else cx_)
                        ny = fy0 if cy_ < fy0 else (fy1 if cy_ > fy1 else cy_)
                        dx = cx_ - nx
                        dy = cy_ - ny
                        d2 = dx * dx + dy * dy
                        collided = (d2 <= cr_ * cr_) if isfinite(d2) \
                            else (hypot(dx, dy) <= cr_)
                elif fk == KIND_C:
                    cx_, cy_, cr_ = f[6], f[7], f[8]
                    ex0, ey0, ex1, ey1 = e[6], e[7], e[8], e[9]
                    nx = ex0 if cx_ < ex0 else (ex1 if cx_ > ex1 else cx_)
                    ny = ey0 if cy_ < ey0 else (ey1 if cy_ > ey1 else cy_)
                    dx = cx_ - nx
                    dy = cy_ - ny
                    d2 = dx * dx + dy * dy
                    collided = (d2 <= cr_ * cr_) if isfinite(d2) \
                        else (hypot(dx, dy) <= cr_)
                else:
                    # 双矩形：AABB 已相交即 SAT 无分离轴，必碰撞。
                    collided = True
                if collided:
                    hit_append((ei, fi) if ei < fi else (fi, ei))

        # 与 4 个前向邻居格的全部物体配对（每个无序格对恰好一次）
        for dc, dr in neighbor_offsets:
            ncx = cx + dc
            ncy = cy + dr
            if 0 <= ncx <= max_cx and 0 <= ncy <= max_cy:
                other = cells.get(ncx * rows + ncy)
                if other is None:
                    continue
                for e in bucket:
                    for f in other:
                        if e[1] < f[0] or f[1] < e[0] \
                                or e[3] < f[2] or f[3] < e[2]:
                            continue
                        ei = e[5]
                        fi = f[5]
                        ek = e[4]
                        fk = f[4]
                        if ek == KIND_C:
                            if fk == KIND_C:
                                dx = e[6] - f[6]
                                dy = e[7] - f[7]
                                rr = e[8] + f[8]
                                d2 = dx * dx + dy * dy
                                collided = (d2 <= rr * rr) if isfinite(d2) \
                                    else (hypot(dx, dy) <= rr)
                            else:
                                cx_, cy_, cr_ = e[6], e[7], e[8]
                                fx0, fy0, fx1, fy1 = f[6], f[7], f[8], f[9]
                                nx = fx0 if cx_ < fx0 else (
                                    fx1 if cx_ > fx1 else cx_)
                                ny = fy0 if cy_ < fy0 else (
                                    fy1 if cy_ > fy1 else cy_)
                                dx = cx_ - nx
                                dy = cy_ - ny
                                d2 = dx * dx + dy * dy
                                collided = (d2 <= cr_ * cr_) if isfinite(d2) \
                                    else (hypot(dx, dy) <= cr_)
                        elif fk == KIND_C:
                            cx_, cy_, cr_ = f[6], f[7], f[8]
                            ex0, ey0, ex1, ey1 = e[6], e[7], e[8], e[9]
                            nx = ex0 if cx_ < ex0 else (
                                ex1 if cx_ > ex1 else cx_)
                            ny = ey0 if cy_ < ey0 else (
                                ey1 if cy_ > ey1 else cy_)
                            dx = cx_ - nx
                            dy = cy_ - ny
                            d2 = dx * dx + dy * dy
                            collided = (d2 <= cr_ * cr_) if isfinite(d2) \
                                else (hypot(dx, dy) <= cr_)
                        else:
                            collided = True
                        if collided:
                            hit_append((ei, fi) if ei < fi else (fi, ei))

    hits.sort()
    return hits


class HitTest:
    # ---- 三类基础判定（无状态纯函数，并发安全）----

    def circle_circle(self, a, b):
        """圆-圆碰撞：圆心距 <= 半径和，相切算碰撞。"""
        ax, ay, ar = _parse_circle(a)
        bx, by, br = _parse_circle(b)
        return _circles_hit(ax, ay, ar, bx, by, br)

    def circle_rect(self, c, r):
        """圆-矩形碰撞：圆心到矩形最近点距离 <= 半径，相切算碰撞。"""
        cx, cy, cr = _parse_circle(c)
        x0, y0, x1, y1 = _parse_rect(r)
        return _circle_rect_hit(cx, cy, cr, x0, y0, x1, y1)

    def rect_rect(self, a, b):
        """矩形-矩形碰撞：分离轴定理，相切算碰撞，结果与参数顺序无关。"""
        ax0, ay0, ax1, ay1 = _parse_rect(a)
        bx0, by0, bx1, by1 = _parse_rect(b)
        return _rects_hit(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1)

    # ---- 批量两两检测（均匀网格空间加速，SAP 回退）----

    def pairwise(self, objects):
        """对一批物体做两两碰撞检测。

        :param objects: 圆 ``((x, y), r)`` 与/或矩形 ``(x, y, w, h)``
            的可迭代序列，亦可使用带标签形式
            ``("circle", (x, y), r)`` / ``("rect", x, y, w, h)``。
        :returns: 碰撞对下标列表 ``[(i, j), ...]``，保证 ``i < j``、
            按字典序排序，与输入物体的排列顺序无关。
        :raises NonFiniteError / InvalidRadiusError / DegenerateRectError:
            输入非法时抛出，三类错误可区分。

        算法：先为每个物体建立轴对齐包围盒并做全量校验，再以均匀
        空间网格做空间加速——每个物体只登记进中心格，cell 边长取
        全体最大包围盒边长，碰撞对只可能出现在同格或 4 个前向
        邻居格中，每个物体对恰好检测一次，无需去重集合；网格退化
        （单点场、巨型物体占满场景等）时自动回退到 sweep-and-prune
        排序扫描。空间加速后平均近 O(n)，非纯暴力 O(n^2)。
        全程仅使用局部变量，无共享可变状态，多线程并发安全。
        """
        entries = []
        append = entries.append
        for idx, obj in enumerate(objects):
            kind, params = _parse_object(obj)
            if kind == _KIND_CIRCLE:
                x, y, r = params
                append((x - r, x + r, y - r, y + r, kind, idx, x, y, r, 0.0))
            else:
                x0, y0, x1, y1 = params
                append((x0, x1, y0, y1, kind, idx, x0, y0, x1, y1))

        if len(entries) < 2:
            return []
        return _uniform_grid(entries)

    # 常见别名，便于不同调用方命名习惯。
    def detect_all(self, objects):
        """同 :meth:`pairwise`。"""
        return self.pairwise(objects)

    def find_collisions(self, objects):
        """同 :meth:`pairwise`。"""
        return self.pairwise(objects)
