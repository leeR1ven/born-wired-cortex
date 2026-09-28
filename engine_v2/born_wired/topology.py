"""Per-source spatial neighbours followed by random remaining targets."""

import itertools
import numbers

import numpy as np


def _exact_order(positions, source):
    ratios = [[float(x).as_integer_ratio() for x in row] for row in positions]
    denominator = max(d for row in ratios for _, d in row)
    points = [[n * (denominator // d) for n, d in row] for row in ratios]
    origin = points[source]
    return sorted(range(len(points)), key=lambda j: (
        sum((x - y) ** 2 for x, y in zip(points[j], origin)), j))


def make_edges(positions, local_degree, random_degree, seed=0):
    """Return int64 src/dst arrays; local edges precede random edges per source."""
    raw = np.asarray(positions)
    if raw.dtype.kind not in "iuf" or raw.ndim != 2:
        raise ValueError("positions must be a real numeric N x D array")
    points = np.array(raw, dtype=np.float64, copy=True)
    if points.shape[0] < 2 or points.shape[1] < 1 or not np.isfinite(points).all():
        raise ValueError("positions require N >= 2, D >= 1 and finite coordinates")
    for degree in (local_degree, random_degree):
        if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, numbers.Integral):
            raise ValueError("degrees must be non-negative integers")
        if degree < 0:
            raise ValueError("degrees must be non-negative integers")
    local_degree, random_degree = int(local_degree), int(random_degree)
    n = len(points)
    degree = local_degree + random_degree
    if degree > n - 1:
        raise ValueError("total degree exceeds available non-self targets")
    rng = np.random.default_rng(seed)
    src = np.repeat(np.arange(n, dtype=np.int64), degree)
    dst = np.empty(n * degree, dtype=np.int64)
    for source in range(n):
        selected = np.empty(0, dtype=np.int64)
        if local_degree:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                differences = points - points[source]
                distances = np.sum(differences * differences, axis=1)
            lost = (distances == 0) & np.any(points != points[source], axis=1)
            if not np.isfinite(distances).all() or lost.any():
                order = np.asarray(_exact_order(points, source), dtype=np.int64)
            else:
                order = np.argsort(distances, kind="stable")
            selected = order[order != source][:local_degree]
        remaining = np.ones(n, dtype=bool)
        remaining[source] = False
        remaining[selected] = False
        random_targets = rng.choice(np.flatnonzero(remaining), random_degree, replace=False)
        dst[source * degree:(source + 1) * degree] = np.concatenate((selected, random_targets))
    return src, dst


_RING_CACHE = {}


def _ring(dimension, radius):
    """Box offsets whose largest coordinate difference is exactly radius."""
    key = (dimension, radius)
    cached = _RING_CACHE.get(key)
    if cached is None:
        if radius == 0:
            cached = ((0,)*dimension,)
        else:
            cached = tuple(candidate for candidate in
                           itertools.product(range(-radius, radius + 1), repeat=dimension)
                           if max(abs(value) for value in candidate) == radius)
        _RING_CACHE[key] = cached
    return cached


def neighbour_edges(positions, local_degree, random_degree, seed=0):
    """The same edges as make_edges, found through a uniform grid.

    make_edges compares every point with every other point, which costs N
    squared comparisons: at fifty thousand cells that is already thousands of
    millions, and at a million it cannot be finished. This walks outwards from
    each point through neighbouring grid boxes instead, and stops as soon as
    the boxes it has not searched cannot hold anything closer than what it
    already has, so it returns the same neighbours make_edges would while the
    work grows with the number of neighbours rather than with N.

    Positions are moved into a unit box first, so one box width means the same
    distance on every axis. Ties are broken by index, as make_edges does, and
    the random remainder is drawn from the same pool in the same order, so the
    two functions agree edge for edge on the same seed.
    """
    raw = np.asarray(positions)
    if raw.dtype.kind not in "iuf" or raw.ndim != 2:
        raise ValueError("positions must be a real numeric N x D array")
    points = np.array(raw, dtype=np.float64, copy=True)
    if points.shape[0] < 2 or points.shape[1] < 1 or not np.isfinite(points).all():
        raise ValueError("positions require N >= 2, D >= 1 and finite coordinates")
    for degree in (local_degree, random_degree):
        if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, numbers.Integral):
            raise ValueError("degrees must be non-negative integers")
        if degree < 0:
            raise ValueError("degrees must be non-negative integers")
    local_degree, random_degree = int(local_degree), int(random_degree)
    n, dimension = points.shape
    degree = local_degree + random_degree
    if degree > n - 1:
        raise ValueError("total degree exceeds available non-self targets")
    rng = np.random.default_rng(seed)
    low = points.min(axis=0)
    extent = points.max(axis=0) - low
    # One scale for every axis, never one scale per axis: dividing each axis by
    # its own width would stretch the space and change which point is nearest.
    scale = float(extent.max()) if extent.max() > 0 else 1.
    unit = (points - low) / scale
    per_axis = max(1, int(round((n / 4.) ** (1./dimension))))
    width = 1./per_axis
    shape = (per_axis,)*dimension
    boxes = np.minimum((unit / width).astype(np.int64), per_axis - 1)
    keys = np.ravel_multi_index(boxes.T, shape)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    member = {}
    start = np.searchsorted(sorted_keys, np.arange(per_axis**dimension))
    stop = np.searchsorted(sorted_keys, np.arange(per_axis**dimension), side="right")
    for box in np.flatnonzero(stop > start):
        member[int(box)] = order[start[box]:stop[box]]
    selected = np.zeros((n, local_degree), dtype=np.int64)
    if local_degree:
        stamp = np.full(n, -1, dtype=np.int64)
        for source in range(n):
            found, distance = [], None
            radius = 0
            while True:
                fresh = []
                for offset in _ring(dimension, radius):
                    box = boxes[source] + np.asarray(offset)
                    if np.any(box < 0) or np.any(box >= per_axis):
                        continue
                    inside = member.get(int(np.ravel_multi_index(box, shape)))
                    if inside is None:
                        continue
                    new = inside[stamp[inside] != source]
                    new = new[new != source]
                    if new.size:
                        stamp[new] = source
                        fresh.append(new)
                if fresh:
                    found.extend(fresh)
                    everything = np.concatenate(found)
                    difference = unit[everything] - unit[source]
                    distance = np.einsum("ij,ij->i", difference, difference)
                if (distance is not None and distance.size >= local_degree
                        and np.partition(distance, local_degree - 1)[local_degree - 1]
                        <= (radius*width)**2):
                    break
                radius += 1
                if radius > per_axis + 1:
                    break
            everything = np.concatenate(found)
            difference = unit[everything] - unit[source]
            distance = np.einsum("ij,ij->i", difference, difference)
            ranking = np.lexsort((everything, distance))
            selected[source] = np.sort(everything[ranking][:local_degree])
    src = np.repeat(np.arange(n, dtype=np.int64), degree)
    dst = np.empty(n*degree, dtype=np.int64)
    for source in range(n):
        remaining = np.ones(n, dtype=bool)
        remaining[source] = False
        remaining[selected[source]] = False
        pool = np.flatnonzero(remaining)
        random_targets = rng.choice(pool, random_degree, replace=False)
        dst[source*degree:(source + 1)*degree] = np.concatenate((selected[source], random_targets))
    return src, dst
