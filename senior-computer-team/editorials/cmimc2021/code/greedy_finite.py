import sys
input = sys.stdin.readline # fast cin

import random
from k_means import dist, closest, kdTree, init

random.seed(1)

def quadratic(a: float, b: float, c: float) -> tuple:
    """ Solves a quadratic """
    disc = b*b - 4*a*c
    if disc < 0:
        return None, None
    disc = disc**0.5/(2*a)
    x1, x2 = -b/(2*a) + disc, -b/(2*a) - disc
    return x1, x2

def nudge(p1: tuple, p2: tuple, c: tuple) -> tuple:
    """ Nudge c in the direction of p1 and p2 then round to avoid error. """
    E = 10**-6
    # vector pointing from c to the midpoint of p1 and p2
    d = ((p1[0] + p2[0])/2 - c[0], (p1[1] + p2[1])/2 - c[1])
    c = (c[0] + E*d[0], c[1] + E*d[1])
    return tuple(map(lambda x: round(x, 6), c))

def fit(p1: tuple, p2: tuple, r: float) -> tuple:
    """ The two circles determined by two points and a radius. """
    # find line which the center lies on
    (x0, y0), (x1, y1) = p1, p2
    c = ((x0 + x1)/2, (y0 + y1)/2)
    dx, dy = x0 - x1, y0 - y1
    if dy == 0: # vertical line
        xc = c[0] + dy/dx*c[1]
        yc1, yc2 = quadratic(1, -2*y0, y0*y0 + (x0*x0 - 2*x0*xc + xc*xc) - r*r)
        if yc1 is None: return
        c1, c2 = (xc, yc1), (xc, yc2)
    else:
        m, i = -dx/dy, dx/dy*c[0] + c[1]
        # solve quadratic equation
        a = m*m + 1
        b = -2*(m*(y0 - i) + x0)
        c = x0*x0 + (y0 - i)*(y0 - i) - r*r
        xc1, xc2 = quadratic(a, b, c)
        if xc1 is None: return
        c1, c2 = (xc1, m*xc1 + i), (xc2, m*xc2 + i)

    c1, c2 = nudge(p1, p2, c1), nudge(p1, p2, c2)
    assert abs(dist(c1, p1) - r*r) < 10**-3 and abs(dist(c2, p2) - r*r) < 10**-3
    return (c1, c2)

def get_circles(points, t, r):
    points = init(min(512, len(points)), points, t)
    t2 = kdTree(points)
    circles = list(points)
    for p1 in points:
        for p2 in t2.nnsearch(p1, 2*r):
            if p1 != p2:
                p = fit(p1, p2, r)
                if p is not None:
                    circles += p
    print(len(circles))
    return circles

N = int(input())
points = [tuple(map(int, input().split())) for i in range(N)]
M = int(input())
radii = [int(input()) for i in range(M)]
circle_order = sorted(range(M), key=lambda i: -radii[i])

active = set(points)
circles = [None]*M
for c in range(M):
    print(c)
    r = radii[circle_order[c]]
    t = kdTree(list(active))
    best, bestc = None, -1
    for center in get_circles(list(active), t, r):
        count = len(t.nnsearch(center, r))
        if count > bestc:
            best, bestc = center, count

    circles[circle_order[c]] = best
    active -= set(t.nnsearch(best, r))

for circle in circles:
    print(" ".join(map(str, circle)))

# count covered
count = set()
for i, circle in enumerate(circles):
    count |= {point for point in points if dist(circle, point) <= radii[i]**2}
assert False, len(count)

