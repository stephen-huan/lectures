import sys
input = sys.stdin.readline # fast cin

from mip import Model, MAXIMIZE, CBC, BINARY, xsum
from k_means import dist, closest, kdTree

def solve(active: list, centers: list, sets: list, M: int) -> list:
    N, K = len(active), len(sets)
    ### model and variables
    m = Model(sense=MAXIMIZE, solver_name=CBC)
    # whether the ith set is picked
    x = [m.add_var(name=f"x{i}", var_type=BINARY) for i in range(K)]
    # whether the ith point is covered
    y = [m.add_var(name=f"y{i}", var_type=BINARY) for i in range(N)]

    ### constraints
    m += xsum(x) == M, "number_circles"
    for i in range(N):
        # if yi is covered, at least one set needs to have it
        included = [x[k] for k in range(K) if active[i] in sets[k]]
        m += xsum(included) >= y[i], f"inclusion{i}"

    ### objective: maximize number of circles covered
    m.objective = xsum(y[i] for i in range(N))

    m.emphasis = 2 # emphasize optimality
    m.verbose = 1
    status = m.optimize()
    circles = [centers[i] for i in range(K) if x[i].x >= 0.99]
    covered = {active[i] for i in range(N) if y[i].x >= 0.99}

    return circles, covered

def gen_grid(t, r, D):
    sets, centers = [], []
    for i in range(round((x1 - x0)/D)):
        for j in range(round((y1 - y0)/D)):
            center = (i*D, j*D)
            centers.append(center)
            sets.append(set(t.nnsearch(center, r)))
    return centers, sets

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
    return (c1, c2)

def get_circles(points, t, r):
    circles = list(points)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p = fit(points[i], points[j], r)
            if p is not None:
                circles += p
    sets = [set(t.nnsearch(circle, r)) for circle in circles]
    return circles, sets

N = int(input())
points = [tuple(map(int, input().split())) for i in range(N)]
M = int(input())
radii = [int(input()) for i in range(M)]
circle_order = sorted(range(M), key=lambda i: -radii[i])

x, y = zip(*points)
x0, x1, y0, y1 = min(x), max(x), min(y), max(y)

freq = {r: [] for r in radii}
for i, r in enumerate(radii):
    freq[r].append(i)

circles = [None]*M
active = set(points)
for r in sorted(freq, reverse=True):
    K = len(freq[r])
    t = kdTree(list(active))
    # centers, sets = gen_grid(t, r, 0.2)
    centers, sets = get_circles(list(active), t, r)
    rcircles, covered = solve(list(active), centers, sets, K)
    for i in range(K):
        circles[freq[r][i]] = rcircles[i]
    active -= covered

for circle in circles:
    print(" ".join(map(str, circle)))

# count covered
count = set()
for i, circle in enumerate(circles):
    count |= {point for point in points if dist(circle, point) <= radii[i]**2}
assert False, len(count)

