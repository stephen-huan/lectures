import sys
input = sys.stdin.readline # fast cin

from k_means import k_means, dist, closest, kdTree

def covered(circle: tuple, r: float) -> int:
    return sum(dist(circle, point) <= r**2 for point in points)

def add(u1: set, u2: set) -> int:
    return len(u2) - len(u2 & u1)

N = int(input())
points = [tuple(map(int, input().split())) for i in range(N)]
M = int(input())
radii = [int(input()) for i in range(M)]
circle_order = sorted(range(M), key=lambda i: radii[i])

centers, ids, groups = k_means(M, points)
def f(i):
    return -sum(dist(point, centers[i]) for point in groups[i])
center_order = sorted(range(M), key=f)

active = set(points)
circles = [None]*M
for i in range(M):
    # print(i)
    t = kdTree(list(active))
    rk, k = circle_order[i], center_order[i]
    candidates = [centers[k]] + groups[k]
    data = points if True else groups[k]
    r, r2 = radii[rk], radii[rk]**2
    circle = max(candidates, key=lambda x: len(t.nnsearch(x, r)))
    circles[rk] = circle
    active -= t.nnsearch(circle, r)

for circle in circles:
    print(" ".join(map(str, circle)))

# count covered
count = set()
for i, circle in enumerate(circles):
    count |= {point for point in points if dist(circle, point) <= radii[i]**2}
assert False, len(count)

