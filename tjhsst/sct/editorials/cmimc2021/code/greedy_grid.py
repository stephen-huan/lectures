import sys
input = sys.stdin.readline # fast cin

from k_means import dist, closest, kdTree

N = int(input())
points = [tuple(map(int, input().split())) for i in range(N)]
M = int(input())
radii = [int(input()) for i in range(M)]
circle_order = sorted(range(M), key=lambda i: -radii[i])

x, y = zip(*points)
x0, x1, y0, y1 = min(x), max(x), min(y), max(y)

active = set(points)
circles = [None]*M
D = 1
for c in range(M):
    print(c)
    r = radii[circle_order[c]]
    t = kdTree(list(active))
    best, bestc = None, -1
    for i in range(round((x1 - x0)/D)):
        for j in range(round((y1 - y0)/D)):
            center = (i*D, j*D)
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

