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

def update_point(p, width, height, step):
    best, bestc = None, -1
    x, y = p
    for i in range(round(width/step)):
        for j in range(round(height/step)):
            center = (x + i*step - width/2, y + j*step - height/2)
            count = len(t.nnsearch(center, r))
            if count > bestc:
                best, bestc = center, count
    return best, bestc

def best_point():
    w, h, d = x1 - x0, y1 - y0, D
    p, count = ((x0 + x1)/2, (y0 + y1)/2), 0
    i = 0
    while True:
        oldp, oldc = p, count
        p, count = update_point(p, w, h, d)
        if dist(oldp, p) < 0.001:
            break
        w, h, d = w/2, h/2, d/2
        i += 1
    return p

active = set(points)
circles = [None]*M
D = 32
for c in range(M):
    r = radii[circle_order[c]]
    t = kdTree(list(active))
    center = best_point()
    circles[circle_order[c]] = center
    active -= set(t.nnsearch(center, r))

for circle in circles:
    print(" ".join(map(str, circle)))

# count covered
count = set()
for i, circle in enumerate(circles):
    count |= {point for point in points if dist(circle, point) <= radii[i]**2}
assert False, len(count)

