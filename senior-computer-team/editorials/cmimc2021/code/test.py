import sys
from k_means import k_means, dist, closest, kdTree

def dist(p1: tuple, p2: tuple) -> float:
    """ Squared distance between two points."""
    return (p1[0] - p2[0])*(p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1])

i = sys.argv[1]
with open(f"in/circlecovers{i}.txt") as f:
    N = int(f.readline())
    points = [tuple(map(int, f.readline().split())) for i in range(N)]
    M = int(f.readline())
    radii = [int(f.readline()) for i in range(M)]

with open(f"out/{i}.txt") as f:
   circles = [tuple(map(float, f.readline().split())) for i in range(M)]

# count covered
count = set()
for i, circle in enumerate(circles):
    count |= {point for point in points if dist(circle, point) <= radii[i]**2}

print(len(count))

