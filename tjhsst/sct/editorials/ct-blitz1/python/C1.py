import sys
input = sys.stdin.readline # fast cin

import heapq

def union_init(n: int) -> tuple: return {i: i for i in range(n)}, {i: 1 for i in range(n)}

def find(parent: dict, u: int) -> int:
    if parent[u] == u:
        return u
    parent[u] = find(parent, parent[u])
    return parent[u]

def union(parent: dict, size: dict, u: int, v: int) -> bool:
    ur, vr = find(parent, u), find(parent, v)
    if ur == vr:
        return False

    x, y = (ur, vr) if size[ur] < size[vr] else (vr, ur)
    parent[x] = y
    size[y] += size[x]
    return True

def rc(point): return N*point[0] + point[1]

N, K = map(int, input().split())
m = [list(map(int, input().split())) for i in range(N)]
dx = [[abs(m[i][j] - m[i + 1][j]) for j in range(N)] for i in range(N - 1)]
dy = [[abs(m[i][j] - m[i][j + 1]) for j in range(N - 1)] for i in range(N)]

v = {0}
h = []
for i in range(N):
    for j in range(N):
        if i < N - 1:
            v.add(dx[i][j])
            heapq.heappush(h, (dx[i][j], (i, j), (i + 1, j)))
        if j < N - 1:
            v.add(dy[i][j])
            heapq.heappush(h, (dy[i][j], (i, j), (i, j + 1)))
v = sorted(v)

parent, size = union_init(N*N)
rooms = 1
for k in v:
    l = []
    while len(h) > 0 and k >= h[0][0]:
        p, start, end = heapq.heappop(h)
        l.append((start, end))

    for start, end in l:
        s, e = rc(start), rc(end)
        union(parent, size, s, e)
        rooms = max(rooms, size[find(parent, s)])

    if rooms >= K:
        print(k)
        break

