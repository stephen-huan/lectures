import sys
input = sys.stdin.readline # fast cin

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

def kruskal(edges) -> list:
    parent, size = union_init(N)
    span = []
    for e in edges:
        u, v, w = e
        u, v = u - 1, v - 1
        # all edges after this will be negative
        if w < 0:
            break
        if find(parent, u) != find(parent, v):
            span.append((w, u, v))
            union(parent, size, u, v)
    return span

N, M = map(int, input().split())
edges = [list(map(int, input().split())) for i in range(M)]
span = kruskal(sorted(edges, key=lambda x: -x[-1]))
print(sum(x[-1] for x in edges) - sum(x[0] for x in span))

