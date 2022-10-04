import sys
input = sys.stdin.readline # fast cin

from collections import deque

def bfs(start, seen):
    q = deque([(None, start)])
    ans = 0
    while len(q) > 0:
        p, n = q.popleft()
        seen.add(n)
        for child in graph[n]:
            if child not in seen:
                seen.add(child)
                q.append((n, child))
            else:
                ans += child != p
    return ans//2

N, M = map(int, input().split())
graph = {i: set() for i in range(N)}
edges = 0
for i in range(M):
    a, b, w = map(lambda x: int(x) - 1, input().split())
    if b not in graph[a]:
        graph[a].add(b)
        graph[b].add(a)
    else:
        edges += 1

seen = set()
for i in range(N):
    if i not in seen:
        edges += bfs(i, seen)

print(edges)

