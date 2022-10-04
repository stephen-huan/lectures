import sys
input = sys.stdin.readline # fast cin

N = int(input())
l = list(input().split())

d = {}
for x in l:
    d[x] = d.get(x, 0) + 1

print(max(d.values()))

