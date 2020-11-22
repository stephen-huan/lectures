import sys
input = sys.stdin.readline # fast cin

def is_valid(i, j): return 0 <= i < N and 0 <= j < N

def get_children(i, j, k):
    return [child for child in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
            if is_valid(*child) and k >= abs(m[i][j] - m[child[0]][child[1]])]

def assign(u, num, k, ids={}):
    stk = [u]
    while len(stk) > 0:
        n = stk.pop()
        ids[n] = num
        for child in get_children(*n, k):
            if child not in ids:
                ids[child] = num
                stk.append(child)

def connected(k):
    ids, num = {}, 0
    for u in [(i, j) for i in range(N) for j in range(N)]:
        if u not in ids:
            assign(u, num, k, ids)
            num += 1
    return ids, num

N, K = map(int, input().split())
m = [list(map(int, input().split())) for i in range(N)]

l, r = 0, 10**9 + 1
while l < r:
    k = (l + r)>>1
    ids, num = connected(k)
    comps = {i: [] for i in range(num)}
    for key, value in ids.items():
        comps[value].append(key)
    rooms = max(map(len, comps.values()))
    if rooms < K:
        l = k + 1
    else:
        r = k

print(l)

