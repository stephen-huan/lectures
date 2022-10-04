import math, random, time

class CartesianTree:

    def __init__(self, key, parent=None, left=None, right=None):
        self.key = key
        self.parent = parent
        self.child = [left, right]

    def __str__(self, n=None, s="", d=0) -> str:
        if d == 0: n = self
        if n is None: return s
        s = self.__str__(n.child[1], s, d + 1)
        s += " "*4*d + str(n.key) + "\n"
        s = self.__str__(n.child[0], s, d + 1)
        return s

def cartesian_number(l: list) -> int:
    """ Returns the Cartesian number for a given list. """
    stk = []
    num = [0]*(2*len(l))
    k = 0
    for i in range(len(l)):
        while len(stk) > 0 and stk[-1] > l[i]:
            stk.pop()
            k += 1
        stk.append(l[i])
        num[k] = 1
        k += 1
    return int("".join(map(str, num)), 2)

def msb(n: int) -> int:
    """ Returns the index of the most significant bit of n.
        Technically can be done in O(1) but too much work.
        http://web.stanford.edu/class/cs166/lectures/16/Slides16.pdf
    """
    return int(math.log2(n))

def sparse_table(l: list) -> list:
    """ Computes a sparse table (think 2^n jump pointers) """
    n, m = len(l), math.ceil(math.log2(len(l)))
    dp = [[] for i in range(n)]

    for i in range(n):
        dp[i].append(i)

    k = 1
    for j in range(m):
        for i in range(n):
            if i + k >= n or j >= len(dp[i + k]):
                break
            # min with lambdas is REALLY slow
            # dp[i].append(min(dp[i][j], dp[i + k][j], key=lambda x: l[x]))
            dp[i].append(dp[i][j] if l[dp[i][j]] <= l[dp[i + k][j]] else dp[i + k][j])
        k <<= 1

    return dp

def sparse_rmq(l: list, table: list, i: int, j: int) -> int:
    """ Returns the index of the minimum element between i and j. """
    k = msb(j - i + 1)
    # return min(table[i][k], table[j - (1 << k) + 1][k], key=lambda x: l[x])
    return table[i][k] if l[table[i][k]] < l[table[j - (1 << k) + 1][k]] else table[j - (1 << k) + 1][k]

def full_table(l: list) -> list:
    """ Computes all possible ranges. """
    n = len(l)
    dp = [[] for i in range(n)]

    for i in range(n):
        dp[i].append(i)

    for j in range(n):
        for i in range(n - 1):
            if i + j >= n or j >= len(dp[i + 1]):
                break
            # dp[i].append(min(dp[i][j], dp[i + 1][j], key=lambda x: l[x]))
            dp[i].append(dp[i][j] if l[dp[i][j]] <= l[dp[i + 1][j]] else dp[i + 1][j])

    return dp

def full_rmq(table: list, i: int, j: int) -> int:
    """ O(1) RMQ. """
    return table[i][j - i]

def fischer_heun(l: list, k=1) -> tuple:
    """ Constructs the structure in O(n). """
    b = max(int(k*math.log2(len(l))) >> 1, 1) # k = 1, not k = 1/2 (>> 2 for 1/2)
    blocks = [l[i: i + b] for i in range(0, len(l), b)]
    a = [min(block) for block in blocks]
    indexes = [min(range(i, min(i + b, len(l))), key=lambda x: l[x]) for i in range(0, len(l), b)]
    table = sparse_table(a)
    ids = [cartesian_number(block) for block in blocks]

    tables = {}
    for i, block in enumerate(blocks):
        if ids[i] not in tables:
            # tables[ids[i]] = full_table(block)
            tables[ids[i]] = sparse_table(block)

    return b, a, indexes, blocks, ids, table, tables
    # return b, a, indexes, ids, table, tables

# def fh_rmq(o: list, b: int, a: list, indexes: list, ids: list, table: list, tables: list, i: int, j: int) -> int:
def fh_rmq(o: list, b: int, a: list, indexes: list, blocks: list, ids: list, table: list, tables: list, i: int, j: int) -> int:
    """ O(1) RMQ. """
    l, r = i//b, j//b         # block indexes
    li, ri = i - l*b, j - r*b # index in the block
    # in same block
    if l == r:
        # return b*l + full_rmq(tables[ids[l]], li, ri)
        return b*l + sparse_rmq(blocks[l], tables[ids[l]], li, ri)
    # return min(b*l + full_rmq(tables[ids[l]], li, b - 1), \
    #            indexes[sparse_rmq(a, table, l + 1, r - 1)] if r - 1 >= l + 1 else -1, \
    #            b*r + full_rmq(tables[ids[r]], 0, ri), key=lambda x: o[x] if x != -1 else float("inf"))
    # i1 = b*l + full_rmq(tables[ids[l]], li, b - 1)
    i1 = b*l + sparse_rmq(blocks[l], tables[ids[l]], li, b - 1)
    i2 = indexes[sparse_rmq(a, table, l + 1, r - 1)] if r - 1 >= l + 1 else -1
    i3 = b*r + sparse_rmq(blocks[r], tables[ids[r]], 0, ri)
    #E i3 = b*r + full_rmq(tables[ids[r]], 0, ri)

    v = i1 if o[i1] < o[i3] else i3
    return v if i2 == -1 or o[v] < o[i2] else i2

# http://web.stanford.edu/class/cs166/lectures/00/Slides00.pdf
# http://web.stanford.edu/class/cs166/lectures/01/Slides01.pdf

l = [32, 45, 16, 18, 9, 33]
l = [9, 3, 7, 1, 8, 12, 10, 20, 15, 18, 5]
l = [27, 18, 28, 18, 28, 45, 90, 45, 23, 53, 60, 28, 74, 71, 35]
# t = cartesian_tree(l)
# print(t)
# print(cartesian_number(l))

l = [16, 18, 33, 98]
# print(full_table(l))

l = [31, 41, 59, 26, 53, 58, 97, 93, 23, 84, 62, 64, 33, 83, 27]

# print(full_table(l))
stable = sparse_table(l)
# print(table)
# print(sparse_rmq(l, stable, 6, 9))
ftable = full_table(l)
# print(full_rmq(l, ftable, 6, 9))

# block size, "summary" array, summary mapping, cartesian numbers, sparse table, blockwise tables
fh = fischer_heun(l)
# v = fh_rmq(l, b, a, indexes, ids, table, tables, 6, 10)
# print(v, l[v])

def generate_array(size: int) -> list: return [random.randint(0, 10**7) for i in range(size)]

# random.seed(10)

l = generate_array(10**2)
# l = [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 2, 1, 1, 1, 0, 1, 1, 1, 1, 3, 0, 2, 1, 1, 0, 1, 2, 1, 1, 2, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 0, 1, 1, 2, 1, 1, 2, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
stable = sparse_table(l)
ftable = full_table(l)
fh = fischer_heun(l)
for i in range(len(l)):
    for j in range(i, len(l)):
        k = l[sparse_rmq(l, stable, i, j)]
        assert k == l[full_rmq(ftable, i, j)] and k == min(l[i:j + 1])
        assert k == l[fh_rmq(l, *fh, i, j)]
        # k = sparse_rmq(l, stable, i, j)
        # assert k == full_rmq(ftable, i, j) and min(range(i, j + 1), key=lambda x: l[x]) == k
        # assert k == fh_rmq(l, *fh, i, j)

SIZE = 10**4
# SIZE = 2**15
RANGES = 10**5
POINTS = 10**2
TRIALS = 10**3
x, y = [], []

# start = time.time()
# for t in range(TRIALS):
#     l = generate_array(SIZE)
#     fischer_heun(l, 1)
# print(time.time() - start)
# quit()

import matplotlib.pyplot as plt
import numpy as np

l = generate_array(SIZE)
queries = []
for t in range(RANGES):
    i = random.randint(0, len(l) - 1)
    j = random.randint(i, len(l) - 1)
    queries.append((i, j))

t = time.time()

for k in np.linspace(0, 2, num=POINTS):
    x.append(k)
    start = time.time()
    fh = fischer_heun(l, k)
    # start = time.time()

    # for q in queries:
    #     fh_rmq(l, *fh, *q)

    y.append(time.time() - start)

print(time.time() - t)

plt.title(r"Preprocessing time versus value of $k$")
# plt.title(r"Query time versus value of $k$")
plt.plot(x, y)
plt.xlabel(r"$k$")
plt.ylabel("Time (seconds)")
plt.savefig("rmq.png")

plt.show()
