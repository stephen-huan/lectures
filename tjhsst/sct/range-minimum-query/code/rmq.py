import math

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

def cartesian_tree(l: list) -> CartesianTree:
    """ Constructs a Cartesian tree for a given list in O(n). """
    stk = []
    for i in range(len(l)):
        c = None
        while len(stk) > 0 and stk[-1].key > l[i]:
            c = stk.pop()
        stk.append(CartesianTree(l[i], stk[-1] if len(stk) > 0 else None, c))
        # add right child
        if len(stk) > 1:
            stk[-2].child[1] = stk[-1]
    return stk[0]

def cartesian_number(l: list) -> int:
    """ Returns the Cartesian number for a given list. """
    stk = []
    num = 0
    for i in range(len(l)):
        while len(stk) > 0 and stk[-1] > l[i]:
            stk.pop()
            num <<= 1
        stk.append(l[i])
        num <<= 1
        num |= 1
    return num << (2*len(l) - msb(num) - 1)

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
            dp[i].append(min(dp[i][j], dp[i + k][j], key=lambda x: l[x]))
        k <<= 1

    return dp

def sparse_rmq(l: list, table: list, i: int, j: int) -> int:
    """ Returns the index of the minimum element between i and j. """
    k = msb(j - i + 1)
    return min(table[i][k], table[j - (1 << k) + 1][k], key=lambda x: l[x])

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
            dp[i].append(min(dp[i][j], dp[i + 1][j], key=lambda x: l[x]))

    return dp

def full_rmq(table: list, i: int, j: int) -> int:
    """ O(1) RMQ. """
    return table[i][j - i]

def fischer_heun(l: list) -> tuple:
    """ Constructs the structure in O(n). """
    b = max(int(math.log2(len(l))) >> 2, 1)
    blocks = [l[i: i + b] for i in range(0, len(l), b)]
    a = [min(block) for block in blocks]
    indexes = [min(range(i, i + b), key=lambda x: l[x]) for i in range(0, len(l), b)]
    table = sparse_table(a)
    ids = [cartesian_number(block) for block in blocks]

    tables = {}
    for i, block in enumerate(blocks):
        if ids[i] not in tables:
            tables[ids[i]] = full_table(block)

    return b, a, indexes, ids, table, tables

def fh_rmq(o: list, b: int, a: list, indexes: list, ids: list, table: list, tables: list, i: int, j: int) -> int:
    """ O(1) RMQ. """
    l, r = i//b, j//b         # block indexes
    li, ri = i - l*b, j - r*b # index in the block
    # in same block
    if l == r:
        return b*l + full_rmq(tables[ids[l]], li, ri)
    return min(b*l + full_rmq(tables[ids[l]], li, b - 1), \
               indexes[sparse_rmq(a, table, l + 1, r - 1)] if r - 1 >= l + 1 else -1, \
               b*r + full_rmq(tables[ids[r]], 0, ri), key=lambda x: o[x] if x != -1 else float("inf"))

### PRESENTATION EXAMPLE

l = [5, 3, 4, 1, 2]

print("dp table")
dp = full_table(l)
for row in dp:
    print([l[i] for i in row])

print("sparse table")
sparse = sparse_table(l)
for row in sparse:
    print([l[i] for i in row])


# exit()

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
b, a, indexes, ids, table, tables = fischer_heun(l)
# v = fh_rmq(l, b, a, indexes, ids, table, tables, 6, 10)
# print(v, l[v])

for i in range(len(l)):
    for j in range(i, len(l)):
        k = sparse_rmq(l, stable, i, j)
        assert k == full_rmq(ftable, i, j) and min(range(i, j + 1), key=lambda x: l[x]) == k
        assert k == fh_rmq(l, b, a, indexes, ids, table, tables, i, j)
