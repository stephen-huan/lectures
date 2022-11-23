import itertools

N = 5
l = list(range(N))
perms = list(itertools.permutations(l))
M = l[N//2] # median

dt = ({0: (6, 1),
       1: (5, 2),
       2: (4, 3),
       6: (10, 7),
       7: (9, 8)
      },
      [(0, 1), (0, 2), (1, 2), (1,), (2,), (0,),
               (1, 2), (0, 2), (0,), (2,), (1,)]
     )

class Int():

    """ Custom integer class to count the number of comparisions sort does. """

    count = 0

    def __init__(self, x: int):
        self.x = x

    def __lt__(self, other):
        """ Needs to implement either __lt__ or __gt__. """
        Int.count += 1
        return self.x < other.x

def walk(dt: tuple, l: list) -> float:
    """ Walks the list down the decision tree. """
    n = count = 0
    while len(dt[1][n]) > 1:      # not a leaf node
        i, j = dt[1][n]           # indexes to compare
        n = dt[0][n][l[i] > l[j]] # update node from result of comparison
        count += 1                # record number of comparisons
    assert l[dt[1][n][0]] == M    # median is the median
    return dt[1][n][0], count     # final median

def split(poss: list, compare: tuple) -> tuple:
    """ Splits the list of possibilities based off the comparision. """
    left, right = [], []
    for l in poss:
        (left if l[compare[0]] <= l[compare[1]] else right).append(l)
    return left, right

def get_median(poss: list) -> int:
    """ Returns the median, None if it can't be determined (not a leaf). """
    i = poss[0].index(M)
    for l in poss:
        if l.index(M) != i:
            return
    return i

def __gen_dt(dt: tuple, poss: list, n: int) -> list:
    """ Generates all possible decision trees. """
    m = poss[0].index(M) if len(poss) == 1 else get_median(poss)
    if m is not None:
        dt[1][n] = (m,)
        return [dt]

    l = []
    for child in ((i, j) for i in range(N) for j in range(i + 1, N)):
        left, right = split(poss, child)
        if len(left) != 0 and len(right) != 0: # comparision achives something
            ids = dt[2]
            child_dt = ({**dt[0]}, {**dt[1]}, ids + 2)
            child_dt[0][n] = (ids + 1, ids + 2)
            child_dt[1][n] = child

            ll = __gen_dt(child_dt, left, ids + 1) # fill in left
            for dt1 in ll:
                lr = __gen_dt(dt1, right, ids + 2) # fill in right
                l += lr
    return l

def gen_dt() -> list:
    """ Generates all possible decision trees. """
    return [dt[:-1] for dt in __gen_dt(({}, {}, 0), perms, 0)]

def gen_good(dt: tuple, poss: list, n: int) -> list:
    """ Generates a locally greedy decision tree. """
    m = poss[0].index(M) if len(poss) == 1 else get_median(poss)
    if m is not None:
        dt[1][n] = (m,)
        return dt

    best, best_d = None, float("inf")
    for child in ((i, j) for i in range(N) for j in range(i + 1, N)):
        left, right = split(poss, child)
        d = abs(len(left) - len(right))
        if d < best_d:
            best, best_d = child, d

    left, right = split(poss, best)
    dt, ids = (dt[0], dt[1], dt[2] + 2), dt[2]
    dt[0][n] = (ids + 1, ids + 2)
    dt[1][n] = best
    dt = gen_good(dt,  left, ids + 1) # fill in left
    dt = gen_good(dt, right, ids + 2) # fill in right
    return dt

def E(dt: tuple) -> float:
    """ Expected number of comparisons for the decision tree. """
    return sum(walk(dt, perm)[1] for perm in perms)/len(perms)

def mx(dt: tuple) -> float:
    """ Expected number of comparisons for the decision tree. """
    return max(walk(dt, perm)[1] for perm in perms)

def render_dt(dt: tuple) -> str:
    """ Returns a Python function as a string that computes the median. """
    I = " "*4
    s = ["def median(l: list) -> float:",
         f'{I}""" Computes the median of l, if len(l) == {N}. """']
    stk = [(0, 1)]
    while len(stk) > 0:            # render decision tree with dfs
        n, d = stk.pop()
        if n % 2 == 1:             # "odd" parity node, else clause
            s.append(f"{I*(d - 1)}else:")
        if len(dt[1][n]) == 1:     # leaf node
            s.append(f"{I*d}return l[{dt[1][n][0]}]")
        else:                      # not a leaf node
            i, j = dt[1][n]
            s.append(f"{I*d}if l[{i}] < l[{j}]:")
            for child in dt[0][n]: # expand on children
                stk.append((child, d + 1))
    return "\n".join(s)

def render_dt_ternary(dt: tuple) -> str:
    """ Returns a Python function as a string that computes the median. """
    I = " "*4
    s = ["def median(l: list) -> float:",
         f'{I}""" Computes the median of l, if len(l) == {N}. """']
    stk = [(0, 1)]
    while len(stk) > 0:            # render decision tree with dfs
        n, d = stk.pop()
        if n % 2 == 1:             # "odd" parity node, else clause
            s.append(f"{I*(d - 1)}else:")
        if len(dt[1][n]) == 1:     # leaf node
            s.append(f"{I*d}return l[{dt[1][n][0]}]")
        else:                      # not a leaf node
            i, j = dt[1][n]
            child1, child2 = dt[1][dt[0][n][0]], dt[1][dt[0][n][1]]
            if len(child1) == len(child2) == 1:
                k, l = child1[0], child2[0]
                s.append(f"{I*d}return l[{l}] if l[{i}] < l[{j}] else l[{k}]")
                # f = "min" if l == i else "max"
                # s.append(f"{I*d}return {f}(l[{i}], l[{j}])")
            else:
                s.append(f"{I*d}if l[{i}] < l[{j}]:")
                for child in dt[0][n]: # expand on children
                    stk.append((child, d + 1))
    return "\n".join(s)


def __ternary(dt: tuple, n: int) -> str:
    """ Recursive helper function. """
    if len(dt[1][n]) == 1:                  # base case, leaf 
        return f"l[{dt[1][n][0]}]"
    u, v = dt[0][n]
    i, j = dt[1][n]
    if len(dt[1][u]) == len(dt[1][v]) == 1: # base case, most trivial ternary
        k, l = dt[1][u][0], dt[1][v][0]
        return f"(l[{k}] if l[{i}] < l[{j}] else l[{l}])"
    return f"({__ternary(dt, u)} if l[{i}] < l[{j}] else {__ternary(dt, v)})"

def render_ternary(dt: tuple) -> str:
    """ Returns a Python function as a string that computes the median. """
    I = " "*4
    s = ["def median(l: list) -> float:",
         f'{I}""" Computes the median of l, if len(l) == {N}. """']
    return "\n".join(s) + "\n    return " + __ternary(dt, 0)

if __name__ == "__main__":
    dt = gen_good(({}, {}, 0), perms, 0)
    print(dt[0])
    print([dt[1][i] for i in range(dt[2] + 1)])
    print(f"Internal nodes: {sum(1 for x in dt[1].values() if len(x) == 2)}")
    print(f"    Leaf nodes: {sum(1 for x in dt[1].values() if len(x) == 1)}")
    print(f" Average value: {E(dt):.3f}, max depth: {mx(dt)}")

    [sorted(map(Int, perm)) for perm in perms]
    print(f"Python sorted comparisions: {Int.count/len(perms):.3f}")

    for render in [render_dt, render_dt_ternary, render_ternary]:
        print("-"*10 + "\n" + render(dt))

    exit()

    poss = gen_dt()
    print(len(poss))
    dt1 = min(poss, key=E)
    print(dt1)
    print(E(dt1))
    print(mx(dt1))
    dt2 = min(poss, key=mx)
    print(E(dt2))
    print(mx(dt2))

