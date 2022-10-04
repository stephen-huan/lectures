# median of medians to find order statistics in O(n)
# CLRS chapter 9 "Medians and Order Statistics"
import sys, random
sys.setrecursionlimit(10**4)
random.seed(1)

### in-place methods from CLRS

def split(a: list, l: int, r: int, k: int) -> int:
    """ Split the list a[l: r + 1] in place about the value a[k]. """
    a[r], a[k], x, i = a[k], a[r], a[k], l # move pivot to the rightmost end
    for j in range(l, r):
        if a[j] <= x:
            a[i], a[j] = a[j], a[i]
            i += 1
    a[i], a[r] = a[r], a[i]                # move pivot to the middle
    return i

def __select(a: list, l: int, r: int, i: int) -> int:
    """ Returns sorted(a[l: r + 1])[i] in O(n). """
    p = split(a, l, r, l)
    k = p - l
    if i == k: # pivot is the answer
        return a[p]
    # recur on sublist and get rid of pivot
    return __select(a, l, p - 1, i) if i < k else \
           __select(a, p + 1, r, i - k - 1)

def select(l: list, i: int, index: bool=False, in_place: bool=False) -> int:
    """ Wrapper over __select. """
    v = __select(l if in_place else list(l), 0, len(l) - 1, i)
    return v if not index else [i for i in range(len(l)) if l[i] == v][0]

### extra memory for ease of accounting for duplicate values

def split(l: list, x: float) -> tuple:
    """ Splits the list by a particular value x. """
    left, mid, right = [], [], []
    for v in l:
        # if the value is equal to the cutoff, add it to the right side 
        (left if v < x else right if v > x else mid).append(v)
    return left, mid, right

def slow_select(l: list, i: int):
    """ Returns sorted(l)[i] in O(n) expected. """
    left, mid, right = split(l, l[0])
    k, m = len(left), len(mid)
    if k <= i <= k + m - 1: # pivot is the answer
        return mid[0]
    # recur on sublist and get rid of pivot
    return slow_select(left, i) if i < k else slow_select(right, i - k - m)

def median(l: list) -> float:
    """ Returns the upper median of l, via a sort. """
    return sorted(l)[len(l)//2]

def select(l: list, i: int):
    """ Returns sorted(l)[i] in O(n) with median of medians as a pivot. """
    if len(l) == 1: # base case
        return l[0]
    medians = [median(l[5*i: 5*(i + 1)]) for i in range(-(-len(l)//5))]
    left, mid, right = split(l, select(medians, len(medians)//2))
    k, m = len(left), len(mid)
    if k <= i <= k + m - 1: # pivot is the answer
        return mid[0]
    # recur on sublist and get rid of pivot
    return select(left, i) if i < k else select(right, i - k - m)

l = [1, 3, 4, 2, 4, 4, 2, 4, 5, 1, 3, 4, 2]
# l = [1, 3, 4, 2, 5]
# l = [1]*10**4
l = [random.random() for i in range(10**3)]
l.sort()
lp = sorted(l)
for i in range(len(l)):
    assert lp[i] == select(l, i)

