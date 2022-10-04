"""
https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/kdtrees.pdf
https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/kdrangenn.pdf
https://people.eecs.berkeley.edu/~jrs/189s19/lec/25.pdf
"""

### median selection

def select_split(l: list, x: float) -> tuple:
    """ Splits the list by a particular value x. """
    left, mid, right = [], [], []
    for v in l:
        # if the value is equal to the cutoff, add it to the right side 
        (left if v < x else right if v > x else mid).append(v)
    return left, mid, right

def __median(l: list) -> float:
    """ Returns the upper median of l, via a sort. """
    l.sort()
    return l[len(l)//2]

"""
Using a median5 function for groups of 5 and handling the last group separately:
b = len(l) % 5 != 0
medians = [__median(l[5*i: 5*(i + 1)]) for i in range(-(-len(l)//5) - b)]
if b:
    medians.append(___median(l[-(len(l) % 5):]))
"""

def select(l: list, i: int):
    """ Returns sorted(l)[i] in O(n) with median of medians as a pivot. """
    if len(l) == 1: # base case
        return l[0]
    medians = [__median(l[5*i: 5*(i + 1)]) for i in range(-(-len(l)//5))]
    left, mid, right = select_split(l, select(medians, len(medians)//2))
    k, m = len(left), len(mid)
    if k <= i <= k + m - 1: # pivot is the answer
        return mid[0]
    # recur on sublist and get rid of pivot
    return select(left, i) if i < k else select(right, i - k - m)

def median(points: list, cd: int) -> tuple:
    """ Picks the point which is the median along the dimension cd. """
    points.sort(key=lambda p: p[cd])
    i = len(points)//2
    return points[i], points[:i] + points[i + 1:]

def median(points: list, cd: int) -> tuple:
    """ Picks the point which is the median along the dimension cd. """
    m = select([point[cd] for point in points], len(points)//2)
    for i in range(len(points)):
        if points[i][cd] == m: # pick any point with value m
            break
    return points[i], points[:i] + points[i + 1:]

### helper methods

def dist(p1: tuple, p2: tuple) -> float:
    """ Squared distance between two points."""
    return sum((p1[i] - p2[i])*(p1[i] - p2[i]) for i in range(len(p1)))

def closest(points: list, q: tuple) -> tuple:
    """ Returns the point closest to q in points (nearest neighbor query). """
    return min(points, key=lambda p: dist(p, q))

def distbb(p: tuple, bb: list) -> float:
    """ Squared distance between a point and a bounding box. """
    # three cases, use x if x is in the box, otherwise one of the bounds
    bbp = tuple(box[0] if x < box[0] else (box[1] if x > box[1] else x)
                for x, box in zip(p, bb))
    return dist(p, bbp)

def trimbb(bb: list, cd: int, p: int, d: int) -> list:
    """ Trims the bounding box by the plane x_cd = p[cd]. """
    if len(bb) == 0: return bb
    bb = list(list(box) for box in bb) # copy
    bb[cd][1 - d] = p[cd]              # update, assuming p[cd] is valid
    return bb

### k-d tree

def subsplit(pointsd: list, seen: set) -> list:
    """ Only takes the points in pointsd that are in seen. """
    return [[point for point in points if point in seen] for points in pointsd]
    return list(map(lambda l: list(filter(lambda p: p in seen, l)), pointsd))

def split(points: list, cd: int, p: int) -> tuple:
    """ Splits the list of points by the plane x_cd = p[cd]. """
    left, right = [], []
    for point in points:
        # add point with the same value as p at cd to the right side
        (left if point[cd] < p[cd] else right).append(point)
    return left, right

class kdNode:

    """ kd-tree node. """

    def __init__(self, point: tuple=None, cd: int=0) -> None:
        self.child, self.point, self.cd, self.bb = [None, None], point, cd, []
        self.D, self.tight_bb = len(point) if point is not None else 0, False

    def __str__(self, n: "kdNode"=None, d: int=0) -> str:
        """ Fancy string representation. """
        if d == 0: n = self # called with None by defualt, set to the root
        if n is None: return [] # leaf node
        s = [f"{' '*4*d}{n.point}"]
        s += self.__str__(n.child[0], d + 1)
        s += self.__str__(n.child[1], d + 1)
        return "\n".join(s) if d == 0 else s

    def dir(self, p: tuple) -> int:
        """ Gets the proper left/right child depending on the point. """
        return p[self.cd] >= self.point[self.cd]

    def tighten(self, t: "KdNode"=None) -> None:
        """ Tighten bounding boxes in O(nd). """
        if t is None: t = self # called with None, set to the root 
        l, r, t.tight_bb = t.child[0], t.child[1], True
        # recur on children
        if l is not None: self.tighten(l)
        if r is not None: self.tighten(r)
        if l is None and r is None:  # leaf node, box is just the singular point
            t.bb = [(t.point[d], t.point[d]) for d in range(t.D)]
        elif l is None or r is None: # one child, inherit box of child
            t.bb = l.bb if l is not None else r.bb
            t.bb = [(min(box[0], v), max(box[1], v))  # add node's point
                    for box, v in zip(t.bb, t.point)]
        else:                        # two children, combine boxes
            t.bb = [(min(bbl[0], bbr[0], v), max(bbl[1], bbr[1], v))
                    for bbl, bbr, v in zip(l.bb, r.bb, t.point)]

    def __add(self, t: "kdNode", p: tuple, parent: "kdNode"=None) -> None:
        """ Insert the given point into the tree. """
        if t is None:      # found leaf to insert new node in 
            t = kdNode(p, (parent.cd + 1) % parent.D)
        elif t.point == p: # ignore duplicates
            return t
        else:              # update pointers
            t.child[t.dir(p)] = self.__add(t.child[t.dir(p)], p, t)
            t.tight_bb = False # no longer use tight bounding boxes
            # is it worth O(d log n) instead of O(log n) for tighter boxes?
            # if so, manually update t.bb over each of the d dimensions
        return t

    def add(self, p: tuple) -> None:
        """ Wrapper over the recursive helper function __add. """
        if self.point is None: # empty tree, simply change our own point
            self.__init__(p)
        self.__add(self, p)

    def __closest(self, t: "kdNode", p: tuple, curr_bb: list) -> tuple:
        """ Returns the closest point to p in the tree (nearest neighbor). """
        # all points in this bounding box farther than existing point
        bb = t.bb if t is not None and t.tight_bb else curr_bb
        if t is None or distbb(p, bb) > self.best_dist:
            return
        # update best point
        d = dist(p, t.point)
        if d < self.best_dist:
            self.best, self.best_dist = t.point, d
        # visit subtrees in order of distance from p
        i, j = t.dir(p), 1 - t.dir(p)
        self.__closest(t.child[i], p, trimbb(curr_bb, t.cd, t.point, i))
        self.__closest(t.child[j], p, trimbb(curr_bb, t.cd, t.point, j))

    def closest(self, p: tuple) -> tuple:
        """ Wrapper over the recursive helper function __closest. """
        self.best, self.best_dist = None, float("inf")
        bb = [[-float("inf"), float("inf")] for d in range(len(p))]
        self.__closest(self, p, [] if self.tight_bb else bb)
        return self.best

class kdTreeSort(kdNode):

    """ Thin wrapper over a kd-node to build a tree from a list of points.  """

    def __init__(self, points: list=[]) -> None:
        super().__init__()
        if len(points) > 0:
            # no need for duplicate points
            self.points, D = list(set(points)), len(points[0])
            # keep track of points via index in order to reduce copy size
            self.point_id = {point: i for i, point in enumerate(self.points)}
            # sort points on each dimension, running time dominated by the sort
            pointsd = [sorted(range(len(self.points)),
                       key=lambda p: self.points[p][d]) for d in range(D)]
            self.__build_tree(self, pointsd)

    def __splitd(self, pointsd: list, cd: int, p: int) -> tuple:
        """ Splits the list of points by the plane x_cd = p[cd]. """
        left, right, pi = set(), set(), self.point_id[p]
        for pid in pointsd[0]:
            if pid != pi:
                # add point with the same value as p at cd to the right side
                (left if self.points[pid][cd] < p[cd] else right).add(pid)
        return subsplit(pointsd, left), subsplit(pointsd, right)

    def __build_tree(self, t: kdNode, pointsd: list, cd: int=0) -> kdNode:
        """ Constructs a kd-tree in O(dn log n). """
        N, D, t.cd, t.tight_bb = len(pointsd[cd]), len(pointsd), cd, True
        t.point = self.points[pointsd[cd][N//2]] # median
        t.D, next_cd = D, (cd + 1) % D
        t.child = [self.__build_tree(kdNode(), l, next_cd) if len(l[0]) > 0
                   else None for l in self.__splitd(pointsd, cd, t.point)]
        # bounding box rectangle, tighter than tracking plane cuts
        t.bb = [[self.points[pointsd[d][0]][d], self.points[pointsd[d][-1]][d]]
                for d in range(D)]
        return t

class kdTree(kdNode):

    """ Thin wrapper over a kd-node to build a tree from a list of points.  """

    def __init__(self, points: list=[]) -> None:
        super().__init__()
        if len(points) > 0:
            # no need for duplicate points
            self.__build_tree(self, list(set(points)))
            self.tighten()

    def __build_tree(self, t: kdNode, points: list, cd: int=0) -> kdNode:
        """ Constructs a kd-tree in O(n log n). """
        N, D, t.cd = len(points), len(points[0]), cd
        t.point, points = median(points, cd) # median
        t.D, next_cd = D, (cd + 1) % D
        t.child = [self.__build_tree(kdNode(), l, next_cd) if len(l) > 0
                   else None for l in split(points, cd, t.point)]
        return t

if __name__ == "__main__":
    points = [(51, 75), (25, 40), (70, 70), (10, 30), (35, 90), (55, 1),
              (60, 80), (1, 10), (50, 50)]
    # t = kdTreeSort(points)
    # t = kdTree(points)
    # t = kdNode()
    # for point in points:
    #     t.insert(point)
    # print(t)
    # print(t.closest((20, 50)))
    # print(closest(points, (20, 50)))

    # exit()

    import random, time
    random.seed(1)

    N, K, D = 10**5, 10**5, 2
    points = [tuple(random.random() for d in range(D)) for k in range(K)]
    test = [tuple(random.random() for d in range(D)) for i in range(N)]

    tests = [1, 1, 0, 1, 0]

    ### sort first, O(dn log n)
    if tests[0]:
        s = time.time()
        t = kdTreeSort(points)
        print(f"Time to build (presort k-d tree): {time.time() - s:.3f}")

        s = time.time()
        for point in test:
            t.closest(point)
        print(f"  Time to run (presort k-d tree): {time.time() - s:.3f}")

    ### select median, O(n log n)
    if tests[1]:
        s = time.time()
        tm = kdTree(points)
        print(f"Time to build  (median k-d tree): {time.time() - s:.3f}")

        s = time.time()
        for point in test:
            tm.closest(point)
        print(f"  Time to run  (median k-d tree): {time.time() - s:.3f}")

    ### insert point by point, expected O(n log n) and O(n^2) worst case
    if tests[2]:
        s = time.time()
        t2 = kdNode()
        for point in points:
            t2.add(point)
        # t2.tighten()
        print(f"Time to build   (naive k-d tree): {time.time() - s:.3f}")

        s = time.time()
        for point in test:
            t2.closest(point)
        print(f"  Time to run   (naive k-d tree): {time.time() - s:.3f}")

    ### testing correctness

    if tests[3]:
        for point in test:
            ans = closest(points, point)
            if tests[0]: assert ans == t.closest(point)
            if tests[1]: assert ans == tm.closest(point)
            if tests[2]: assert ans == t2.closest(point)

    if tests[4]:
        s = time.time()
        for point in test:
            closest(points, point)
        print(f"  Time to run            (naive): {time.time() - s:.3f}")

