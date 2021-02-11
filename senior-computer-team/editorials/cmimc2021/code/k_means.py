import random, heapq
random.seed(1)

INIT = 0       # 0 - random sample, 1 - k-means:), 2 - k-means++
CONT = True    # whether to continue trying to prune centers
PSUEDO = True  # psuedo-owners, allow <= cases
KDTREE = False # use kd-tree for leaves
F = 4          # time penalty for kd-tree traversal
APPROX = False # whether to use the approximation algorithm
B = 0.8        # constant for approximation algorithm

### BIT to maintain CMF

class BIT:

    def __init__(self, n) -> None:
        if isinstance(n, list):
            self.l = [0]*(len(n) + 1)
            for i in range(len(n)):
                self.update(i, n[i])
        else:
            self.l = [0]*(n + 1)

    def __str__(self) -> str:
        return str(self.l)

    def query(self, i: int) -> float:
        """ sum of elements up to (and including) i """
        i += 1
        ans = 0
        while i > 0:
            ans += self.l[i]
            i -= (i & -i)
        return ans

    def range(self, i: int, j: int) -> float:
        """ sum of elements between i and j, inclusive on both ends """
        return self.query(j) - (self.query(i - 1) if i > 0 else 0)

    def update(self, i: int, v: float) -> None:
        """ add v to the index at i """
        i += 1
        while i < len(self.l):
            self.l[i] += v
            i += (i & -i)

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
    return sorted(l)[len(l)//2]

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
    l = [point[cd] for point in points]
    m = select(l, len(l)//2)
    for i in range(len(points)):
        if points[i][cd] == m:
            break
    return points[i], points[:i] + points[i + 1:]

### Helper methods

def norm(l: list) -> list:
    """ Normalizes a list into a pmf by dividing by its sum. """
    s = sum(l)
    return [x/s for x in l]

def sample(p: list) -> float:
    """ Samples a value from a random variable. """
    r = random.random()
    i = cmf = 0
    while i < len(p) - 1:
        cmf += p[i]
        if r < cmf:
            break
        i += 1
    return i

def dist(p1: tuple, p2: tuple) -> float:
    """ Squared distance between two points."""
    return (p1[0] - p2[0])*(p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1])
    return sum((p1[i] - p2[i])*(p1[i] - p2[i]) for i in range(len(p1)))

def closest(points: list, q: tuple) -> tuple:
    """ Returns the point closest to q in points (nearest neighbor query). """
    return min(points, key=lambda p: dist(p, q))

def centroid(points: list, weight: dict) -> tuple:
    """ Returns the centroid of a list of points. """
    denom, D = sum(weight[p] for p in points), len(points[0])
    return tuple(sum(weight[p]*p[d] for p in points)/denom for d in range(D))

def mix(pairs: list) -> list:
    """ Takes a list of vector, weight pairs and returns a final centroid. """
    D, denom = len(pairs[0][0]), sum(list(zip(*pairs))[1])
    return tuple(sum(u[d]*w for u, w in pairs)/denom for d in range(D))

def distbb(p: tuple, bb: list) -> float:
    """ Squared distance between a point and a bounding box. """
    # three cases, use x if x is in the box, otherwise one of the bounds
    bbp = tuple(box[0] if x < box[0] else (box[1] if x > box[1] else x)
                for x, box in zip(p, bb))
    return dist(p, bbp)

### geometric reasoning methods

def dominate(c1: tuple, c2: tuple, bb: list) -> bool:
    """ Does c1 dominate c2 with respect to bb?
    (every point in bb is closer to c1 than it is to c2). """
    p = tuple(box[0] if x2 < x1 else box[1] for x1, x2, box in zip(c1, c2, bb))
    d1, d2 = dist(p, c1), dist(p, c2)
    return d1 < d2 or (PSUEDO and d1 == d2)

### k-d tree

def split(points: list, cd: int, p: int) -> tuple:
    """ Splits the list of points by the plane x_cd = p[cd]. """
    left, right = [], []
    for point in points:
        # add point with the same value as p at cd to the right side
        (left if point[cd] < p[cd] else right).append(point)
    return left, right

def is_leaf(n) -> bool:
    return n.child[0] is None and n.child[1] is None

class kdNode:

    """ kd-tree node. """

    def __init__(self, point: tuple=None, cd: int=0) -> None:
        self.child, self.point, self.cd, self.bb = [None, None], point, cd, []
        self.D, self.tight_bb = len(point) if point is not None else 0, False

    def tighten(self, t: "KdNode"=None) -> None:
        """ Tighten bounding boxes in O(nd). """
        if t is None: t = self # called with None, set to the root 
        l, r, t.tight_bb = t.child[0], t.child[1], True
        # recur on children
        if l is not None: self.tighten(l)
        if r is not None: self.tighten(r)

        t.num = self.weight[t.point]
        if l is None and r is None:  # leaf node, box is just the singular point
            t.bb = [(t.point[d], t.point[d]) for d in range(t.D)]
            t.centroid = t.point
        elif l is None or r is None: # one child, inherit box of child
            child = l if l is not None else r
            t.bb = child.bb
            t.bb = [(min(box[0], v), max(box[1], v))  # add node's point
                    for box, v in zip(t.bb, t.point)]
            t.centroid = mix([(t.point, t.num), (child.centroid, child.num)])
            t.num += child.num
        else:                        # two children, combine boxes
            t.bb = [(min(bbl[0], bbr[0], v), max(bbl[1], bbr[1], v))
                    for bbl, bbr, v in zip(l.bb, r.bb, t.point)]
            pairs = [(t.point, t.num), (l.centroid, l.num), (r.centroid, r.num)]
            t.centroid = mix(pairs)
            t.num += l.num + r.num

    def __closest(self, t: "kdNode", p: tuple) -> tuple:
        """ Returns the closest point to p in the tree (nearest neighbor). """
        # all points in this bounding box farther than existing point
        if t is None or distbb(p, t.bb) > self.best_dist:
            return
        # update best point
        d = dist(p, t.point)
        if d < self.best_dist:
            self.best, self.best_dist = t.point, d
        # visit subtrees in order of distance from p
        i = p[t.cd] >= t.point[t.cd]
        self.__closest(t.child[i], p)
        self.__closest(t.child[1 - i], p)

    def closest(self, p: tuple) -> tuple:
        """ Wrapper over the recursive helper function __closest. """
        self.best, self.best_dist = None, float("inf")
        self.__closest(self, p)
        return self.best

    def dist(self, p: tuple) -> tuple:
        """ Wrapper over the recursive helper function __closest. """
        self.best, self.best_dist = None, float("inf")
        self.__closest(self, p)
        return self.best_dist

    def nnsearch(self, p: tuple, r: float) -> tuple:
        """ Finds the points that are within a radius of r from p. """
        l = []
        h = [(0, 0)]
        ids = {0: self}
        i = 1
        while len(h) > 0:
            d, ni = heapq.heappop(h)
            n = ids[ni]
            if d > r*r: # stop processing if out of circle
                continue
            if is_leaf(n):
                l.append(n.point)
            else:
                for child in n.child + [kdNode(n.point)]:
                    if child is not None:
                        d = dist(p, child.point) if is_leaf(child) else \
                            distbb(p, child.bb)
                        ids[i] = child
                        heapq.heappush(h, (d, i))
                        i += 1
        return l

class kdTree(kdNode):

    """ Thin wrapper over a kd-node to build a tree from a list of points.  """

    def __init__(self, points: list=[], weight: dict={}):
        super().__init__()
        self.weight = weight if len(weight) != 0 else \
            {point: 1 for point in points}
        if len(points) > 0:
            # no need for duplicate points
            self.__build_tree(self, points)
            self.tighten()
        self.points, self.cmf = points, BIT(len(points))
        self.point_id = {point: i for i, point in enumerate(points)}
        self.l = [0]*len(points)

    def __build_tree(self, t: kdNode, points: list, cd: int=0) -> kdNode:
        """ Constructs a kd-tree in O(n log n). """
        N, D, t.cd = len(points), len(points[0]), cd
        t.point, points = median(points, cd) # median
        t.D, next_cd = D, (cd + 1) % D
        t.child = [self.__build_tree(kdNode(), l, next_cd) if len(l) > 0
                   else None for l in split(points, cd, t.point)]
        return t

    def __prune(self, t: kdNode) -> bool:
        """ Whether to prune at this current node using a heuristic. """
        return t.num*sum(((b1[1] - b1[0])/(b2[1] - b2[0]))**2
                         for b1, b2 in zip(t.bb, self.bb)) <= pow(B, self.i)

    def __close(self, centers: list) -> None:
        """ Set up paramaters for kd-tree closest search. """
        self.t = kdTree(centers)
        # cost roughly O(log n + 2^d)
        self.cost = F*((len(centers) - 1).bit_length() + (1 << len(centers[0])))

    def close(self, centers: list, p: tuple) -> tuple:
        """ Finds the closest point with a mix of brute force and kd-tree. """
        if KDTREE and len(centers) >= self.cost:
            return self.t.closest(p)
        return closest(centers, p)

    def __update(self, t: kdNode, centers: list, d: dict, cont: bool) -> None:
        """ Updates each point to its closest center. """
        l, r = t.child
        if l is None and r is None: # leaf
            d[self.close(centers, t.point)].append((t.centroid, t.num))
            return
        dists = [distbb(c, t.bb) for c in centers]
        i = min(range(len(centers)), key=lambda i: dists[i])
        c, dis = centers[i], dists[i]
        if PSUEDO or dists.count(dis) == 1:  # c is uniquely the smallest
            poss, new_centers = [c], centers[:i] + centers[i + 1:]
            for j in range(len(new_centers)):
                if not dominate(c, new_centers[j], t.bb):
                    poss.append(new_centers[j])
                    if not cont:   # whether to continue and prune more centers
                        poss += new_centers[j + 1:]
                        break
            if len(poss) == 1:     # c dominates all other centers
                d[c].append((t.centroid, t.num))
                return
            centers = poss         # blacklist dominated centers
        if APPROX and self.__prune(t):
            for c in centers:      # split points evenly amongst centers 
                d[c].append((t.centroid, t.num/len(centers)))
            return
        # since our tree has points on intermediate nodes, handle
        d[self.close(centers, t.point)].append((t.point, self.weight[t.point]))
        # recur on children
        if l is not None: self.__update(l, centers, d, cont)
        if r is not None: self.__update(r, centers, d, cont)

    def update(self, centers: list, cont: bool=CONT) -> list:
        """ Wrapper over the recursive helper function __update. """
        if KDTREE: self.__close(centers)
        d = {c: [] for c in centers}
        self.__update(self, centers, d, cont)
        return [mix(pairs) for c, pairs in d.items()]

    def __get_children(self, t: kdNode, children: list) -> list:
        """ Recursively gets the children of a kd-tree. """
        l, r = t.child
        children.append(t.point)
        if l is not None: self.__get_children(l, children)
        if r is not None: self.__get_children(r, children)
        return children

    def get_children(self, t: kdNode) -> list:
        """ Wrapper function. """
        return self.__get_children(t, [])

    def __update_point(self, ids: dict, point: tuple, center: tuple) -> None:
        """ Updates the probability distribition of a point. """
        ids[point] = self.d[center]
        i, w = self.point_id[point], self.weight[point]
        temp, self.l[i] = self.l[i], w*dist(point, center)
        self.cmf.update(i, self.l[i] - temp)
        # self.cmf.update(i, w*dist(point, center) - self.cmf.range(i, i))

    def __ids(self, t: kdNode, centers: list, ids: dict, cont: bool) -> None:
        """ Finds the mapping of each point to its corresponding center. """
        l, r = t.child
        if l is None and r is None: # leaf
            self.__update_point(ids, t.point, self.close(centers, t.point))
            return
        dists = [distbb(c, t.bb) for c in centers]
        i = min(range(len(centers)), key=lambda i: dists[i])
        c, dis = centers[i], dists[i]
        if PSUEDO or dists.count(dis) == 1:  # c is uniquely the smallest
            poss, new_centers = [c], centers[:i] + centers[i + 1:]
            for j in range(len(new_centers)):
                if not dominate(c, new_centers[j], t.bb):
                    poss.append(new_centers[j])
                    if not cont:   # whether to continue and prune more centers
                        poss += new_centers[j + 1:]
                        break
            if len(poss) == 1:     # c dominates all other centers
                if ids.get(t.point, None) != self.d[c]: # changed
                    for point in self.get_children(t):  # slow but necessary
                        self.__update_point(ids, point, c)
                return
            centers = poss         # blacklist dominated centers
        # since our tree has points on intermediate nodes, handle
        self.__update_point(ids, t.point, self.close(centers, t.point))
        # recur on children
        if l is not None: self.__ids(l, centers, ids, cont)
        if r is not None: self.__ids(r, centers, ids, cont)

    def metadata(self, centers: list, ids: dict=None, cont: bool=CONT) -> tuple:
        """ Returns a mapping of point to its closest center. """
        if KDTREE: self.__close(centers)
        self.d = {c: i for i, c in enumerate(centers)}
        ids = ids if ids is not None else {}
        self.__ids(self, centers, ids, cont)
        return ids

    def sample(self) -> tuple:
        """ Samples a point according to k-means++. """
        p = random.random()
        denom = self.cmf.query(len(self.points) - 1) # sum of all values
        l, r = 0, len(self.points)
        while l < r:
            m = (l + r)>>1
            if self.cmf.query(m)/denom <= p:
                l, r = m + 1, r
            else:
                l, r = l, m
        return self.points[l]

### k-means

def init(K: int, points: list, t: kdTree) -> list:
    """ Returns a list of K initial points with the k-means++ strategy. """
    if INIT == 0:
        return random.sample(points, K)
    # start with an arbitrary point
    l = [points[sample(norm([1 for p in points]))]]
    ids = t.metadata(l)
    if INIT == 1:
        for k in range(K - 1):
            l.append(t.sample())
            ids = t.metadata(l, ids)
    elif INIT == 2:
        for k in range(K - 1):
            d = lambda p: dist(p, l[ids[p]])
            l.append(points[sample(norm([d(q) for q in points]))])
            ids = t.metadata(l, ids)
    return l

def k_means(K: int, points: list) -> tuple:
    """ Applies the k-means clustering algorithm on the data. """
    # weight points to remove redundant points while keeping centroids the same
    freq = {}
    for point in points:
        freq[point] = freq.get(point, 0) + 1
    points = list(freq)
    # kd-tree on the points, only needs to be built once
    t = kdTree(points, freq)
    t.i = 1 # iteration number for the heuristic
    old = sorted(init(K, points, freq, t))
    centers = sorted(t.update(old))
    while centers != old: # no good way of checking convergene
        t.i += 1
        centers, old = sorted(t.update(centers)), centers
    # generate metadata 
    ids = t.metadata(centers)
    groups = {k: [] for k in range(K)}
    for point, k in ids.items():
        groups[k].append(point)
    return centers, ids, groups

