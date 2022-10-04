import random
random.seed(1)

### helper methods

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

def dist(u: tuple, v: tuple) -> float:
    """ Squared distance between two vectors. """
    return sum((x - y)*(x - y) for x, y in zip(u, v))

def closest(points: list, q: tuple) -> tuple:
    """ Returns the point closest to q in points (nearest neighbor query). """
    return min(points, key=lambda p: dist(p, q))

def centroid(points: list, weight: dict) -> tuple:
    """ Returns the centroid of a list of points. """
    denom, D = sum(weight[p] for p in points), len(points[0])
    return tuple(sum(weight[p]*p[d] for p in points)/denom for d in range(D))

### k-means

def init(K: int, points: list, weight: dict) -> list:
    """ Returns a list of K initial points with the k-means++ strategy. """
    # start with an arbitrary point
    l = [points[sample(norm([weight[p] for p in points]))]]
    for k in range(K - 1):
        i = sample(norm([dist(q, closest(l, q))*weight[q] for q in points]))
        l.append(points[i])
    return l

def k_means(K: int, points: list) -> tuple:
    """ Applies the k-means clustering algorithm on the data. """
    # weight points to remove redundant points while keeping centroids the same
    freq = {}
    for point in points:
        freq[point] = freq.get(point, 0) + 1
    centers = init(K, list(freq.keys()), freq)
    d = {center: i for i, center in enumerate(centers)}
    ids = {point: None for point in points}
    while True:
        # assign a point to its nearest center
        changed, groups = False, {k: [] for k in range(K)}
        for point in freq:
            old, ids[point] = ids[point], d[closest(centers, point)]
            changed |= ids[point] != old
            groups[ids[point]].append(point)
        if not changed:
            return centers, ids, groups
        # update center to the centroid of its associated group
        d = {}
        for k in range(K):
            centers[k] = centroid(groups[k], freq)
            d[centers[k]] = k

