import timeit

def median1(l: list) -> float:
    """ Computes the median of l, if len(l) == 5. """
    if l[0] < l[1]:
        if l[2] < l[3]:
            if l[0] < l[2]:
                if l[2] < l[4]:
                    if l[3] < l[4]:
                        if l[1] < l[3]:
                            if l[1] < l[2]:
                                return l[2]
                            else:
                                return l[1]
                        else:
                            return l[3]
                    else:
                        if l[1] < l[4]:
                            if l[1] < l[2]:
                                return l[2]
                            else:
                                return l[1]
                        else:
                            return l[4]
                else:
                    if l[0] < l[4]:
                        if l[1] < l[2]:
                            if l[1] < l[4]:
                                return l[4]
                            else:
                                return l[1]
                        else:
                            return l[2]
                    else:
                        if l[1] < l[2]:
                            return l[1]
                        else:
                            return l[2]
            else:
                if l[0] < l[4]:
                    if l[1] < l[4]:
                        if l[1] < l[3]:
                            return l[1]
                        else:
                            if l[0] < l[3]:
                                return l[3]
                            else:
                                return l[0]
                    else:
                        if l[3] < l[4]:
                            if l[0] < l[3]:
                                return l[3]
                            else:
                                return l[0]
                        else:
                            return l[4]
                else:
                    if l[0] < l[3]:
                        return l[0]
                    else:
                        if l[2] < l[4]:
                            if l[3] < l[4]:
                                return l[4]
                            else:
                                return l[3]
                        else:
                            return l[3]
        else:
            if l[0] < l[3]:
                if l[3] < l[4]:
                    if l[2] < l[4]:
                        if l[1] < l[2]:
                            if l[1] < l[3]:
                                return l[3]
                            else:
                                return l[1]
                        else:
                            return l[2]
                    else:
                        if l[1] < l[4]:
                            if l[1] < l[3]:
                                return l[3]
                            else:
                                return l[1]
                        else:
                            return l[4]
                else:
                    if l[0] < l[4]:
                        if l[1] < l[3]:
                            if l[1] < l[4]:
                                return l[4]
                            else:
                                return l[1]
                        else:
                            return l[3]
                    else:
                        if l[1] < l[2]:
                            if l[1] < l[3]:
                                return l[1]
                            else:
                                return l[3]
                        else:
                            return l[3]
            else:
                if l[0] < l[4]:
                    if l[1] < l[4]:
                        if l[1] < l[2]:
                            return l[1]
                        else:
                            if l[0] < l[2]:
                                return l[2]
                            else:
                                return l[0]
                    else:
                        if l[2] < l[4]:
                            if l[0] < l[2]:
                                return l[2]
                            else:
                                return l[0]
                        else:
                            return l[4]
                else:
                    if l[0] < l[2]:
                        return l[0]
                    else:
                        if l[2] < l[4]:
                            return l[4]
                        else:
                            return l[2]
    else:
        if l[2] < l[3]:
            if l[0] < l[3]:
                if l[0] < l[4]:
                    if l[0] < l[2]:
                        if l[2] < l[4]:
                            return l[2]
                        else:
                            return l[4]
                    else:
                        return l[0]
                else:
                    if l[1] < l[4]:
                        if l[2] < l[4]:
                            return l[4]
                        else:
                            if l[0] < l[2]:
                                return l[0]
                            else:
                                return l[2]
                    else:
                        if l[1] < l[2]:
                            if l[0] < l[2]:
                                return l[0]
                            else:
                                return l[2]
                        else:
                            return l[1]
            else:
                if l[3] < l[4]:
                    if l[0] < l[4]:
                        if l[1] < l[2]:
                            return l[3]
                        else:
                            if l[1] < l[3]:
                                return l[3]
                            else:
                                return l[1]
                    else:
                        if l[1] < l[3]:
                            return l[3]
                        else:
                            if l[1] < l[4]:
                                return l[1]
                            else:
                                return l[4]
                else:
                    if l[2] < l[4]:
                        if l[1] < l[4]:
                            return l[4]
                        else:
                            if l[1] < l[3]:
                                return l[1]
                            else:
                                return l[3]
                    else:
                        if l[1] < l[2]:
                            return l[2]
                        else:
                            if l[1] < l[3]:
                                return l[1]
                            else:
                                return l[3]
        else:
            if l[0] < l[2]:
                if l[0] < l[4]:
                    if l[0] < l[3]:
                        if l[2] < l[4]:
                            return l[3]
                        else:
                            if l[3] < l[4]:
                                return l[3]
                            else:
                                return l[4]
                    else:
                        return l[0]
                else:
                    if l[1] < l[4]:
                        if l[3] < l[4]:
                            return l[4]
                        else:
                            if l[0] < l[3]:
                                return l[0]
                            else:
                                return l[3]
                    else:
                        if l[1] < l[3]:
                            if l[0] < l[3]:
                                return l[0]
                            else:
                                return l[3]
                        else:
                            return l[1]
            else:
                if l[2] < l[4]:
                    if l[0] < l[4]:
                        if l[1] < l[2]:
                            return l[2]
                        else:
                            return l[1]
                    else:
                        if l[1] < l[2]:
                            return l[2]
                        else:
                            if l[1] < l[4]:
                                return l[1]
                            else:
                                return l[4]
                else:
                    if l[3] < l[4]:
                        if l[1] < l[4]:
                            return l[4]
                        else:
                            if l[1] < l[2]:
                                return l[1]
                            else:
                                return l[2]
                    else:
                        if l[1] < l[3]:
                            return l[3]
                        else:
                            if l[1] < l[2]:
                                return l[1]
                            else:
                                return l[2]

def median2(l: list) -> float:
    """ Computes the median of l, if len(l) == 5. """
    if l[0] < l[1]:
        if l[2] < l[3]:
            if l[0] < l[2]:
                if l[2] < l[4]:
                    if l[3] < l[4]:
                        if l[1] < l[3]:
                            return l[2] if l[1] < l[2] else l[1]
                        else:
                            return l[3]
                    else:
                        if l[1] < l[4]:
                            return l[2] if l[1] < l[2] else l[1]
                        else:
                            return l[4]
                else:
                    if l[0] < l[4]:
                        if l[1] < l[2]:
                            return l[4] if l[1] < l[4] else l[1]
                        else:
                            return l[2]
                    else:
                        return l[1] if l[1] < l[2] else l[2]
            else:
                if l[0] < l[4]:
                    if l[1] < l[4]:
                        if l[1] < l[3]:
                            return l[1]
                        else:
                            return l[3] if l[0] < l[3] else l[0]
                    else:
                        if l[3] < l[4]:
                            return l[3] if l[0] < l[3] else l[0]
                        else:
                            return l[4]
                else:
                    if l[0] < l[3]:
                        return l[0]
                    else:
                        if l[2] < l[4]:
                            return l[4] if l[3] < l[4] else l[3]
                        else:
                            return l[3]
        else:
            if l[0] < l[3]:
                if l[3] < l[4]:
                    if l[2] < l[4]:
                        if l[1] < l[2]:
                            return l[3] if l[1] < l[3] else l[1]
                        else:
                            return l[2]
                    else:
                        if l[1] < l[4]:
                            return l[3] if l[1] < l[3] else l[1]
                        else:
                            return l[4]
                else:
                    if l[0] < l[4]:
                        if l[1] < l[3]:
                            return l[4] if l[1] < l[4] else l[1]
                        else:
                            return l[3]
                    else:
                        if l[1] < l[2]:
                            return l[1] if l[1] < l[3] else l[3]
                        else:
                            return l[3]
            else:
                if l[0] < l[4]:
                    if l[1] < l[4]:
                        if l[1] < l[2]:
                            return l[1]
                        else:
                            return l[2] if l[0] < l[2] else l[0]
                    else:
                        if l[2] < l[4]:
                            return l[2] if l[0] < l[2] else l[0]
                        else:
                            return l[4]
                else:
                    if l[0] < l[2]:
                        return l[0]
                    else:
                        return l[4] if l[2] < l[4] else l[2]
    else:
        if l[2] < l[3]:
            if l[0] < l[3]:
                if l[0] < l[4]:
                    if l[0] < l[2]:
                        return l[2] if l[2] < l[4] else l[4]
                    else:
                        return l[0]
                else:
                    if l[1] < l[4]:
                        if l[2] < l[4]:
                            return l[4]
                        else:
                            return l[0] if l[0] < l[2] else l[2]
                    else:
                        if l[1] < l[2]:
                            return l[0] if l[0] < l[2] else l[2]
                        else:
                            return l[1]
            else:
                if l[3] < l[4]:
                    if l[0] < l[4]:
                        if l[1] < l[2]:
                            return l[3]
                        else:
                            return l[3] if l[1] < l[3] else l[1]
                    else:
                        if l[1] < l[3]:
                            return l[3]
                        else:
                            return l[1] if l[1] < l[4] else l[4]
                else:
                    if l[2] < l[4]:
                        if l[1] < l[4]:
                            return l[4]
                        else:
                            return l[1] if l[1] < l[3] else l[3]
                    else:
                        if l[1] < l[2]:
                            return l[2]
                        else:
                            return l[1] if l[1] < l[3] else l[3]
        else:
            if l[0] < l[2]:
                if l[0] < l[4]:
                    if l[0] < l[3]:
                        if l[2] < l[4]:
                            return l[3]
                        else:
                            return l[3] if l[3] < l[4] else l[4]
                    else:
                        return l[0]
                else:
                    if l[1] < l[4]:
                        if l[3] < l[4]:
                            return l[4]
                        else:
                            return l[0] if l[0] < l[3] else l[3]
                    else:
                        if l[1] < l[3]:
                            return l[0] if l[0] < l[3] else l[3]
                        else:
                            return l[1]
            else:
                if l[2] < l[4]:
                    if l[0] < l[4]:
                        return l[2] if l[1] < l[2] else l[1]
                    else:
                        if l[1] < l[2]:
                            return l[2]
                        else:
                            return l[1] if l[1] < l[4] else l[4]
                else:
                    if l[3] < l[4]:
                        if l[1] < l[4]:
                            return l[4]
                        else:
                            return l[1] if l[1] < l[2] else l[2]
                    else:
                        if l[1] < l[3]:
                            return l[3]
                        else:
                            return l[1] if l[1] < l[2] else l[2]

def median3(l: list) -> float:
    """ Computes the median of l, if len(l) == 5. """
    if l[0] < l[1]:
        if l[2] < l[3]:
            if l[0] < l[2]:
                if l[2] < l[4]:
                    if l[3] < l[4]:
                        if l[1] < l[3]:
                            return max(l[1], l[2])
                        else:
                            return l[3]
                    else:
                        if l[1] < l[4]:
                            return max(l[1], l[2])
                        else:
                            return l[4]
                else:
                    if l[0] < l[4]:
                        if l[1] < l[2]:
                            return max(l[1], l[4])
                        else:
                            return l[2]
                    else:
                        return min(l[1], l[2])
            else:
                if l[0] < l[4]:
                    if l[1] < l[4]:
                        if l[1] < l[3]:
                            return l[1]
                        else:
                            return max(l[0], l[3])
                    else:
                        if l[3] < l[4]:
                            return max(l[0], l[3])
                        else:
                            return l[4]
                else:
                    if l[0] < l[3]:
                        return l[0]
                    else:
                        if l[2] < l[4]:
                            return max(l[3], l[4])
                        else:
                            return l[3]
        else:
            if l[0] < l[3]:
                if l[3] < l[4]:
                    if l[2] < l[4]:
                        if l[1] < l[2]:
                            return max(l[1], l[3])
                        else:
                            return l[2]
                    else:
                        if l[1] < l[4]:
                            return max(l[1], l[3])
                        else:
                            return l[4]
                else:
                    if l[0] < l[4]:
                        if l[1] < l[3]:
                            return max(l[1], l[4])
                        else:
                            return l[3]
                    else:
                        if l[1] < l[2]:
                            return min(l[1], l[3])
                        else:
                            return l[3]
            else:
                if l[0] < l[4]:
                    if l[1] < l[4]:
                        if l[1] < l[2]:
                            return l[1]
                        else:
                            return max(l[0], l[2])
                    else:
                        if l[2] < l[4]:
                            return max(l[0], l[2])
                        else:
                            return l[4]
                else:
                    if l[0] < l[2]:
                        return l[0]
                    else:
                        return max(l[2], l[4])
    else:
        if l[2] < l[3]:
            if l[0] < l[3]:
                if l[0] < l[4]:
                    if l[0] < l[2]:
                        return min(l[2], l[4])
                    else:
                        return l[0]
                else:
                    if l[1] < l[4]:
                        if l[2] < l[4]:
                            return l[4]
                        else:
                            return min(l[0], l[2])
                    else:
                        if l[1] < l[2]:
                            return min(l[0], l[2])
                        else:
                            return l[1]
            else:
                if l[3] < l[4]:
                    if l[0] < l[4]:
                        if l[1] < l[2]:
                            return l[3]
                        else:
                            return max(l[1], l[3])
                    else:
                        if l[1] < l[3]:
                            return l[3]
                        else:
                            return min(l[1], l[4])
                else:
                    if l[2] < l[4]:
                        if l[1] < l[4]:
                            return l[4]
                        else:
                            return min(l[1], l[3])
                    else:
                        if l[1] < l[2]:
                            return l[2]
                        else:
                            return min(l[1], l[3])
        else:
            if l[0] < l[2]:
                if l[0] < l[4]:
                    if l[0] < l[3]:
                        if l[2] < l[4]:
                            return l[3]
                        else:
                            return min(l[3], l[4])
                    else:
                        return l[0]
                else:
                    if l[1] < l[4]:
                        if l[3] < l[4]:
                            return l[4]
                        else:
                            return min(l[0], l[3])
                    else:
                        if l[1] < l[3]:
                            return min(l[0], l[3])
                        else:
                            return l[1]
            else:
                if l[2] < l[4]:
                    if l[0] < l[4]:
                        return max(l[1], l[2])
                    else:
                        if l[1] < l[2]:
                            return l[2]
                        else:
                            return min(l[1], l[4])
                else:
                    if l[3] < l[4]:
                        if l[1] < l[4]:
                            return l[4]
                        else:
                            return min(l[1], l[2])
                    else:
                        if l[1] < l[3]:
                            return l[3]
                        else:
                            return min(l[1], l[2])

def median4(l: list) -> float:
    """ Computes the median of l, if len(l) == 5. """
    return (((((((l[2] if l[1] < l[2] else l[1]) if l[1] < l[3] else l[3]) if l[3] < l[4] else ((l[2] if l[1] < l[2] else l[1]) if l[1] < l[4] else l[4])) if l[2] < l[4] else (((l[4] if l[1] < l[4] else l[1]) if l[1] < l[2] else l[2]) if l[0] < l[4] else (l[1] if l[1] < l[2] else l[2]))) if l[0] < l[2] else (((l[1] if l[1] < l[3] else (l[3] if l[0] < l[3] else l[0])) if l[1] < l[4] else ((l[3] if l[0] < l[3] else l[0]) if l[3] < l[4] else l[4])) if l[0] < l[4] else (l[0] if l[0] < l[3] else ((l[4] if l[3] < l[4] else l[3]) if l[2] < l[4] else l[3])))) if l[2] < l[3] else (((((l[3] if l[1] < l[3] else l[1]) if l[1] < l[2] else l[2]) if l[2] < l[4] else ((l[3] if l[1] < l[3] else l[1]) if l[1] < l[4] else l[4])) if l[3] < l[4] else (((l[4] if l[1] < l[4] else l[1]) if l[1] < l[3] else l[3]) if l[0] < l[4] else ((l[1] if l[1] < l[3] else l[3]) if l[1] < l[2] else l[3]))) if l[0] < l[3] else (((l[1] if l[1] < l[2] else (l[2] if l[0] < l[2] else l[0])) if l[1] < l[4] else ((l[2] if l[0] < l[2] else l[0]) if l[2] < l[4] else l[4])) if l[0] < l[4] else (l[0] if l[0] < l[2] else (l[4] if l[2] < l[4] else l[2]))))) if l[0] < l[1] else (((((l[2] if l[2] < l[4] else l[4]) if l[0] < l[2] else l[0]) if l[0] < l[4] else ((l[4] if l[2] < l[4] else (l[0] if l[0] < l[2] else l[2])) if l[1] < l[4] else ((l[0] if l[0] < l[2] else l[2]) if l[1] < l[2] else l[1]))) if l[0] < l[3] else (((l[3] if l[1] < l[2] else (l[3] if l[1] < l[3] else l[1])) if l[0] < l[4] else (l[3] if l[1] < l[3] else (l[1] if l[1] < l[4] else l[4]))) if l[3] < l[4] else ((l[4] if l[1] < l[4] else (l[1] if l[1] < l[3] else l[3])) if l[2] < l[4] else (l[2] if l[1] < l[2] else (l[1] if l[1] < l[3] else l[3]))))) if l[2] < l[3] else ((((l[3] if l[2] < l[4] else (l[3] if l[3] < l[4] else l[4])) if l[0] < l[3] else l[0]) if l[0] < l[4] else ((l[4] if l[3] < l[4] else (l[0] if l[0] < l[3] else l[3])) if l[1] < l[4] else ((l[0] if l[0] < l[3] else l[3]) if l[1] < l[3] else l[1]))) if l[0] < l[2] else (((l[2] if l[1] < l[2] else l[1]) if l[0] < l[4] else (l[2] if l[1] < l[2] else (l[1] if l[1] < l[4] else l[4]))) if l[2] < l[4] else ((l[4] if l[1] < l[4] else (l[1] if l[1] < l[2] else l[2])) if l[3] < l[4] else (l[3] if l[1] < l[3] else (l[1] if l[1] < l[2] else l[2])))))))

def median4(l: list) -> float:
    """ Computes the median of l, if len(l) == 5. """
    a, b, c, d, e = l
    return (((((((c if b < c else b) if b < d else d) if d < e else ((c if b < c else b) if b < e else e)) if c < e else (((e if b < e else b) if b < c else c) if a < e else (b if b < c else c))) if a < c else (((b if b < d else (d if a < d else a)) if b < e else ((d if a < d else a) if d < e else e)) if a < e else (a if a < d else ((e if d < e else d) if c < e else d)))) if c < d else (((((d if b < d else b) if b < c else c) if c < e else ((d if b < d else b) if b < e else e)) if d < e else (((e if b < e else b) if b < d else d) if a < e else ((b if b < d else d) if b < c else d))) if a < d else (((b if b < c else (c if a < c else a)) if b < e else ((c if a < c else a) if c < e else e)) if a < e else (a if a < c else (e if c < e else c))))) if a < b else (((((c if c < e else e) if a < c else a) if a < e else ((e if c < e else (a if a < c else c)) if b < e else ((a if a < c else c) if b < c else b))) if a < d else (((d if b < c else (d if b < d else b)) if a < e else (d if b < d else (b if b < e else e))) if d < e else ((e if b < e else (b if b < d else d)) if c < e else (c if b < c else (b if b < d else d))))) if c < d else ((((d if c < e else (d if d < e else e)) if a < d else a) if a < e else ((e if d < e else (a if a < d else d)) if b < e else ((a if a < d else d) if b < d else b))) if a < c else (((c if b < c else b) if a < e else (c if b < c else (b if b < e else e))) if c < e else ((e if b < e else (b if b < c else c)) if d < e else (d if b < d else (b if b < c else c)))))))

def median5(l: list) -> float:
    """ Computes the median of l, if len(l) == 5. """
    return sorted(l)[len(l)//2]

def median6(l: list) -> float:
    """ Computes the median of l, if len(l) == 5. """
    l.sort()
    return l[len(l)//2]

if __name__ == "__main__":
    import itertools
    N = 5
    l = list(range(N))
    perms = list(itertools.permutations(l))
    for l in perms: # check correctness
        assert median1(l) == median2(l) == median3(l) == median4(l) == median5(l)

    for i, name in enumerate(["if/else", "leaf", "max", "ternary", "sort", "sort2"]):
        t = timeit.timeit(f"median{i + 1}([3, 1, 4, 2, 5])", globals=globals())
        print(f"{name:>7}: {t:.6f}")

