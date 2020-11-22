# modulo
M = 1000000007

def dotp(u: list, v: list) -> float:
    """ dot product """
    return sum(u[i]*v[i] for i in range(len(u)))

def col(m: list, i: int) -> list:
    """ column of a matrix """
    return [m[j][i] for j in range(len(m))]

def mod_mat_mult(a: list, b: list, m: int) -> list:
    """ matrix multiplication with modulo """
    return [[dotp(a[i], col(b, j)) % m for j in range(len(b[0]))] for i in range(len(a))]

def identity(n: int) -> list:
    """ identity matrix of size nxn """
    return [[int(i == j) for j in range(n)] for i in range(n)]

def mod_mat_exp(A: list, k: int, m: int) -> list:
    """ returns A^k % m"""
    v = identity(len(A))
    while k > 0:
        if k & 1 == 1:
            v = mod_mat_mult(v, A, m)
        k >>= 1
        A = mod_mat_mult(A, A, m)
    return v

def mod_seq(a: int, b: int, n: int, m: int) -> int:
    """ returns the nth sequence element mod m"""
    A = [[a, b],
         [1, 0]
        ]
    return mod_mat_exp(A, n - 1, m)[0][0] if n > 0 else 0

A, B, N = map(int, input().split())
print(mod_seq(A, B, N, M))
