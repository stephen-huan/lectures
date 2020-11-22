# for i in range(N):
#     for j in range(M):
#         print(f"({forcex[i][j]}, {forcey[i][j]})", end=" ")
#     print()

"""
for i in range(Q):
    line = input().split()
    a, b, line = int(line[0]), int(line[1]), line[2:]
    path = 0
    for j in range(0, len(line), 2):
        # vector in magnitude-direction notation
        mag, dir = int(line[j]), dirs[line[j + 1]]
        for k in range(mag):
            # dot product
            path += dir[0]*forcex[a][b] + dir[1]*forcey[a][b]
            # vector addition
            a, b = a + dir[0], b + dir[1]
    print(path)
    work += path

print(work)
"""

# for i in range(Q):
#     line = input().split()
#     a, b, line = int(line[0]), int(line[1]), line[2:]
#     path = 0
#     for j in range(0, len(line), 2):
#         # vector in magnitude-direction notation
#         mag, dir = int(line[j]), dirs[line[j + 1]]
#         for k in range(mag):
#             # dot product
#             print((forcex[a][b], forcey[a][b]), dir[0]*forcey[a][b] + dir[1]*forcex[a][b])
#             path += dir[0]*forcey[a][b] + dir[1]*forcex[a][b]
#             # vector addition
#             a, b = a + dir[0], b + dir[1]
#     print(path)
#     print()
#     work += path
#
# print(work)

""" unfortunately the discrete gradient does not work
# precompute the gradient to the vector field, such that
# grad f = F = <x, 2y> (where grad f = <df/dx, df/dy>)
# we know the vector field is conservative so the gradient exists
grad = [[0]*M for i in range(N)]
for y in range(1, N):
    for x in range(1, M):
        # propagate differential in the x direction
        grad[y][x] = x - 1 + grad[y][x - 1]
    if y < N - 1:
        # propagate differential in the y direction
        grad[y + 1][0] = 2*y + grad[y][0]

for i in range(Q):
    line = input().split()
    a, b, line = int(line[0]), int(line[1]), line[2:]
    # endpoint
    c, d = a, b
    for j in range(0, len(line), 2):
        # vector in magnitude-direction notation
        mag, dir = int(line[j]), dirs[line[j + 1]]
        # vector addition
        c, d = c + mag*dir[0], d + mag*dir[1]
    # line integral^b_a of F dr = f(r(b)) - f(r(a))
    print(grad[c][d] - grad[a][b])
    work += grad[c][d] - grad[a][b]
"""
