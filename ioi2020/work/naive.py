N, M, Q = map(int, input().split())

# force matricies
forcex, forcey = [[0]*M for i in range(N)], [[0]*M for i in range(N)]
for i in range(N):
    line = list(map(int, input().split()))
    for j in range(0, 2*M, 2):
        forcex[i][j//2] = line[j]
        forcey[i][j//2] = line[j + 1]

# direction lookup
dirs = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
work = 0

for i in range(Q):
    line = input().split()
    a, b, line = int(line[0]), int(line[1]), line[2:]
    path = 0
    for j in range(0, len(line), 2):
        # vector in magnitude-direction notation
        mag, dir = int(line[j]), dirs[line[j + 1]]
        for k in range(mag):
            # dot product
            path += dir[0]*forcey[a][b] + dir[1]*forcex[a][b]
            # vector addition
            a, b = a + dir[0], b + dir[1]
    work += path

print(work)
