N, M, Q = map(int, input().split())

# force matricies
forcex, forcey = [[0]*M for i in range(N)], [[0]*M for i in range(N)]
for i in range(N):
    line = list(map(int, input().split()))
    for j in range(0, 2*M, 2):
        forcex[i][j//2] = line[j]
        forcey[i][j//2] = line[j + 1]

# prefix sums in x and y directions
def query(prefix, x1, y1, x2, y2):
    return prefix[x2 + 1][y2 + 1] - prefix[x2 + 1][y1] - prefix[x1][y2 + 1] + prefix[x1][y1]

prefixx, prefixy = [[0]*(M + 1) for i in range(N + 1)], [[0]*(M + 1) for i in range(N + 1)]

for i in range(1, len(prefixx)):
    for j in range(1, len(prefixx[0])):
        prefixx[i][j] += prefixx[i - 1][j] + forcex[i - 1][j - 1]

for i in range(1, len(prefixx)):
    for j in range(1, len(prefixx[0])):
        prefixx[i][j] += prefixx[i][j - 1]

for i in range(1, len(prefixy)):
    for j in range(len(prefixy[0])):
        prefixy[i][j] += prefixy[i - 1][j] + forcey[i - 1][j - 1]

for i in range(1, len(prefixy)):
    for j in range(1, len(prefixy[0])):
        prefixy[i][j] += prefixy[i][j - 1]

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
        # use prefix sums to compute dot product over the entire vector
        mag -= 1
        path += query(prefixy, a, b, a + mag*dir[0], b + mag*dir[1])*dir[0] + \
                query(prefixx, a, b, a + mag*dir[0], b + mag*dir[1])*dir[1]
        # vector addition
        mag += 1
        a, b = a + mag*dir[0], b + mag*dir[1]
    work += path

print(work)
