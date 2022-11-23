import pickle, random

### Hog game

# score needed to win
GOAL_SCORE = 100
# max number of dice the player can roll, ignore rolling 1, sides of the dice 
K, START, SIDES = 10, 2, 6
INF = float("inf")
# Dynamic programming approach to compute the probability mass function. 
dp = [0]*(6*K + 1)
for i in range(START, SIDES + 1):
    dp[i] += 1

pmf_matrix = [dp]
for i in range(K - 1):
    temp = [0]*(6*K + 1)
    for j in range(len(dp) - 6):
        for v in range(START, SIDES + 1):
            temp[j + v] += dp[j]
    dp = temp
    pmf_matrix.append(list(dp))

def free_bacon(score: int) -> int:
    return 10 - (score % 10) + score//10

def is_swap(player_score: int, opponent_score: int) -> bool:
    tens_digit = (opponent_score % 100)//10
    return abs((player_score % 10) - (opponent_score % 10)) == tens_digit

def is_feral(delta: int, k: int) -> bool: 
    return abs(delta - k) == 2

### Recursive

def state(score: int, opponent_score: int, delta: int) -> tuple:
    new, opp = score + delta, opponent_score
    return (opp, new) if not is_swap(new, opp) else (new, opp)

def recur2(f, score, opponent_score, delta):
    return 1 - f(*state(score, opponent_score, delta))[1]

def recur5(f, score, opponent_score, turn, last_delta, prev_delta, delta, roll):
    return 1 - f(*state(score, opponent_score, delta + 3*is_feral(last_delta, roll)), 
                 turn ^ 1, prev_delta, delta if delta <= 12 else INF)[1]

def expected_value(recur, k, *args) -> float:
    pmf, denom = pmf_matrix[k - 1], 5**k
    return (1 - (5/6)**k)*recur(1, *args) + \
               ((5/6)**k)*sum(recur(d, *args)*pmf[d] for d in range(2*k, 6*k + 1))/denom  

def strategy2(recur, opponent_score) -> list:
    return [(0, recur(free_bacon(opponent_score)))] + \
           [(k, expected_value(recur, k)) for k in range(1, K + 1)]

def strategy5(recur, opponent_score) -> list:
    return [(0, recur(free_bacon(opponent_score), 0))] + \
           [(k, expected_value(recur, k, k)) for k in range(1, K + 1)]

def best_move(strategy: list) -> tuple:
    return max(strategy, key=lambda x: (x[1], -x[0])) 

### Pickle methods

def load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

def dump(fname, obj):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

