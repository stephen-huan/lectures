import pickle, random
import matplotlib.pyplot as plt

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


fig, ax = plt.subplots()

for k in range(1, 11):
    x = range(2*k, 6*k + 1)
    plt.hist(x, bins=4*k + 1, weights=pmf_matrix[k - 1][2*k: 6*k + 1], density=True, label=f"{k}")

ax.legend()
plt.title(f"Probability mass function of rolling $k$ dice")
plt.xlabel('Sum of rolls')
plt.ylabel('Probability')
plt.savefig(f"pmf.png")

