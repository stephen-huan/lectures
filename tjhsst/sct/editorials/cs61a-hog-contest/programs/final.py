from functools import lru_cache
from hoglib import *

dp_matrix = {(i, j, b): {} for i in range(GOAL_SCORE) for j in range(GOAL_SCORE) for b in range(2)}
# dp_matrix = load("dp_matrix.pickle")
prob = load("perfect_play_count.pickle")
prob_backup = load("self_play_count.pickle")

# chance of opponent making a mistake
EPSILON = 0.01

@lru_cache(maxsize=None)
def ignore_feral_dp(score, opponent_score):
    # base case: if the game is over, return whether I win or not
    if score >= GOAL_SCORE or opponent_score >= GOAL_SCORE:
        return (None, score >= GOAL_SCORE)

    recur = lambda delta: recur2(ignore_feral_dp, score, opponent_score, delta)
    return best_move(strategy2(recur, opponent_score))

def ignore_feral_strategy(score, opponent_score):
    return ignore_feral_dp(score, opponent_score)[0]

def perfect_dp(score, opponent_score, turn, last_delta, prev_delta):
    if score >= GOAL_SCORE or opponent_score >= GOAL_SCORE:
        return (None, score >= GOAL_SCORE)

    prefix = dp_matrix[(score, opponent_score, turn)]
    key = (last_delta, prev_delta)
    if key in prefix:
        return prefix[key] 

    state = (score, opponent_score, turn, last_delta, prev_delta)
    recur = lambda delta, roll: recur5(perfect_dp, *state, delta, roll)
    prefix[key] = best_move(strategy5(recur, opponent_score))
    return prefix[key]

def perfect_strategy(*args):
    return perfect_dp(*args)[0]

@lru_cache(maxsize=None)
def final_strategy_dp(score, opponent_score, turn=False):
    if score >= GOAL_SCORE or opponent_score >= GOAL_SCORE:
        return (None, score >= GOAL_SCORE)

    def recur(delta, feral_bonus):
        return 1 - final_strategy_dp(*state(delta + feral_bonus), turn ^ 1)[1]

    def expected_value(k, feral_bonus):
        pmf, denom = pmf_matrix[k - 1], 5**k
        return (1 - (5/6)**k)*recur(1, feral_bonus) + \
                   ((5/6)**k)*sum(recur(d, feral_bonus)*pmf[d] for d in range(2*k, 6*k + 1))/denom  

    def feral_value(f, k, v):
        p = 0.5*count0.get(k, 0)/denom0 + 0.5*count1.get(k, 0)/denom1
        return p*(f(v, 3) if p > 0 else 0) + (1 - p)*(f(v, 0) if p < 1 else 0)

    def weight(deltas, turn):
        return prob[(score, opponent_score, turn)].get(deltas,
        prob_backup[(score, opponent_score, turn)].get(deltas, 0.25)) 

    def delta_dist(turn):
        prefix = dp_matrix[(score, opponent_score, turn)]
        # track the probability of feral hogs occuring for each dice roll.
        denom = sum(weight(deltas, turn) for deltas in prefix)
        count = {}
        for last_delta, prev_delta in prefix:
            w = weight((last_delta, prev_delta), turn)
            count[last_delta - 2] = count.get(last_delta - 2, 0) + w
            count[last_delta + 2] = count.get(last_delta + 2, 0) + w 
        return count, denom

    count0, denom0, count1, denom1 = *delta_dist(False), *delta_dist(True) 
    # position is impossible, default to standard dp just in case
    if denom0 == 0 and denom1 == 0:
        return final_strategy_dp1(score, opponent_score)
    # one is impossible, another isn't - copy possible into impossible
    if max(denom0, denom1) != 0 and min(denom0, denom1) == 0:
        if denom0 != 0:
            count1, denom1 = count0, denom0
        else:
            count0, denom0 = count1, denom1 

    strategy = [(0, feral_value(recur, 0, free_bacon(opponent_score)))] + \
               [(k, feral_value(expected_value, k, k)) for k in range(1, K + 1)]

    best_roll, best_ev = best_move(strategy)
    # assume opponent randomly makes mistakes
    return (best_roll, (1 - EPSILON)*best_ev + EPSILON*sum(x[1] for x in strategy)/len(strategy)) \
           if turn else (best_roll, best_ev)

def final_strategy(score, opponent_score):
    return final_strategy_dp(score, opponent_score)[0]

def final_strategy2(score, opponent_score, turn, last_delta, prev_delta):
    return final_strategy(score, opponent_score)

if __name__ == "__main__":
    # perfect_strategy(0, 0, False, 0, 0)
    # dump("dp_matrix.pickle", dp_matrix)
    # exit()

    msg = input("Are you sure you want to overwrite the pickle file? ").lower() 
    if len(msg) > 0 and msg[0] == "y":
        mat = [[final_strategy(i, j) for j in range(100)] for i in range(100)]
        dump("mat.pickle", mat)

