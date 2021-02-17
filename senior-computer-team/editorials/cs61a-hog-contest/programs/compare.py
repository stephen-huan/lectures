from functools import lru_cache
from hoglib import *

def recur5(f, score, opponent_score, turn, last_delta, prev_delta, delta, roll, give_all):
    return 1 - f(*state(score, opponent_score, delta + 3*is_feral(last_delta, roll)), 
                 turn ^ 1, prev_delta, delta if delta <= 12 else INF, give_all)

def compare(f, s1, s2, both=True):
    global F1, F2
    F1, F2 = s1, s2
    v1 = f(0, 0, False, 0, 0)
    if both:
        f.cache_clear()
        F1, F2 = F2, F1
        v2 = 1 - f(0, 0, False, 0, 0)
    else:
        v2 = v1
    return (v1 + v2)/2 

@lru_cache(maxsize=None)
def server(score, opponent_score, turn, last_delta, prev_delta, give_all=False):
    if score >= GOAL_SCORE or opponent_score >= GOAL_SCORE:
        return score >= GOAL_SCORE

    state = (score, opponent_score, turn, last_delta, prev_delta)
    recur = lambda delta, roll: recur5(server, *state, delta, roll, give_all)

    f = F1 if not turn else F2
    k = f(score, opponent_score) if not give_all else \
        f(score, opponent_score, turn, last_delta, prev_delta)
    return recur(free_bacon(opponent_score), 0) if k == 0 else \
           expected_value(recur, k, k)

all_info = lambda *args: server(*args, True)
all_info.cache_clear = lambda: server.cache_clear()

count = {(i, j, b): {} for i in range(GOAL_SCORE) for j in range(GOAL_SCORE) for b in range(2)}
dp_matrix = {}

def deltas(score, opponent_score, turn, last_delta, prev_delta, give_all=True):
    if score >= GOAL_SCORE or opponent_score >= GOAL_SCORE:
        return score >= GOAL_SCORE

    prefix = count[(score, opponent_score, turn)]
    prefix[(last_delta, prev_delta)] = prefix.get((last_delta, prev_delta), 0) + 1
    key = (score, opponent_score, turn, last_delta, prev_delta)
    if key in dp_matrix:
        return dp_matrix[key]

    recur = lambda delta, roll: recur5(deltas, *key, delta, roll, give_all)

    f = F1 if not turn else F2
    k = f(score, opponent_score) if not give_all else \
        f(score, opponent_score, turn, last_delta, prev_delta)
    dp_matrix[key] = recur(free_bacon(opponent_score), 0) if k == 0 else \
                     expected_value(recur, k, k)
    return dp_matrix[key]

def dict_cache_clear():
    global dp_matrix
    dp_matrix = {}

deltas.cache_clear = dict_cache_clear 

if __name__ == "__main__":
    from final_strategy import final_strategy
    from baseline_strategy import baseline_strategy
    from final import ignore_feral_strategy, perfect_strategy, final_strategy2

    ### final v baseline
    # print(compare(server, ignore_feral_strategy, baseline_strategy))

    ### final v perfect player
    # print(compare(all_info, perfect_strategy, lambda x, y, *args: final_strategy(x, y), False))

    ### attempts to extract a delta distribution
    # print(compare(deltas, perfect_strategy, perfect_strategy))
    # print(compare(deltas, final_strategy2, final_strategy2))

    # dump("count.pickle", count)

