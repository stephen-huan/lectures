"""
    This file contains your final_strategy that will be submitted to the contest.
    It will only be run on your local machine, so you can import whatever you want!
    Remember to supply a unique PLAYER_NAME or your submission will not succeed.
"""

PLAYER_NAME = 'Mikoto Misaka and the Tree Diagram'  # Change this line!


from hoglib import load
mat = load("mat.pickle")

def final_strategy(score, opponent_score):
    return mat[score][opponent_score]

