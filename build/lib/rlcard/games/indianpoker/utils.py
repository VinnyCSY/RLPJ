import numpy as np

RANKS = 'A23456789TJQK'

def compare_hands(hands):
    '''
    Compare all palyer's all seven cards
    Args:
        hands(list): cards of those players with same highest hand_catagory.
        e.g. hands = [['CT', 'ST', 'H9', 'B9', 'C2', 'C8', 'C7'], ['CJ', 'SJ', 'H9', 'B9', 'C2', 'C8', 'C7'], ['CT', 'ST', 'H9', 'B9', 'C2', 'C8', 'C7']]
    Returns:
        [0, 1, 0]: player1 wins
        [1, 0, 0]: player0 wins
        [1, 1, 1]: draw
        [1, 1, 0]: player1 and player0 draws

    if hands[0] == None:
        return [0, 1]
    elif hands[1] == None:
        return [1, 0]
    '''
    hand_rank = [evaluate_hand(hand) for hand in hands]

    # all the players in this round, 0 for losing and 1 for winning or draw
    all_players = [1 if i == max(hand_rank) else 0 for i in hand_rank]

    return all_players
def evaluate_hand(hand):
    '''
    Evaluate rank of hand
    Args:
        hands(list): cards of those players with same highest hand_catagory.
        e.g. hands = [['CT', 'ST', 'H9', 'B9', 'C2', 'C8', 'C7'], ['CJ', 'SJ', 'H9', 'B9', 'C2', 'C8', 'C7'], ['CT', 'ST', 'H9', 'B9', 'C2', 'C8', 'C7']]
    Returns:
        rank: scalar of evaluated rank score
    '''
    if hand is None:
        return -1
    else:
        return RANKS.index(hand[0][1])
