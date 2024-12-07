import numpy as np


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
    RANKS = 'A23456789TJQK'
    hand_rank = []
    all_players = [0]*len(hands) #all the players in this round, 0 for losing and 1 for winning or draw

    for hand in hands:
        if hand is None:
            hand_rank.append(-1)
        else:
            rank = RANKS.index(hand[1])
            hand_rank.append(rank)

    all_players = [1 if i == max(hand_rank) else 0 for i in hand_rank]# potential winner are those with same max card_catagory

    return all_players