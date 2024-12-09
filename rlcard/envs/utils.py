from rlcard.games.indianpoker.utils import evaluate_hand, RANKS
from rlcard.games.indianpoker.round import Action
import numpy as np

def indianpoker_pattern(prev_traj):
    '''
        pattern: analysis of pattern/tendency of opponents
            1st col: number of players
            2st col: number of opponents(players-1)
            3rd col: maximum rank of rival cards(from each opponent)
            4th col: action of each opponent
    '''
    pattern = np.zeros((len(prev_traj), len(prev_traj)-1, len(RANKS), len(Action)))
    
    for player_id in range(len(prev_traj)):
        opp_j = 0
        for j, player_states in enumerate(prev_traj):
            if j==player_id:
                continue
            assert len(player_states)%2==1

            for k in range(0, len(player_states), 2):
                if k==len(player_states)-1: # done
                    pass
                else:
                    state = player_states[k]
                    action = player_states[k+1]
                    rival_cards = state['raw_obs']['rival_cards']
                    
                    rank = max([evaluate_hand(hand) for hand in rival_cards])
                    if isinstance(action, Action): # for human action
                        action = action.value
                    pattern[player_id, opp_j, rank, action] += 1
            
            opp_j += 1

    return np.array(pattern)
