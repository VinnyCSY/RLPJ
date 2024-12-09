from rlcard.games.indianpoker.utils import evaluate_hand, RANKS
import numpy as np

def pattern(state, prev_traj, player_id):
    obs = state['obs']

    import pdb; pdb.set_trace()
    prev_cards = prev_traj[0][-1]['raw_obs']['rival_cards']
    idx = prev_cards.index(None)
    prev_cards[idx] = prev_traj[0][-1]['raw_obs']['hand']
    
    assert None not in prev_cards
    assert prev_traj[0][0]['legal_actions'] is not None
    
    # patterns of opponents
    '''
        patterns: analysis of pattern/tendency of opponents
            1st col: number of opponents
            2st col: maximum rank of rival cards(from each opponent)
            3rd col: action of each opponent
    '''
    patterns_of_opponents = []
    
    for i, player_states in enumerate(prev_traj):
        if i==player_id:
            continue
        patterns = np.zeros((len(RANKS), len(prev_traj[0][0]['legal_actions'])))
        
        assert len(player_states)%2==1
        for j in range(0, len(player_states), 2):
            if j==len(player_states)-1: # done
                pass
            else:
                state = player_states[j]
                action = player_states[j+1]
                rival_cards = state['raw_obs']['rival_cards']
                assert rival_cards[j] is None
                
                rank = max([evaluate_hand(hand) for hand in rival_cards])
                patterns[rank, action] += 1
        patterns_of_opponents.append(patterns)
    import pdb; pdb.set_trace()
    return np.array(patterns_of_opponents)
