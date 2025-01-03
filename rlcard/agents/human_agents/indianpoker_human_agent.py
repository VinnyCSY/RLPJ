from rlcard.utils.utils import print_card


class HumanAgent(object):
    ''' A human agent for No Limit Holdem. It can be used to play against trained models
    '''

    def __init__(self, num_actions, chips=0):
        ''' Initilize the human agent

        Args:
            num_actions (int): the size of the ouput action space
        '''
        self.use_raw = True
        self.num_actions = num_actions
        # self.chips = chips

    @staticmethod
    def step(state):
        ''' Human agent will display the state and make decisions through interfaces

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action decided by human
        '''
        _print_state(state['raw_obs'], state['action_record'])
        action = int(input('>> You choose action (integer) : '))
        while action < 0 or action >= len(state['legal_actions']):
            print('Action illegal...')
            action = int(input('>> Re-choose action (integer): '))
        return state['raw_legal_actions'][action]

    def eval_step(self, state):
        ''' Predict the action given the curent state for evaluation. The same to step here.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state), {}

def _print_state(state, action_record):
    ''' Print out the state

    Args:
        state (dict): A dictionary of the raw state
        action_record (list): A list of the historical actions
    '''
    _action_list = []
    for i in range(1, len(action_record)+1):
        if action_record[-i][0] == state['current_player']:
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses', pair[1])

    for i, rival_hand in enumerate(state['rival_cards']):
        if rival_hand is None:
            continue
        print('==========  Rival Player',i,'- Hand   ==========')
        print_card(rival_hand)

    print('===============     Chips      ===============')
    print('In Pot:',state["pot"])
    for i in range(len(state["stakes"])):
        print('Agent {}: {}'.format(i, state["stakes"][i]))

    print('\n=========== Actions You Can Choose ===========')
    print(', '.join([str(index) + ': ' + str(action) for index, action in enumerate(state['legal_actions'])]))
    print('')
    print(state)
