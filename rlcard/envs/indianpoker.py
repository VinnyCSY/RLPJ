import json
import os
import numpy as np
from collections import OrderedDict

import rlcard
from rlcard.envs import Env
from rlcard.games.indianpoker import Game
from rlcard.games.indianpoker.round import Action

DEFAULT_GAME_CONFIG = {
        'game_num_players': 2,
        'chips_for_each': 100,
        'dealer_id': None,
        }

class IndianPokerEnv(Env):
    ''' IndianPoker Environment
    '''

    def __init__(self, config):
        ''' Initialize the IndianPoker environment
        '''
        self.name = 'indian-poker'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.actions = Action
        self.state_shape = [[54] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]
        
        self.prev_trajectories = None
        self.game_set = True
        # for raise_amount in range(1, self.game.init_chips+1):
        #     self.actions.append(raise_amount)

        with open(os.path.join(rlcard.__path__[0], 'games/limitholdem/card2index.json'), 'r') as file:
            self.card2index = json.load(file)

    def reset(self, save_setting=False):
        ''' Start a new game

        Returns:
            (tuple): Tuple containing:

                (numpy.array): The begining state of the game
                (int): The begining player
        '''
        if save_setting and not self.game_set:
            state, game_pointer = self.game.continue_game()
        else:
            state, game_pointer = self.game.init_game()
        self.game_set = False
        self.action_recorder = []
        return self._extract_state(state), game_pointer

    def run(self, is_training=False, save_setting=False):
        trajectories, payoffs = super().run(is_training, save_setting)
        self.update(trajectories, payoffs)
        return trajectories, payoffs
    
    def _get_legal_actions(self):
        ''' Get all leagal actions

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        '''
        return self.game.get_legal_actions()

    def _extract_state(self, state):
        ''' Extract the state representation from state dictionary for agent

        Note: Currently the use the hand cards and the public cards. TODO: encode the states

        Args:
            state (dict): Original state from the game

        Returns:
            observation (list): combine the player's score and dealer's observable score for observation
        '''
        extracted_state = {}

        legal_actions = OrderedDict({action.value: None for action in state['legal_actions']})
        extracted_state['legal_actions'] = legal_actions

        hand = state['hand']
        my_chips = state['my_chips']
        all_chips = state['all_chips']
        cards = hand
        idx = [self.card2index[card] for card in cards]
        obs = np.zeros(54)
        obs[idx] = 1
        obs[52] = float(my_chips)
        obs[53] = float(max(all_chips))
        extracted_state['obs'] = obs

        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        extracted_state['action_record'] = self.action_recorder
        # TODO: extracted_state['pattern']

        return extracted_state

    def get_payoffs(self):
        ''' Get the payoff of a game

        Returns:
           payoffs (list): list of payoffs
        '''
        return np.array(self.game.get_payoffs())

    def _decode_action(self, action_id):
        ''' Decode the action for applying to the game

        Args:
            action id (int): action id

        Returns:
            action (str): action for the game
        '''
        legal_actions = self.game.get_legal_actions()
        if self.actions(action_id) not in legal_actions:
            if Action.CHECK in legal_actions:
                return Action.CHECK
            else:
                print("Tried non legal action", action_id, self.actions(action_id), legal_actions)
                return Action.FOLD
        return self.actions(action_id)

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['chips'] = [self.game.players[i].remained_chips for i in range(self.num_players)]
        state['rival_cards'] = self.game.rival_cards
        state['hand_cards'] = [[c.get_index() for c in self.game.players[i].hand] for i in range(self.num_players)]
        state['current_player'] = self.game.game_pointer
        state['legal_actions'] = self.game.get_legal_actions()
        return state


    def update(self, trajectories, payoffs):
        '''
        update env after the game finishes
        '''
        self.prev_trajectories = trajectories
        all_players, self.game_set = self.game.update(trajectories, payoffs)