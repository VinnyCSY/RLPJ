import json
import os
import numpy as np
from collections import OrderedDict

import rlcard
from rlcard.envs import Env
from rlcard.envs.utils import indianpoker_pattern as pattern
from rlcard.games.indianpoker import Game
from rlcard.games.indianpoker.round import Action
from rlcard.games.indianpoker.utils import RANKS
from rlcard.utils import print_card

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
        self.pattern = np.zeros((self.num_players, self.num_players-1, len(RANKS), len(Action)))
        self.game_set = True
        self.save_setting = True
        self.print_setting = False
        # for raise_amount in range(1, self.game.init_chips+1):
        #     self.actions.append(raise_amount)

        with open(os.path.join(rlcard.__path__[0], 'games/limitholdem/card2index.json'), 'r') as file:
            self.card2index = json.load(file)

    def init_setting(self, save_setting, print_setting):
        # save setting: True if you want to continue the game until ALL_IN / False if you want to reset the game when it's end
        self.save_setting = save_setting
        # print_setting: True if you want to print the game result(your card, chips)
        self.print_setting = print_setting

    def reset(self):
        ''' Start a new game

        Returns:
            (tuple): Tuple containing:

                (numpy.array): The begining state of the game
                (int): The begining player
        '''
        if self.save_setting and not self.game_set:
            state, game_pointer = self.game.continue_game()
        else:
            state, game_pointer = self.game.init_game()
        self.game_set = False
        self.action_recorder = []
        return self._extract_state(state), game_pointer

    def run(self, is_training=False):
        '''
        Run a complete game, either for evaluation or training RL agent.

        Args:
            is_training (boolean): True if for training purpose.

        Returns:
            (tuple) Tuple containing:

                (list): A list of trajectories generated from the environment.
                (list): A list payoffs. Each entry corresponds to one player.

        Note: The trajectories are 3-dimension list. The first dimension is for different players.
              The second dimension is for different transitions. The third dimension is for the contents of each transiton
        '''
        if is_training:
            self.print_setting = False

        trajectories = [[] for _ in range(self.num_players)]
        state, player_id = self.reset()

        # Loop to play the game
        trajectories[player_id].append(state)
        while not self.is_over():
            # start new game if done
            if self.game.is_over():
                # analyze pattern each
                trajectories = [[] for _ in range(self.num_players)]
                state, player_id = self.reset()

                # Loop to play the game
                trajectories[player_id].append(state)
                

            # Agent plays
            if not is_training:
                action, _ = self.agents[player_id].eval_step(state)
            else:
                action = self.agents[player_id].step(state)

            # Environment steps
            next_state, next_player_id = self.step(action, self.agents[player_id].use_raw)
            # Save action
            trajectories[player_id].append(action)

            # Set the state and player
            state = next_state
            player_id = next_player_id

            # game is done
            if self.game.is_over():
                # update payoffs
                payoffs = self.get_payoffs()
                
                all_players, self.game_set = self.game.update(payoffs)
                
                # Add a final state to all the players
                for player_id in range(self.num_players):
                    state = self.get_state(player_id)
                    trajectories[player_id].append(state)
                
                # update trajectories, patterns
                self.prev_trajectories = trajectories
                
                new_pattern = pattern(trajectories)
                self.pattern += new_pattern

                # print result if it's over
                if self.print_setting:
                    self.print_result(payoffs)

            else:
                trajectories[player_id].append(state)

        
        return trajectories, payoffs
    
    def is_over(self):
        ''' Check whether the curent game is over

        Returns:
            (boolean): True if current game is over
        '''
        if self.save_setting:
            return self.game_set
        return self.game.is_over() 

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

        hand = state['rival_cards']
        my_chips = state['my_chips']
        all_chips = state['all_chips']
        cards = [x for x in hand if x is not None][0]
        idx = [self.card2index[card] for card in cards]
        obs = np.zeros(54)
        obs[idx] = 1
        obs[52] = float(my_chips)
        obs[53] = float(max(all_chips))
        extracted_state['obs'] = obs

        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        extracted_state['action_record'] = self.action_recorder

        player_id = state['rival_cards'].index(None)
        extracted_state['pattern'] = self.pattern[player_id]

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


    def print_result(self, payoffs):
        '''
        Print the result of the game if it's over
        '''
        state_info = self.get_perfect_information()
        print('===============     Cards all Players    ===============')
        for i, hands in enumerate(state_info['hand_cards']):
            print('=============  Player',i,'- Hand   =============')
            print_card(hands)
            
        print('===============     Result     ===============')
        if payoffs[0] > 0:
            print('You win {} chips!'.format(payoffs[0]))
        elif payoffs[0] == 0:
            print('It is a tie.')
        else:
            print('You lose {} chips!'.format(-payoffs[0]))
        print('')
        for i, chips in enumerate(state_info['chips']):
            print('Agent {}: {}'.format(i, chips))
        # print(self.pattern)
        input("Press any key to continue...")
        

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