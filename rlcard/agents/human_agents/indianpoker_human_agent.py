''' Limit Hold 'em rule model
'''
import rlcard
from rlcard.models.model import Model
from rlcard.games.indianpoker.utils import evaluate_hand

class IndianPokerRuleAgentV1(object):
    ''' Limit Hold 'em Rule agent version 1
    '''

    def __init__(self):
        self.use_raw = True

    @staticmethod
    def step(state):
        ''' Predict the action when given raw state. A simple rule-based AI.
        Args:
            state (dict): Raw state from the game

        Returns:
            action (str): Predicted action
        '''
        legal_actions = state['raw_legal_actions']
        state = state['raw_obs']
        hand = state['hand']
        public_cards = state['public_cards']
        action = 'fold'
        
        import random
        return random.choice(legal_actions)

    def eval_step(self, state):
        ''' Step for evaluation. The same to step
        '''
        return self.step(state), []

class IndianPokerRuleAgentV1(Model):
    ''' Limitholdem Rule Model version 1
    '''

    def __init__(self):
        ''' Load pretrained model
        '''
        env = rlcard.make('limit-holdem')

        rule_agent = IndianPokerRuleAgentV1()
        self.rule_agents = [rule_agent for _ in range(env.num_players)]

    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        return self.rule_agents

    @property
    def use_raw(self):
        ''' Indicate whether use raw state and action

        Returns:
            use_raw (boolean): True if using raw state and action
        '''
        return True