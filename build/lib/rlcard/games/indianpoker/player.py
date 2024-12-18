from enum import Enum


class PlayerStatus(Enum):
    ALIVE = 0
    FOLDED = 1
    ALLIN = 2



class IndianPokerPlayer:
    def __init__(self, player_id, init_chips, np_random):
        """
        Initialize a player.

        Args:
            player_id (int): The id of the player
            init_chips (int): The number of chips the player has initially
        """
        self.np_random = np_random
        self.player_id = player_id
        self.hand = []
        self.status = PlayerStatus.ALIVE

        # The chips that this player has put in until now
        self.in_chips = 0
        self.remained_chips = init_chips
    
    def reset(self):
        """
        Reset hands of player
        """
        assert self.in_chips == 0, f"in_chips {self.in_chips}"
        self.hand = []
        self.in_chips = 0
        self.status = PlayerStatus.ALIVE

    
    def update(self, payoff=0):
        """
        If no chips remained, fail to resume the game
        """
        self.remained_chips += self.in_chips
        self.remained_chips += payoff
        self.in_chips = 0
        return self.remained_chips > 0
    
    def get_state(self, rival_cards, all_chips, legal_actions):
        """
        Encode the state for the player

        Args:
            public_cards (list): A list of public cards that seen by all the players
            all_chips (int): The chips that all players have put in

        Returns:
            (dict): The state of the player
        """
        return {
            'hand': [c.get_index() for c in self.hand],
            'rival_cards': rival_cards,
            'all_chips': all_chips,
            'my_chips': self.in_chips,
            'legal_actions': legal_actions
        }

    def get_player_id(self):
        return self.player_id
    
    def bet(self, chips):
        quantity = chips if chips <= self.remained_chips else self.remained_chips
        self.in_chips += quantity
        self.remained_chips -= quantity
