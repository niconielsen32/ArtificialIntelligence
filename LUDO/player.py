import numpy as np
from qTable import Rewards
from stateSpace import Action, State, StateSpace


class QLearningAgent(StateSpace):
    ai_player_idx = -1
    debug = False
    q_learning = None
    state = None
    action = None

    def __init__(self, ai_player_idx, gamma=0.3, learning_rate=0.2):
        super().__init__()
        self.q_learning = Rewards(len(State), len(Action), gamma=gamma, lr=learning_rate)
        self.ai_player_idx = ai_player_idx

    def update(self, players, pieces_to_move, dice):
        super().update(players, self.ai_player_idx, pieces_to_move, dice)
        action_table = self.action_table_player.get_action_table()
        state, action = self.q_learning.choose_next_action(self.ai_player_idx, action_table)
        pieces_to_move = self.action_table_player.get_piece_to_move(state, action)
        self.state = state
        self.action = action

        return pieces_to_move

    def reward(self, players, pieces_to_move):
        super().get_possible_actions(players, self.ai_player_idx, pieces_to_move)
        new_action_table = np.nan_to_num(self.action_table_player.get_action_table(), nan=0.0)
        self.q_learning.reward(self.state, new_action_table, self.action)
