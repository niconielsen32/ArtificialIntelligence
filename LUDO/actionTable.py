import numpy as np


class ActionTableEntry():
    def __init__(self, piece, value):
        super().__init__()
        self.pice = piece
        self.value = value

    def add_entry(self, piece, value):
        self.__piece.append(piece)
        self.__value.append(value)


class ActionTable():
    action_table = None
    state = 0

    def __init__(self, states, actions):
        super().__init__()
        self.states = states
        self.actions = actions
        self.reset()

    def set_state(self, state):
        self.state = state.value

    def get_action_table(self):
        return self.action_table

    def get_piece_to_move(self, state, action):
        if state < 0 or action < 0:
            return -1
        return int(self.piece_to_move[state, action])

    def reset(self):
        self.action_table = np.full((self.states, self.actions), np.nan)
        self.piece_to_move = np.full((self.states, self.actions), np.nan)

    # todo: piece implementation missing...
    def update_action_table(self, action, piece, value):
        if np.isnan(self.action_table[self.state, action.value]):
            self.action_table[self.state, action.value] = 1
            self.piece_to_move[self.state, action.value] = piece