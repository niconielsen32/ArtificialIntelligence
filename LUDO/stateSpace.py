from enum import Enum

import numpy as np
from actionTable import ActionTable
from ludopy.player import Player


class State(Enum):
    HOME = 0  # state in which the token is home, which also is the starting position.
    SAFE = 1  # state in which the token has reached goal.
    UNSAFE = 2  # state where the token is safe but not at home or goal.


class Action(Enum):
    # Home
    HOME_MoveOut = 0  # Moving token out of start
    HOME_MoveDice = 1  # Moving eyes of dice
    HOME_Goal = 2  # Move into goal position
    HOME_Star = 3  # Move on star (jump forward)
    HOME_Globe = 4  # Move to a save point
    HOME_Protect = 5  # Move to same token as yourself -> save
    HOME_Kill = 6  # Kill another player
    HOME_Die = 7  # Move to a field where opponent has 2 or more pieces
    HOME_GoalZone = 8  # Move into goal zone

    # Save
    SAFE_MoveOut = 9  # Moving token out of start
    SAFE_MoveDice = 10  # Moving eyes of dice
    SAFE_Goal = 11  # Move into goal position
    SAFE_Star = 12  # Move on star (jump forward)
    SAFE_Globe = 13  # Move to a save point
    SAFE_Protect = 14  # Move to same token as yourself -> save
    SAFE_Kill = 15  # Kill another player
    SAFE_Die = 16  # Move to a field where opponent has 2 or more pieces
    SAFE_GoalZone = 17  # Move into goal zone

    # Unsave
    UNSAFE_MoveOut = 18  # Moving token out of start
    UNSAFE_MoveDice = 19  # Moving eyes of dice
    UNSAFE_Goal = 20  # Move into goal position
    UNSAFE_Star = 21  # Move on star (jump forward)
    UNSAFE_Globe = 22  # Move to a save point
    UNSAFE_Protect = 23  # Move to same token as yourself -> save
    UNSAFE_Kill = 24  # Kill another player
    UNSAFE_Die = 25  # Move to a field where opponent has 2 or more pieces
    UNSAFE_GoalZone = 26  # Move into goal zone




class StateSpace():

    quarter_game_size = 13
    star_positions = [5, 12, 18, 25, 31, 38, 44, 51]
    globe_positions_global = [9, 22, 35, 48]
    globe_positions_local = [1]
    globe_positions_enemy_local = []
    danger_positions_local = [14, 27, 40]
    local_player_position = [Player(), Player(), Player(), Player()]
    global_player_position = [Player(), Player(), Player(), Player()]
    action_table_player = ActionTable(len(State), len(Action))
    q_learning = None

    def __init__(self):
        super().__init__()

    def global_position(self, player, piece):
        return self.global_player_position[player].pieces[piece]

    def local_position(self, player, piece):
        return self.local_player_position[player].pieces[piece]

    def update_action_table(self, player, action, piece, value):
        self.action_table_player.update_action_table(action, piece, value)
   
    def update_player_positions(self, players):
        self.local_player_position = players
        idx_player = 0
        
        for player in players:
            idx_piece = 0
            for piece in player.pieces:
                if piece == 0:
                    self.global_player_position[idx_player].pieces[idx_piece] = 0
                elif piece == 59:
                    self.global_player_position[idx_player].pieces[idx_piece] = 59
                else:
                    self.global_player_position[idx_player].pieces[idx_piece] = (piece + (self.quarter_game_size * idx_player)) % 52
                idx_piece = idx_piece + 1
            idx_player = idx_player + 1

    def get_global_position(self, player_idx, local_position):
        if local_position == 0:
            return 0
        elif local_position == 59:
            return 59
        else:
            return (local_position + (self.quarter_game_size * player_idx)) % 52

    def check_if_piece_safe(self, player, piece):
        is_protected = len(np.where(self.local_player_position[player].pieces == self.local_position(player, piece))[0]) > 1
        return self.check_if_piece_is_safe_at_location(self.local_position(player, piece), self.global_position(player, piece), is_protected)

    def check_if_piece_is_safe_at_location(self, local_postion, global_position, is_protected):
        if global_position in self.globe_positions_global or local_postion in self.globe_positions_local:
            return True
        # check if piece is in goal zone
        if local_postion >= 53:
            return True
        if local_postion != 0 and local_postion != 59 and is_protected:
            return True
        return False

    def check_if_piece_is_in_danger_at_location(self, local_postion, global_postion):
        if local_postion > 53 or local_postion == 1:
            return False

        if global_postion in self.globe_positions_global:
            return False

        if local_postion in self.danger_positions_local:
            return True
        danger_positions = np.empty

        for i in range(1, 6):
            danger_positions = np.append(danger_positions, np.add(self.enemyList, i))
        if global_postion in danger_positions:
            return True
        return False

    def check_if_piece_is_in_danger(self, player, piece):
        return self.check_if_piece_is_in_danger_at_location(self.local_position(player, piece), self.global_position(player, piece))

    def get_enemy_list(self, player):
        kill_list = []
        die_list = []
        self.enemy_list = []
        for enemy_player_idx in range(len(self.global_player_position)):
            for enemy_piece_index in range(len(self.global_player_position[enemy_player_idx].pieces)):
                enemy_position = self.global_position(enemy_player_idx, enemy_piece_index)
                enemy_local_position = self.local_position(enemy_player_idx, enemy_piece_index)
                if enemy_local_position in player.pieces:
                    continue
                if enemy_position > 53 or enemy_position == 0:
                    continue
                if enemy_position in self.globe_positions_global or enemy_local_position in self.globe_positions_local:
                    die_list.append(enemy_position)
                    continue
                if enemy_position in kill_list:
                    die_list.append(enemy_position)
                    kill_list.remove(enemy_position)
                else:
                    kill_list.append(enemy_position)
        enemyList = []
        enemyList.extend(kill_list)
        enemyList.extend(die_list)
        return (kill_list, die_list, enemyList)

    def get_target_player_state(self, player, piece, dice):
        local_pos = self.local_position(player, piece) + dice
        global_pos = self.global_position(player, piece) + dice
        is_protected = len(np.where(self.local_player_position[player].pieces == local_pos)[0]) >= 1
        if self.local_position(player, piece) == 0:
            return State.HOME
        if self.check_if_piece_is_safe_at_location(local_pos, global_pos, is_protected):
            return State.SAFE
        return State.UNSAFE

    def set_player_state(self, player, piece):
        if self.local_position(player, piece) == 0:
            self.action_table_player.set_state(State.HOME)
        elif self.check_if_piece_safe(player, piece):
            self.action_table_player.set_state(State.SAFE)
        else:
            self.action_table_player.set_state(State.UNSAFE)

    def update_move_out_action(self, player, piece, dice):
        if self.local_position(player, piece) == 0 and dice == 6:
            next_state = self.get_target_player_state(player, piece, dice).value
            if self.get_global_position(player, 1) in self.enemyList:
                self.update_action_table(player, Action(Action.HOME_Kill.value + next_state * 9), piece, 1)
            else:
                self.update_action_table(player, Action(Action.HOME_MoveOut.value + next_state * 9), piece, 1)
            return True
        return False

    def update_move_dice_action(self, player, piece, dice):
        if self.local_position(player, piece) == 0:
            return False
        if self.local_position(player, piece) + dice <= 59:
            next_state = self.get_target_player_state(player, piece, dice).value
            self.update_action_table(player, Action(Action.HOME_MoveOut.value + next_state * 9), piece, 1)
            return True

    def update_goal_action(self, player, piece, dice):
        local_target_position = self.local_position(player, piece) + dice
        if local_target_position == 59:
            next_state = self.get_target_player_state(player, piece, dice).value
            self.update_action_table(player, Action(Action.HOME_Goal.value + next_state * 9), piece, 1)
            return True
        return False

    def update_star_action(self, player, piece, dice):
        if self.local_position(player, piece) == 0:
            return False
        if (self.local_position(player, piece) + dice) in self.star_positions:
            next_state = self.get_target_player_state(player, piece, dice).value
            self.update_action_table(player, Action(Action.HOME_Star.value + next_state * 9), piece, 1)
            # self.update_action_table(player, Action.Star,piece,1, dice)
            return True
        return False

    def update_globe_action(self, player, piece, dice):
        if self.local_position(player, piece) == 0:
            return False
        if (self.global_position(player, piece) + dice) in self.globe_positions_global:
            next_state = self.get_target_player_state(player, piece, dice).value
            self.update_action_table(player, Action(Action.HOME_Globe.value + next_state * 9), piece, 1)
            # self.update_action_table(player, Action.Globe,piece,1, dice)
            return True
        return False

    def update_protect_action(self, player, piece, dice):
        if self.local_position(player, piece) == 0:
            return False
        target_position = self.local_position(player, piece) + dice
        if target_position > 53:
            return False
        for i in range(len(self.local_player_position)):
            if i == piece:
                continue
            if target_position == self.local_position(player, i):
                next_state = self.get_target_player_state(player, piece, dice).value
                self.update_action_table(player, Action(Action.HOME_Protect.value + next_state * 9), piece, 1)
                # self.update_action_table(player, Action.Protect,piece,1, dice)
                return True
        return False

    def update_kill_action(self, player, piece, dice, kill_list):
        if self.local_position(player, piece) == 0:
            local_target_position = 1
        else:
            local_target_position = self.local_position(player, piece) + dice
        if local_target_position > 53:
            return False

        target_position = self.global_position(player, piece) + dice
        if (
            target_position in kill_list
            and target_position not in self.globe_positions_global
            and local_target_position not in self.globe_positions_local
            and local_target_position not in self.danger_positions_local
        ):
            next_state = self.get_target_player_state(player, piece, dice).value
            self.update_action_table(player, Action(Action.HOME_Kill.value + next_state * 9), piece, 1)
            # self.update_action_table(player,Action.Kill,piece,1)
            return True
        return False

    def update_die_action(self, player, piece, dice, dieList):
        if self.local_position(player, piece) == 0:
            return False
        local_target_position = self.local_position(player, piece) + dice
        if local_target_position > 53:
            return False

        target_position = self.global_position(player, piece) + dice
        if target_position in dieList:
            next_state = self.get_target_player_state(player, piece, dice).value
            self.update_action_table(player, Action(Action.HOME_Die.value + next_state * 9), piece, 1)
            # self.update_action_table(player, Action.Die,piece,1, dice)
            return True
        return False

    def update_goal_zone(self, player, piece, dice):
        if self.local_position(player, piece) == 0:
            return False
        local_target_position = self.local_position(player, piece) + dice
        if local_target_position > 53 and local_target_position < 59:
            next_state = self.get_target_player_state(player, piece, dice).value
            self.update_action_table(player, Action(Action.HOME_GoalZone.value + next_state * 9), piece, 1)
            # self.update_action_table(player,Action.GoalZone,piece,1, dice)
            return True
        return False
    
    
    def get_possible_actions(self, players, current_player, pieces_to_move):
        self.update_player_positions(players)
        self.action_table_player.reset()
        player = players[current_player]
        (killList, dieList, enemyList) = self.get_enemy_list(player)
        self.enemyList = enemyList
        for piece in pieces_to_move:
            for dice in range(1, 6):
                self.set_player_state(current_player, piece)
                self.update_move_out_action(current_player, piece, dice)
                self.update_goal_action(current_player, piece, dice)
                self.update_star_action(current_player, piece, dice)
                self.update_globe_action(current_player, piece, dice)
                self.update_protect_action(current_player, piece, dice)
                self.update_kill_action(current_player, piece, dice, killList)
                self.update_die_action(current_player, piece, dice, dieList)
                self.update_goal_zone(current_player, piece, dice)
                self.update_move_out_action(current_player, piece, dice)

    def check_goal_zone(self, player, piece, dice):
        local_position = self.local_position(player, piece)
        local_target_position = local_position + dice
        if local_target_position < 53:
            return False
        if local_position >= 53:
            self.action_table_player.set_state(State.SAFE)
        if local_target_position == 59:
            self.update_action_table(player, Action(Action.SAFE_Goal), piece, 1)
            return True
        self.update_action_table(player, Action(Action.SAFE_GoalZone), piece, 1)
        return True

    def update(self, players, current_player, pieces_to_move, dice):
        self.update_player_positions(players)
        self.action_table_player.reset()
        player = players[current_player]
        (killList, dieList, enemyList) = self.get_enemy_list(player)
        self.enemyList = enemyList
        for piece in pieces_to_move:
            self.set_player_state(current_player, piece)
            if self.update_move_out_action(current_player, piece, dice):
                continue
            if self.check_goal_zone(current_player, piece, dice):
                continue
            if self.update_die_action(current_player, piece, dice, dieList):
                continue
            if self.update_star_action(current_player, piece, dice):
                continue
            if self.update_globe_action(current_player, piece, dice):
                continue
            if self.update_protect_action(current_player, piece, dice):
                continue
            if self.update_kill_action(current_player, piece, dice, killList):
                continue
            if self.update_move_dice_action(current_player, piece, dice):
                continue