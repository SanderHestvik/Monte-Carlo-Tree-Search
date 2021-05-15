from enum import Enum
from params import Params
import numpy as np

class P(Enum):
    One = 1
    Two = -1


class NimGame():

    def __init__(self, board_state=(1, Params.board_size)):
        self.pile_size = board_state[1]
        self.max_move = Params.max_possible_moves
        self.player: P = P(board_state[0])  # 1 or 2
        self.winner: P = None

    def get_available_moves(self):
        """Returns a vector like 0,1,2,3,4 where 0 corresponds to 1 piece because 0 indexing"""
        return np.array([move for move in range(0, min(self.pile_size, self.max_move))])

    def get_available_moves_full_vector(self):
        vector = np.ones(Params.max_possible_moves)
        vector[min(self.pile_size, self.max_move):] = 0
        return vector

    def do_move(self, move_index: int):
        self.pile_size -= move_index + 1
        self.swich_player()
        if (self.is_game_ended()):
            self.winner = self.player.value #TODO untested

    def swich_player(self):
        if (self.player == P.One):
            self.player = P.Two
            return
        self.player = P.One

    def get_state(self):
        return np.array([self.player.value, self.pile_size])

    def set_state(self, board_state):
        self.player = P(int(board_state[0]))
        self.pile_size = int(board_state[1])

    def display(self):
        print("(current player:{:3}, pieces left:{:3d}, is_ended: {} ".format(self.player, self.pile_size,
                                                                              self.is_game_ended()))

    def is_game_ended(self):
        assert self.pile_size >= 0

        if (self.pile_size == 0):
            return True
        return False

    def get_optimal_move_dist(self, pieces_left):
        self.pile_size = pieces_left
        """for training a test nn"""
        mod = self.pile_size%(self.max_move+1)
        optimal_pieces = 0
        if(mod==0):
            optimal_pieces= self.max_move
        elif(mod==1):
            optimal_pieces= 1
        else:
            optimal_pieces = mod-1
        vector = np.zeros(Params.nn_output_size)
        vector[optimal_pieces-1]=1
        return vector
