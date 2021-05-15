import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from typing import List
from params import Params

class WallState(Enum):
    Nada = 0
    NorthEast = 1 #Player 1
    SouthWest = 2 #Player 1
    NorthWest = 3 #Player 2
    SouthEast = 4 #Player 2
    North = 5   #NorthWest and NorthEast
    South = 6   #SouthWest and SouthEast
    West = 7 #NorthWest and SouthWest
    East = 8 #NorthEast and SouthEast

class DisplayBoard:


    def __init__(self, board):
        self.board = board
        self.nodes = board.flat_nodes
        self.g_pos = {}
        self.graph = nx.Graph()
        self.initialize_graph()


    def initialize_graph(self):
        def get_x_y_pos(node) -> (float, float):
            x_pos, y_pos = (0, 0)
            r, c = node.get_pos()

            x_pos = (c - r) / 2
            y_pos = (c + r) * 2
            return x_pos, y_pos

        for node in self.nodes:
            x_position, y_position = get_x_y_pos(node)
            self.g_pos[node] = (x_position, y_position)
            self.graph.add_node(node)
            for neighbour in node.board_neighbours:
                self.graph.add_edge(node, neighbour)

    def show_board(self):
        for node in self.nodes:
            node_color = "blue" if node.player_id_for_piece == 1 else "red" if node.player_id_for_piece == -1 else "grey"
            self.graph.nodes[node]['color'] = node_color
        node_size = 3000
        sizes = np.full(len(self.nodes), fill_value=node_size)
        colors = np.array(self.graph.nodes(data='color', default="pink"))[:, 1]
        plt.figure(figsize=(10,10))

        ax = plt.gca()
        ax.invert_yaxis()
        ax.set_aspect(0.3)
        nx.draw(self.graph, self.g_pos, node_color=colors, node_size=sizes, ax=ax, with_labels=True)
        plt.show()


class Board:

    def __init__(self):

        self.current_player = 1
        self.board_size = Params.board_size
        self.flat_nodes = np.ndarray  # All nodes in a flat list for easy iteration
        self.board = np.full((self.board_size, self.board_size), None, dtype=object)
        self.available_moves_full_vector = np.ones_like(self.board)
        self.board_state_vector = np.zeros_like(self.board)
        self.game_ended = False
        self.winner = None

        for r, _ in enumerate(self.board):
            for c, _ in enumerate(self.board):
                self.board[r, c] = Node(r, c, board=self)
        self.board[0,0].connected_walls_board = WallState.North
        self.board[0, -1].connected_walls_board = WallState.East
        self.board[-1, 0].connected_walls_board = WallState.West
        self.board[-1, -1].connected_walls_board = WallState.South
        for node in self.board[0, 1:-1]:
            node.connected_walls_board = WallState.NorthEast
        for node in self.board[1:-1, 0]:
            node.connected_walls_board = WallState.NorthWest
        for node in self.board[-1, 1:-1]:
            node.connected_walls_board = WallState.SouthWest
        for node in self.board[1:-1, -1]:
            node.connected_walls_board = WallState.SouthEast




        # Helper function
        def get_nodes_flat() -> np.ndarray:
            nodes = []
            for row in self.board:
                for el in row:
                    if type(el) is Node:
                        nodes.append(el)
            return np.array(nodes, dtype=object)
        #TODO maybe remove
        self.flat_nodes = get_nodes_flat()

        for row in self.board:
            for node in row:
                if node is not None: #TODO can likely remove
                    node.find_and_connect_neighbours()
        self.displayBoard = DisplayBoard(self)


    def __str__(self) -> str:
        return self.board.__str__()

    def get_available_moves_full_vector(self):
        return self.available_moves_full_vector.flat
    def set_board_state(self, state_vec):
        """
        Should only be called on freshly created board
        :param state_vec: full state vector, current player in first index
        :return:
        """
        ##TODO test that it works
        current_player_in_state = state_vec[0]
        for index, piece in enumerate(state_vec[1:]):
            assert piece ==1 or piece == -1 or piece == 0
            if piece ==1 or piece == -1:
                self.current_player = piece
                self.do_move(index)
        self.current_player = current_player_in_state




    def get_available_moves(self) -> np.array:
        """Returns list of indices in flat_nodes that are open"""
        #TODO use numpy is probably faster
        available_moves = []
        for indice, val in enumerate(self.available_moves_full_vector.flat):
            if val == 1:
                available_moves.append(indice)
        return np.array(available_moves)

    def do_move(self, move: int):
        node = self.flat_nodes[move]
        self.available_moves_full_vector.flat[move] = 0
        assert node.player_id_for_piece == 0, "Piece{} already put down in spot{}".format(node.player_id_for_piece, move)
        node.put_piece(self.current_player)
        self.current_player *= -1 #To switch betwen 1 and -1
        self.board_state_vector.flat[move] = self.current_player

    def has_moves_left(self):
        """Not implemented"""
        if self.get_available_moves().__len__() >= 1:
            return True
        else:
            return False

    def end_game(self):
        self.game_ended = True
        self.winner = self.current_player
    def is_game_ended(self):
        return self.game_ended

    def count_pieces_left(self):
        return sum([node.has_piece for node in self.flat_nodes])

    def display(self):
        self.displayBoard.show_board()

    def get_state(self):
        #TODO use specical node later
        state = np.vectorize(lambda n: n.player_id_for_piece)(self.board.flat)

        return np.insert(state, 0, self.current_player)


class Node():
    #wall_connected
    #neighpours:list

    def __init__(self, r, c, board):
        self.player_id_for_piece = 0
        """0 means no piece, 1 and -1 means corresponding player where -1 is player 2"""
        self.board =board
        self.piece_neighbours = []
        self.connected_walls_game = WallState.Nada

        #Unchanged after init
        self.board_neighbours: list[Node] =[]
        self.connected_walls_board = WallState.Nada
        self.c = c
        self.r = r
    def get_pos(self):
        """get pos tuple, only used by displayBoard"""
        return (self.r,self.c)
    def find_and_connect_neighbours(self):
        """Called during board init to let each node know who is its neighbours, never called after this"""
        r = self.r
        c = self.c
        directions = [(r-1,c),(r-1,c+1), (r, c-1), (r, c+1), (r+1, c-1),(r+1, c)]
        for coordinate in directions:
            try:
                if -1 in coordinate:
                    raise IndexError
                neigh = self.board.board[coordinate]
                if neigh is None:
                    raise ValueError
                self.board_neighbours.append(neigh)
            except IndexError:
                #If trying to access outside the board
                pass
            except ValueError:
                pass

    def put_piece(self, player):
        """player:int 1 or -1, to identify the player that has been selected"""
        assert player == 1 or player == -1, "Illegal player id"
        self.player_id_for_piece = player
        self.connect_to_neighbour_pieces()
        self.connect_to_board_wall()
        self.update_wall_connections()

    def connect_to_neighbour_pieces(self):
        """Called when piece is placed on this node"""
        assert self.player_id_for_piece != 0, "Should only be called after piece has been placed"
        for node_neigh in self.board_neighbours:
            if node_neigh.player_id_for_piece == self.player_id_for_piece:
                self.piece_neighbours.append(node_neigh)
                node_neigh.connected_by_other_node(self)


    def connect_to_board_wall(self):
        if self.connected_walls_board == WallState.Nada:
            self.connected_walls_game = WallState.Nada
            return
        if self.player_id_for_piece == 1:
            if self.connected_walls_board == WallState.North:
                self.connected_walls_game = WallState.NorthEast
                return
            if self.connected_walls_board == WallState.South:
                self.connected_walls_game = WallState.SouthWest
                return
            if self.connected_walls_board == WallState.East:
                self.connected_walls_game = WallState.NorthEast
                return
            if self.connected_walls_board == WallState.West:
                self.connected_walls_game = WallState.SouthWest
                return
            if self.connected_walls_board == WallState.NorthEast:
                self.connected_walls_game = WallState.NorthEast
                return
            if self.connected_walls_board == WallState.SouthWest:
                self.connected_walls_game = WallState.SouthWest
                return
        if self.player_id_for_piece == -1:
            if self.connected_walls_board == WallState.North:
                self.connected_walls_game = WallState.NorthWest
                return
            if self.connected_walls_board == WallState.South:
                self.connected_walls_game = WallState.SouthEast
                return
            if self.connected_walls_board == WallState.East:
                self.connected_walls_game = WallState.SouthEast
                return
            if self.connected_walls_board == WallState.West:
                self.connected_walls_game = WallState.NorthWest
                return
            if self.connected_walls_board == WallState.NorthWest:
                self.connected_walls_game = WallState.NorthWest
                return
            if self.connected_walls_board == WallState.SouthEast:
                self.connected_walls_game = WallState.SouthEast
                return


    def update_wall_connections(self):
        for neighbour in self.piece_neighbours:
            if neighbour.connected_walls_game != self.connected_walls_game:
                if self.connected_walls_game != WallState.Nada and neighbour.connected_walls_game != WallState.Nada:
                    self.board.end_game()
                    return
                #Here we know that either self or neighbour is state Nada
                if neighbour.connected_walls_game == WallState.Nada:
                    neighbour.update_wall_connections()
                #self must be nada
                self.connected_walls_game = neighbour.connected_walls_game
                self.update_wall_connections()
                return





    def connected_by_other_node(self, neighbour):
        self.piece_neighbours.append(neighbour)


    def __repr__(self):
        return str(self.player_id_for_piece)
        #return "id:{}, game_wall:{}".format(self.player_id_for_piece, self.connected_walls_game)


benis = Board()
benis.set_board_state(np.array([ 1,  0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0, -1,
  0,  0,  0,  0,  0,  0, -1,  0, -1,  0,  1,  0,  0]))