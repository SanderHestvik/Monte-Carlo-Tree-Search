import math
import time
import numpy as np

from nim import NimGame
from params import Params
from hex_game import Board as Game
import copy
import random

class Node():
    def __init__(self, board_state:np.array, legal_actions:np.array):
        self.c_factor = Params.c_factor

        self.board_state = board_state.copy() #TODO might be able to remove copy
        self.N_visits = 0
        self.actions = legal_actions #action is the input for the hex game
        self.N_action_visits = np.zeros(legal_actions.shape)
        self.Q_values = self.N_action_visits.copy()
        self.E_t = self.N_action_visits.copy() #Acummulated reward
        self.Q_u_values = self.N_action_visits.copy() #TODO maybe init to random very small number to have random pick from start

        self.children = self.N_action_visits.copy().tolist() #TODO try remove copy


    def eval_q_u_values(self): #TODO maybe remove
        n_visit_log = math.log2(self.N_visits)
        player = self.board_state[0]
        assert player ==1 or player == -1, "Player needs to be 1 or -1 on for eval to be correct"
        for index in range(len(self.actions)):
            self.Q_u_values[index] =  self.Q_values[index] + self.calculate_u_s_a(index, n_visit_log)*player
        return self.Q_u_values

    def calculate_u_s_a(self, index, n_visit_log):
        return self.c_factor*math.sqrt((n_visit_log) / (1 + self.N_action_visits[index]))

    def update(self, reward, action_index):
        """called when backtracking, reward is from rollout and action_index is the action used"""
        self.N_visits +=1
        self.N_action_visits[action_index] += 1
        self.E_t[action_index] += reward
        self.Q_values[action_index] = self.E_t[action_index]/self.N_action_visits[action_index]
        self.eval_q_u_values() #TODO maybe move to avoid unneccesary updating of this

    def has_child_node(self, action_index):
        return not (self.children[action_index] == 0)

    def get_child_node(self, action_index):
        assert not (self.children[action_index] == 0) #TODO remove when code works
        return self.children[action_index]

    def add_child_node(self, action_index, game_board):
        assert self.children[action_index] ==0, "Should not overwrite created node"
        self.children[action_index] = Node(game_board.get_state(), legal_actions=game_board.get_available_moves())

# timer class template from https://realpython.com/python-timer/#creating-a-python-timer-class
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self, name):
        self.name = name
        self._start_time = None
        self.total_time = 0.0

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self.total_time += elapsed_time
        self._start_time = None
        #print(f"Elapsed time: {elapsed_time:0.4f} seconds")

    def __repr__(self):
        return "name: {}, used {} seconds during run".format(self.name, self.total_time)




class MCTS():
    rollout_timer = Timer("ROLLOUT_TIMER")
    select_move_timer = Timer("SELECT_MOVE_TIMER")
    tree_policy_timer = Timer("TREE POLICY TIMER")
    def __init__(self, RL_system, start_board = Game()):
        self.RL_system = RL_system
        self.game = Game #NimGame or HexGame for example
        self.sim_board = copy.deepcopy(start_board)
        self.root_node:Node = Node(board_state=self.sim_board.get_state(), legal_actions=self.sim_board.get_available_moves())
        self.root_game = copy.deepcopy(self.sim_board)
        self.current_node:Node = self.root_node
        self.backup_trace = [] #contains (Node, action_index)



    def update_root_node(self, node_action_index):
        self.root_node = self.root_node.children[node_action_index]
    def re_init(self, start_board):
        self.sim_board = copy.deepcopy(start_board)
        self.root_node = Node(self.sim_board.get_state(), legal_actions=self.sim_board.get_available_moves())
        self.current_node = self.root_node
        self.root_game = copy.deepcopy(start_board)
    def mc_search(self, timed=False):
        """:returns (action_index, move_index)"""
        if timed:
            counter = 0
            start_time = time.time()
            while counter < Params.min_simulations_when_timed or (time.time() - start_time) <= Params.time_seconds_per_mc_search:
                self.simulate()
                counter += 1
            print("did {} simulations in {}.2f seconds".format(counter, time.time() - start_time))
        else:
            for _ in range(Params.simulations_count):
                self.simulate()

        action_index = self.select_real_action_index()
        move_index = self.root_node.actions[action_index]
        self.root_game.do_move(move_index)
        return (action_index, move_index)



    def get_root_distribution(self):
        distribution = np.zeros(Params.nn_output_size)
        for action_index, action_visits in zip(self.root_node.actions, self.root_node.N_action_visits):
            distribution[action_index] = action_visits
        distribution = distribution/distribution.sum()
        return distribution

    def select_real_action_index(self):
        return np.argmax(self.root_node.N_action_visits)

    def simulate(self):
        self.sim_board = copy.deepcopy(self.root_game)
        self.current_node = self.root_node
        self.tree_policy() #Appends tree actions to backup_trace
        reward_z = self.rollout() #TODO Important, ensure it returns z when tree policy finds end state
        self.backup(reward_z)


    def backup(self, reward_z:int):
        #TODO consider using rollout states for trace aswell
        for(node, action_index) in self.backup_trace:
            node.update(reward_z, action_index)
        self.backup_trace.clear() #TODO move to simulate

    def tree_policy(self):
        if (Params.display_tree_moves == True):
            print("\nTREE MOVES")
            self.sim_board.display()
            time.sleep(Params.sleep_time_after_display_rollout)
        if Params.time_code:
           MCTS.tree_policy_timer.start()
        while not self.sim_board.is_game_ended():
            if(self.current_node.N_visits == 0):
                self.current_node.N_visits +=1
                if Params.time_code:
                    MCTS.tree_policy_timer.stop()
                return #We have found an unvisited node, thus rollout
            action_index = self.select_action_index()
            move_index = self.current_node.actions[action_index]
            self.sim_board.do_move(move_index)

            if(Params.display_tree_moves == True):
                self.sim_board.display()
                time.sleep(Params.sleep_time_after_display_rollout)
            if not self.current_node.has_child_node(action_index):
                self.current_node.add_child_node(action_index=action_index, game_board=self.sim_board)
            self.backup_trace.append((self.current_node, action_index))
            self.current_node = self.current_node.get_child_node(action_index=action_index)
        if Params.time_code:
            MCTS.tree_policy_timer.stop()



    def select_action_index(self):
        """Returns action index, action index is index in mcts node"""
        if Params.time_code:
           MCTS.select_move_timer.start()
        self.current_node.eval_q_u_values()
        if self.current_node.board_state[0] == 1: #TODO I assume I am always player 1 and opponent -1
            best = np.amax(self.current_node.Q_u_values)
            bests = np.argwhere(self.current_node.Q_u_values==best).flat
            a= np.random.choice(bests)
            if Params.time_code:
                MCTS.select_move_timer.stop()
            return a
        if self.current_node.board_state[0] == -1:
            best = np.amin(self.current_node.Q_u_values)
            bests = np.argwhere(self.current_node.Q_u_values == best).flat
            a = np.random.choice(bests)
            if Params.time_code:
                MCTS.select_move_timer.stop()
            return a
        raise Exception("Should not get here") #TODO remove later

    def rollout(self):
        if Params.time_code == True:
            MCTS.rollout_timer.start()
        if (Params.display_rollout == True):
            print("\nROLLOUT")
            self.sim_board.display()
            time.sleep(Params.sleep_time_after_display_rollout)
        while not self.sim_board.is_game_ended():
            if random.random() < Params.epsilon:
                selected_move = np.random.choice(self.sim_board.get_available_moves())
            else:
                selected_move = self.RL_system.default_policy(self.sim_board.get_state(), self.sim_board.get_available_moves_full_vector())

            self.sim_board.do_move(selected_move)
            #if(Params.display_rollout ==True):
            #    self.sim_board.display()
            #    time.sleep(Params.sleep_time_after_display_rollout)
        if (Params.display_rollout == True):
            print(self.sim_board.winner)
        if Params.time_code == True:
            MCTS.rollout_timer.stop()
        return self.sim_board.winner #+1 if player 1, -1 if player two




