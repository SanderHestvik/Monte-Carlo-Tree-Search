from params import Params
import params
import tensorflow as tf
import math
import time
from MCTreeSearch import MCTS
from hex_game import Board as Game
import copy

#TODO remove imports after testing
from nim import NimGame
import numpy as np
import random as r

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

class RLSystem():
    def __init__(self):
        self.tournament_wins = 0
        self.game = Game
        self.board = self.game()
        self.actor = ActorNN("default")
        self.replay_buffer = self.actor.RBUF
        self.mcts = MCTS(self)
        self.next_starting_player = 1

    def reset_game_and_change_start_player(self):
        self.board = self.game()
        self.board.current_player = self.next_starting_player
        self.next_starting_player *= -1

    def get_battle_move(self, board):
        """Used when in tournament"""
        #Need to have a method with same name in both actor and RL_agent
        return self.get_move_from_mcts(board)

    def get_move_from_mcts(self, board):
        self.board = copy.deepcopy(board)
        self.mcts = MCTS(self, start_board=self.board)
        action_index, move_index = self.mcts.mc_search(timed=True)
        return move_index

    def play_game_with_mcts_and_train(self):
        self.reset_game_and_change_start_player()
        counter = 0
        self.mcts.re_init(self.board) #Sets state to start
        while not self.board.is_game_ended():
            counter += 1
            print("real moves done:{}".format(counter))
            action_index, move_index = self.mcts.mc_search()
            self.add_case_to_RBUF()
            self.board.do_move(move_index)
            self.mcts.update_root_node(action_index)

            if Params.display_actual_game == True:
                self.board.display()
        self.actor.train()

    def default_policy(self, board_state:np.array, legal_moves_full_vec:np.array):
        #predict_timer.start()
        move = self.actor.default_policy(board_state, legal_moves_full_vec)       #TODO more of the logic should happen here
        #predict_timer.stop()
        return move
    def add_case_to_RBUF(self ):
        distribution = self.mcts.get_root_distribution()

        assert math.isclose(distribution.sum(), 1, abs_tol=0.001), "distribution should sum to 1 but was: {}".format(distribution.sum())  # TODO remove later

        self.replay_buffer.add_case(self.mcts.root_node.board_state, distribution)

    def save(self, file_name):
        tf.keras.models.save_model(self.actor.model, "neural_nets/"+file_name)
    def __repr__(self):
        self.actor.tournament_wins = self.tournament_wins
        return self.actor.__repr__()

class StupidActor():
    def get_move_distribution(self, board_state):
        return [1] + [0 for _ in range(Params.nn_output_size-1)]


    def default_policy(self, board_state):
        """return index for move, may be illegal"""
        distribution = self.get_move_distribution(board_state)
        max_val = max(distribution)
        return distribution.index(max_val)

class IndexIterator():
    def __init__(self,  to_i, from_i=0):
        """

        :param to_i: non inclusive, must be higher than from_i
        :param from_i: inclusive, must be less than to_i (default 0)
        """
        self.from_i = from_i
        self.to_i = to_i
        self.current_i = from_i

    def next(self):
        ret = self.current_i
        if self.current_i +1 == self.to_i:
            self.current_i = self.from_i
        else:
            self.current_i += 1

        return ret

class RBUF():

    def __init__(self, size):
        self.targets = []
        self.inputs = []
        self.is_filled = False
        self.size = size
        self.index_iterator = IndexIterator(size)

    def add_case(self, board_state, target_dist):
        if not self.is_filled:
            self.targets.append(target_dist)
            self.inputs.append(board_state)
            if self.targets.__len__() == self.size:
                self.is_filled = True
        else:
            index = self.index_iterator.next()
            self.targets[index] = target_dist
            self.inputs[index] = board_state
    def get_n_random_training_cases(self, size):
        """

        :param size: number of cases, if size is larger than amount of cases all are returned
        :return: tuple of type (inputs:np.array, targets:np.array)
        """
        if size >= self.inputs.__len__():
            return (np.array(self.inputs), np.array(self.targets))

        index_list = np.arange(self.targets.__len__())
        index_list = np.random.choice(index_list, size=size, replace=False)
        return (np.array(self.inputs)[index_list], np.array(self.targets)[index_list])

    def get_n_last_targets(self, n):
        if self.targets.__len__() >=n:
            return np.array(self.targets[-n:])
        else:
            return np.array(self.targets)

    def get_n_last_inputs(self, n):
        if self.inputs.__len__() >=n:
            return np.array(self.inputs[-n:])
        else:
            return np.array(self.inputs)




class ActorNN():
    def __init__(self, name):
        self.tournament_wins = 0
        self.name = name
        self.get_dist_timer = Timer("PREDICT_DISTRIBUTION_TIMER")
        self.RBUF = RBUF(Params.RBUF_size)
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(Params.nn_input_size,), dtype=tf.int64))
        for layer_size in Params.nn_inner_layer_dims:
            self.model.add(tf.keras.layers.Dense(layer_size, activation=Params.nn_inner_activation_function))
        self.model.add(tf.keras.layers.Dense(Params.nn_output_size, activation="softmax"))
        optimizer = Params.nn_optimizer(learning_rate=Params.nn_learning_rate)
        self.model.compile(optimizer=optimizer, loss=Params.nn_loss_function)

    def get_move_distribution(self, board_state):
        #if Params.time_code == True:
        #    self.get_dist_timer.start()
        dist = self.model(board_state )
        #if Params.time_code == True:
        #    self.get_dist_timer.stop()
        return dist
    def get_battle_move(self, board):
        return self.default_policy(board_state=board.get_state(), legal_moves_full_vec=board.get_available_moves_full_vector())
    def default_policy(self, board_state:np.array, legal_moves_full_vec:np.array):
        """returns action index to use for move, same as argument for do_move in game"""

        distribution = self.get_move_distribution(board_state.reshape(1, board_state.shape[0]))
        distribution = distribution*legal_moves_full_vec #removes illegal moves from dist
        assert np.sum(distribution) != 0, "All legal moves have 0 probability" #TODO remove when it works
        return np.argmax(distribution)


    def train(self):
        inputs, targets = self.RBUF.get_n_random_training_cases(Params.RBUF_cases_per_train)
        self.model.fit(inputs, targets, batch_size=5, epochs=Params.nn_training_epochs)
    def __repr__(self):
        return "name: {}, wins: {}".format(self.name, self.tournament_wins)


#Separarate, only in this file to avoid circular import
class TOPP():
    def __init__(self, names_of_nns=[], use_only_nn=True):
        """:use_only_nn, true (default) means only nn selects move, False uses mcts search"""
        self.agents = []
        for nn_name in names_of_nns:
            a = ActorNN(nn_name)
            a.model = tf.keras.models.load_model("neural_nets/"+nn_name, custom_objects={'deepnet_cross_entropy':params.deepnet_cross_entropy})
            if not use_only_nn:
                rl_agent = RLSystem()
                rl_agent.actor = a
                a = rl_agent
            self.agents.append(a)
        self.board = Game()

    def tournament(self):
        for index, actor1 in enumerate(self.agents):
            for actor2 in self.agents[index + 1:]:
                self.battle_actors(actor1, actor2)
    def battle_actors(self, actor_1:ActorNN, actor_2:ActorNN):
        game_count = Params.G
        #TODO might be some asymmetry bug here
        turn = 1 #"actor 1 is 1 and actor 2 is -1
        last_game_start = 1
        #TODO set exploration to 0
        for _ in range(game_count):
            if Params.display_battle_nn:
                self.board.display()
                time.sleep(Params.battle_nn_display_delay)
            while not self.board.is_game_ended():
                current_player = actor_1 if turn == 1 else actor_2
                if Params.display_battle_nn:
                    print(current_player, "will do next move")
                move_index = current_player.get_battle_move(self.board)
                self.board.do_move(move_index)
                if Params.display_battle_nn:
                    self.board.display()
                    time.sleep(Params.battle_nn_display_delay)
                turn *= -1
            if self.board.winner*last_game_start == 1:
                actor_1.tournament_wins += 1
            else:
                actor_2.tournament_wins += 1
            last_game_start *= -1
            turn = last_game_start
            self.board = Game()
        #print("{} won: {} while {} won: {} games".format(actor_1, actor_1.tournament_wins, actor_2, actor_2.tournament_wins))

# actor_1 = ActorNN("ACTOR_10_episodes")
# actor_1.model = tf.keras.models.load_model("neural_nets/hex_first_try_10_episodes")
#
# actor_2 = ActorNN("ACTOR_50_episodes")
# actor_2.model = tf.keras.models.load_model("neural_nets/hex_first_try_50_episodes")
#
# actor_3 = ActorNN("ACTOR_120_episodes")
# actor_3.model = tf.keras.models.load_model("neural_nets/hex_first_try_120_episodes")
#
# actor_4 = ActorNN("ACTOR_350_episodes")
# actor_4.model = tf.keras.models.load_model("neural_nets/hex_first_try_350_episodes")
#
# topp = TOPP(None, None)
# topp.battle_nn(actor_1=actor_1, actor_2=actor_3)

all_timer = Timer("ALL_TIMER")
all_timer.start()

def reload_run(c_val, epsilon_val, list_of_names, last_saved_gen):
    Params.c_factor = c_val
    Params.epsilon = epsilon_val
    for i in range(0, last_saved_gen+1, Params.save_interval):
        list_of_names.append(Params.name_of_run+"_"+str(i))


#Full run
def full_algorithm():
    rl_sys = RLSystem()
    name = Params.name_of_run
    names = ["size_6_big_run_125","size_6_big_run_two_100", "size_6_big_run_three_40"] #Of saved nns
    last_saved_gen = 200 #should be 0 if starting a new run
    if True: #change true to false if starting new run
        reload_run(0.42, 0.015, names, last_saved_gen=last_saved_gen)
    elif Params.save_nns:
        rl_sys.save(name+"_0")
        names.append(name+"_0")
    tournament = TOPP(names_of_nns=names, use_only_nn=Params.use_nn_for_battle)
    tournament.tournament()
    print(*tournament.agents, sep="\n")
    print(names)
    print("epsilon: {}, c_factor: {}\n".format(Params.epsilon, Params.c_factor))
    tournament = None
    for i in range(last_saved_gen+1, Params.episodes+1):
        print("EPISODE_NUMBER: {}".format(i))
        rl_sys.play_game_with_mcts_and_train()
        Params.c_factor = Params.c_factor*Params.c_factor_decay
        Params.epsilon = Params.epsilon*Params.epsilon_decay
        Params.epsilon = max(Params.epsilon, Params.epsilon_min)
        if Params.save_nns:
            if i%Params.save_interval == 0:
                rl_sys.save(name+"_{}".format(i))
                names.append(name+"_{}".format(i))
                tournament = TOPP(names_of_nns=names, use_only_nn=Params.use_nn_for_battle)
                tournament.tournament()
                print(*tournament.agents, sep="\n")
                print(names)
                print("epsilon: {}, c_factor: {}\n".format(Params.epsilon, Params.c_factor))
                tournament = None
    if Params.save_nns:
        tournament = TOPP(names_of_nns=names, use_only_nn=Params.use_nn_for_battle)
        tournament.tournament()
        print(*tournament.agents, sep="\n")


def load_nn_and_tournament():
     name = "size_4_big_nn_episodes"
     names = [name + "_{}".format(runs) for runs in range(0, 101, 20)]
     #names.reverse()
     t = TOPP(names, use_only_nn=Params.use_nn_for_battle)
     t.tournament()
     print(*t.agents, sep="\n")

def battle_0_vs_100():
    names = ["size_4_big_nn_episodes_100", "size_4_big_nn_episodes_0"]
    t = TOPP(names, use_only_nn=Params.use_nn_for_battle)
    #Params.display_battle_nn = True
    t.tournament()
    print(*t.agents, sep="\n")
def load_agent(nn_file_path:str, use_mcts=True):
    actor = ActorNN(nn_file_path)
    actor.model = tf.keras.models.load_model(nn_file_path, custom_objects={'deepnet_cross_entropy':params.deepnet_cross_entropy})
    if use_mcts:
        rl_agent = RLSystem()
        rl_agent.actor = actor
        return rl_agent
    return actor

if __name__ == "__main__":
    #battle_0_vs_100()
    full_algorithm()
    #load_nn_and_tournament()
all_timer.stop()
print(all_timer)
print(MCTS.rollout_timer)
print(MCTS.select_move_timer)
print(MCTS.tree_policy_timer)
