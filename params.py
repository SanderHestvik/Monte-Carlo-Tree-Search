
import tensorflow as tf
#from hex_game import Board as HexGame
#from nim import NimGame
class NimParams():
    board_size = 3
    max_possible_moves = 7
    #game = NimGame

    c_factor = 1.0

    nn_input_size = 2
    nn_inner_layer_dims = [25,25,25]
    nn_inner_activation_function = "relu"
    nn_output_size = max_possible_moves
    nn_optimizer = tf.keras.optimizers.SGD
    nn_learning_rate = 0.01 #TODO research
    nn_loss_function = "categorical_crossentropy"


def safelog(tensor, base=0.0001):
    return tf.math.log(tf.math.maximum(tensor, base))


def deepnet_cross_entropy(targets, outs):
    return tf.reduce_mean(tf.reduce_sum(-1 * targets * safelog(outs), axis=[1]))


class HexParams():
    name_of_run = "size_6_big_run_four"

    board_size = 6
    max_possible_moves = -1 #Unused in Hex

    c_factor = 1.0
    c_factor_decay = 0.995
    epsilon = 0.90
    #game = HexGame
    epsilon_decay = 0.9669
    epsilon_min = 0.015
    save_nns = True
    time_seconds_per_mc_search = 2 #Only for battle
    use_nn_for_battle = True
    min_simulations_when_timed = 700 #Only for battle

    G = 10 #Should be divisible by 2 for fair match
    M = 35
    episodes = 350
    save_interval = int(episodes/M)
    nn_input_size = board_size**2+1
    nn_inner_layer_dims =  [108, 108]
    simulations_count = 900
    nn_inner_activation_function = "relu"
    nn_output_size = board_size**2
    nn_optimizer = tf.keras.optimizers.Adam
    nn_learning_rate = 0.0015

    nn_loss_function = deepnet_cross_entropy
    nn_training_epochs = 40
    mini_batch_size = 10
    RBUF_size = 60
    RBUF_cases_per_train = 20



class Params():
    _params = HexParams
    #GLOBAL Params
    name_of_run = _params.name_of_run
    display_rollout = False
    display_actual_game = False
    display_tree_moves = False
    display_battle_nn = False
    battle_nn_display_delay = 0.0
    time_code = False
    sleep_time_after_display_rollout = 0.5
    sleep_time_after_display_real_move = 0.2
    save_nns = _params.save_nns
    use_nn_for_battle = _params.use_nn_for_battle

    time_seconds_per_mc_search = _params.time_seconds_per_mc_search

    simulations_count = _params.simulations_count
    min_simulations_when_timed = _params.min_simulations_when_timed
    board_size = _params.board_size
    #Only used in nim
    max_possible_moves = _params.max_possible_moves

    epsilon = _params.epsilon
    epsilon_decay = _params.epsilon_decay
    epsilon_min = _params.epsilon_min

    c_factor = _params.c_factor
    c_factor_decay = _params.c_factor_decay
    G = _params.G
    M = _params.M
    episodes = _params.episodes
    save_interval = _params.save_interval
    nn_input_size = _params.nn_input_size
    nn_inner_layer_dims = _params.nn_inner_layer_dims
    nn_inner_activation_function = _params.nn_inner_activation_function
    nn_output_size = _params.nn_output_size
    nn_optimizer = _params.nn_optimizer
    nn_loss_function = _params.nn_loss_function
    nn_learning_rate = _params.nn_learning_rate
    nn_training_epochs = _params.nn_training_epochs
    mini_batch_size = _params.mini_batch_size
    RBUF_size = _params.RBUF_size
    RBUF_cases_per_train = _params.RBUF_cases_per_train
    def __init__(self):
        assert False, "Should not be initialized"

