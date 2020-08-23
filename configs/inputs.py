import sys

path = sys.path[0] + "/"

ENV_INPUTS = {
                "init_money": 10000,
                "window_size": 20,
                "transaction_cost_rate": 0.0003,
                "n_actions": 3         # should be odd number
                }

DQN_INPUTS = {
                "n_actions":  ENV_INPUTS["n_actions"],      # output size of NN; should be odd number
                "n_states": 27,  # input size of NN; n_states is window_size + 5
                "learning_rate": 0.001,
                "reward_decay": 0.9,
                "e_greedy": 0.99    ,
                "replace_target_iter": 300,
                "memory_size": 200,
                "batch_size": 64,
                "e_greedy_increment": 0.01,
                "hidden_layer": 3,
                "hidden_units": 25
                }



# ticker of trading stock
DATA_INPUTS = {
                "ticker": "simulation",
                "start": '2018-01-01',
                "end":'2020-12-31',
                "train_test_ratio": 0.7, # in-sample and out-of-sample ratio
                "rolling_increments": 1,
                "intraday": True
                }

# max round
EPISODES = 100
option = 1
