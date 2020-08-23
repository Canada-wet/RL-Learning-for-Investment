from models.DQN import DeepQNetwork
from models.env import Environment
from analyticTools.performance import get_performance
from preprocessing.data_wrangling import *
from configs.inputs import *
import matplotlib.pyplot as plt
import pprint
import datetime


def plot_profit_list(max_round, total_profit_list):
    if max_round > 5:
        fig = plt.figure(figsize = (15,5))
        plt.plot([i+1 for i in range(max_round)], total_profit_list)
        plt.ylabel('total profit')
        plt.xlabel('training episode')
        plt.savefig(path + 'results/profit_per_episode.png')
        plt.close()


def main(max_round, OHLCV_df, risk_free, sp500, total_OHLCV_df, environment = None, train=True, verbose=True, agent=None, initial = True):
    # load OHLCV_df, risk free rate and benchmark
    #OHLCV_df, risk_free, sp500 = load_series(train, **DATA_INPUTS)
    # env = Environment(OHLCV_df, risk_free, **ENV_INPUTS)
    if train and initial:
        env = Environment(OHLCV_df, risk_free, total_OHLCV_df, **ENV_INPUTS)
        agent = DeepQNetwork(**DQN_INPUTS)
    elif train and not initial:
        env = Environment(OHLCV_df, risk_free, total_OHLCV_df, **ENV_INPUTS)
    else:
        env = Environment(OHLCV_df, risk_free, total_OHLCV_df, train, **ENV_INPUTS) if initial else \
        Environment(OHLCV_df, risk_free, total_OHLCV_df, train, environment.cash, environment.inventory, \
        environment.market_value, environment.total_value, environment.pre_value, environment.total_profit, \
        environment.t, environment.total_t, environment.states_sell, environment.states_buy, environment.portfolio_cumulative_rtns, \
        environment.stock_cumulative_rtns, environment.portfolio_daily_rtns, environment.transaction_cost_history,\
        environment.daily_values, environment.logs, **ENV_INPUTS)

    step = 0
    total_profit_list = []
    for episode in range(max_round):
        # initial observation
        s = env.reset()

        while True:

            # RL agent choose action based on observation
            a = agent.choose_action(s, train)
            # print (a)
            # RL agent take action and get next observation and reward
            s_, r, done = env.step(a, verbose = not train)

            agent.store_transition(s, a, r, s_)

            if (step > 200) and (step % 5 == 0) and train:
                agent.learn()

            # swap observation
            s = s_

            # break while loop when end of this episode
            if done:
                break
            step += 1
            # agent.plot_loss()

        if verbose and option == 0:
            print("episode:%d, step:%d, total profit:%f"%(episode+1,step,env.total_profit))

        total_profit_list.append(env.total_profit)

    # stats = get_performance(env.get_daily_net_values().pct_change(), risk_free)
    # pprint.pprint(stats)
    if option == 0:
        # agent.plot_loss()
        env.draw(sp500)
        env.save_logger()
        plot_profit_list(max_round, total_profit_list)

    # return agent if train else env
    if train:
        return agent,total_profit_list
    else:
        return env

def forward_main(option, cont = True, initial = True):
    OHLCV_df, risk_free, sp500 = load_series(**DATA_INPUTS)
    intraday = DATA_INPUTS["intraday"]

    if option == 0:
        main(EPISODES, OHLCV_df, risk_free, sp500,\
        OHLCV_df, train = True, verbose = True)
    else:
        train_test_ratio = DATA_INPUTS["train_test_ratio"]
        rolling_increments = DATA_INPUTS["rolling_increments"]
        size = len(OHLCV_df)

        train_increment = train_test_ratio * (1/rolling_increments)
        test_increment = (1-train_test_ratio) * (1/rolling_increments)

        start = 0
        middle = int(start + size * train_increment)
        test_start = int(start + size * train_increment)
        end = int(middle + size * test_increment)

        env = None
        agent = None

        if (rolling_increments == 1):
            print_period(intraday, start, middle, end, OHLCV_df)

            agent,total_profit_list = main(EPISODES, OHLCV_df[start:middle], risk_free[start:middle], sp500[start:middle],\
            OHLCV_df[test_start:],env, True, True, agent, initial)
            env = main(1, OHLCV_df[middle:end], risk_free[middle:end], sp500[middle:end],\
            OHLCV_df[test_start:], env, False, False, agent, initial)

        else:
            while cont:
                print_period(intraday, start, middle, end, OHLCV_df)
                agent,total_profit_list = main(EPISODES, OHLCV_df[start:middle], risk_free[start:middle], sp500[start:middle],\
                OHLCV_df[test_start:], train = True, verbose = True)
                # print(total_profit_list)

                env = main(1, OHLCV_df[middle:end], risk_free[middle:end], sp500[middle:end], OHLCV_df[test_start:], env, \
                False, False, agent, initial)
                # print(env.cash),

                start = int(start + size * test_increment)
                middle = int(start + size * train_increment)
                end = int(middle + size * test_increment)

                initial = False
                if (end >= size) and middle < size:

                    start = start
                    middle = middle
                    end = size

                    print_period(intraday, start, middle, end, OHLCV_df)

                    agent,total_profit_list = main(EPISODES, OHLCV_df[start:middle], risk_free[start:middle], sp500[start:middle],\
                    OHLCV_df[test_start:], train = True, verbose = True)

                    env = main(1, OHLCV_df[middle:end], risk_free[middle:end], sp500[middle:end], OHLCV_df[test_start:], env, \
                    False, False, agent, initial)
                    cont = False
        # agent.plot_loss()
        env.draw(sp500)
        env.save_logger()
        # print(total_profit_list)
        plot_profit_list(EPISODES, total_profit_list)

def print_period(intraday,start,middle,end,OHLCV_df):

    print ("Trainning start: \t %s, Training end: \t %s \n\
    Testing start: \t %s, Testing end: \t %s"%(datetime.datetime.date(OHLCV_df.index[start]),datetime.datetime.date(OHLCV_df.index[middle-1]),\
    datetime.datetime.date(OHLCV_df.index[middle]),datetime.datetime.date(OHLCV_df.index[end-1])))\
    if intraday == False else \
    print ("Trainning start: \t %s, Training end: \t %s \n\
    Testing start: \t %s, Testing end: \t %s"%(OHLCV_df.index[start],OHLCV_df.index[middle-1],\
    OHLCV_df.index[middle],OHLCV_df.index[end-1]))


if __name__ == "__main__":
    # in sample training
    #main(EPISODES, train=True, verbose=True)
    forward_main(option)
    # out of sample test
    # main(1, train=False, verbose=False, agent=agent)

    # import os
    # import subprocess
    # subprocess.call(['python', os.path.join("/Users/shin/desktop/RL-Trading", 'visualization/app.py')])
