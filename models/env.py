import numpy as np
import pandas as pd
from ta import volume,volatility,trend,momentum,others
from empyrical import sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio
import warnings
from configs.inputs import path

warnings.filterwarnings('ignore')

class Environment:
    def __init__(self, OHLCV_df, risk_free, total_OHLCV_df,\
    train = True, cash = 0, inventory = 0, market_value = 0, total_value = 0, pre_value = 0, total_profit = 0, t = 0, \
    total_t = 0, states_sell = [], states_buy = [], portfolio_cumulative_rtns = [0], stock_cumulative_rtns = [0],\
    portfolio_daily_rtns = [], transaction_cost_history = [], daily_values = [0], logs = [], \
    init_money = 0, window_size = 0, transaction_cost_rate = 0, n_actions = 0):

        self.OHLCV_df = OHLCV_df
        self.timeseries = OHLCV_df['Adj Close'].values
        self.total_timeseries = total_OHLCV_df['Adj Close'].values
        self.risk_free = risk_free.values
        self.date = OHLCV_df.index
        self.total_date = total_OHLCV_df.index
        self.init_money = init_money
        self.window_size = window_size
        self.transaction_cost_rate = transaction_cost_rate

        temp = np.linspace(0, 1, n_actions // 2 + 1)
        self.action_amount = np.append(temp, -temp[1:])
        self.train = train
        self.initial = True if cash == 0 else False

        self.cash = cash
        self.inventory = inventory
        self.market_value = market_value
        self.total_value = total_value
        self.pre_value = pre_value
        self.total_profit = total_profit
        self.t = 0
        self.total_t = total_t

        self.states_sell = states_sell
        self.states_buy = states_buy
        self.portfolio_cumulative_rtns = portfolio_cumulative_rtns
        self.stock_cumulative_rtns = stock_cumulative_rtns
        self.portfolio_daily_rtns = portfolio_daily_rtns
        self.transaction_cost_history = transaction_cost_history
        self.daily_values = daily_values
        self.logs = logs
    
    def reset(self):
        """
        reset some parameters back to time 0
        """
        if self.train or self.initial:
            self.cash = self.init_money
            self.inventory = 0
            self.market_value = 0
            self.total_value = 0
            self.pre_value = self.init_money
            self.total_profit = 0
            self.t = 0
            self.total_t = 0

            self.states_sell = []
            self.states_buy = []
            self.portfolio_cumulative_rtns = [0]
            self.stock_cumulative_rtns = [0]
            self.portfolio_daily_rtns = []
            self.transaction_cost_history = []
            self.daily_values = [self.init_money]
            self.logs = []
            return self.get_state(0)
        else:
            return self.get_state(0)

    def get_state(self, t):
        """
        return the state (define state here) given time t
        """
        # time feature
        d = t - self.window_size
        if d <= 0:
            state = np.zeros((5, 5))
        else:
            feature1 = volume.acc_dist_index(high=self.OHLCV_df[d-1:t]["High"], low=self.OHLCV_df[d-1:t]["Low"], close=self.OHLCV_df[d-1:t]["Adj Close"],volume=self.OHLCV_df[d-1:t]['Volume'])
            feature2 = volatility.average_true_range(high=self.OHLCV_df[d-1:t]["High"], low=self.OHLCV_df[d-1:t]["Low"], close=self.OHLCV_df[d-1:t]["Adj Close"])
            feature3 = trend.macd(close=self.OHLCV_df[d-1:t]["Adj Close"], n_slow=self.window_size, n_fast=self.window_size//2,fillna=True)
            feature4 = momentum.rsi(close=self.OHLCV_df[d-1:t]["Adj Close"])
            feature5 = others.daily_return(close=self.OHLCV_df[d-1:t]["Adj Close"])
            state = pd.concat([feature1, feature2,feature3,feature4,feature5],axis=1).values[-5:,:]
            # block = self.timeseries[d-1:t]
            # state =  np.diff(np.log(block))
        state = state.flatten()
        state = np.append(state, (self.inventory, self.cash))

        return state

    def get_transaction_cost(self, amount):
        """
        return total cost/money and transaction fee for each buy or sell action
        """
        transaction_cost = self.transaction_cost_rate * amount * self.timeseries[self.t]
        # set minmum transaction_cost is $5
        if transaction_cost < 5 and amount > 0:
            transaction_cost = 5

        return transaction_cost

    def buy(self, action):
        """
        helper function: buy the asset
        """
        buy_num = 100
        transaction_cost = self.get_transaction_cost(buy_num)
    
        self.inventory += buy_num
        self.market_value += self.timeseries[self.t] * buy_num
        self.cash -= buy_num * self.timeseries[self.t] + transaction_cost
        self.states_buy.append(self.total_t)
        self.transaction_cost_history.append(transaction_cost)
        return buy_num

    def sell(self, action):
        """
        helper function: sell the position
        """
        sell_num = 100
        transaction_cost = self.get_transaction_cost(sell_num)
        
        self.cash += sell_num * self.timeseries[self.t] - transaction_cost
        self.market_value = 0
        self.states_sell.append(self.total_t)
        self.transaction_cost_history.append(transaction_cost)
        self.inventory -= sell_num
        return sell_num

    def step(self, action, verbose):
        """
        return a new state, reward, done (finished) given an action
        can add trading signal here when buy or sell
        """
        action = self.action_amount[action]
        amount = 0
        a = "buy" if action > 0 else "sell"
        if action > 0:
            amount = self.buy(action)
            if verbose:
                print("on {}: buy {} unit at price {}".format(self.date[self.t], amount, self.timeseries[self.t]))

        elif action < 0:
            amount = self.sell(action)
            if verbose:
                print("on {}: sell {} unit at price {}, total balance {}".format(
                    self.date[self.t], amount, self.timeseries[self.t], self.cash))

        self.market_value = self.timeseries[self.t] * self.inventory
        self.total_value = self.market_value + self.cash
        self.total_profit = self.total_value - self.init_money

        self.portfolio_daily_rtns.append((self.total_value - self.pre_value) / self.pre_value)
        self.portfolio_cumulative_rtns.append(self.total_profit / self.init_money)
        self.stock_cumulative_rtns.append((self.timeseries[self.t] - self.timeseries[0]) / self.timeseries[0])
        self.daily_values.append(self.total_value)
        
        reward = (1+np.sign(action)*(self.timeseries[self.t+1]/self.timeseries[self.t]-1))*(self.timeseries[self.t]/self.timeseries[self.t-5])
        # reward = -action*self.timeseries[self.t]-self.get_transaction_cost(amount)
        # add the action to the logger
        if amount != 0:
            self.logs.append([self.date[self.t + 1],a,action,amount,self.inventory,
                              self.timeseries[self.t],self.cash,
                              self.market_value,self.total_value,self.transaction_cost_history[-1]])

        done = False
        self.pre_value = self.total_value
        self.t += 1
        self.total_t += 1

        if self.t == len(self.date) - 1:
            done = True
        s_ = self.get_state(self.t)

        return s_, reward, done

    def get_daily_net_values(self):
        """
        return the daily total value of the portfolio with transaction fees
        """
        daily_net_values = (self.init_money * (1 + np.array(self.portfolio_cumulative_rtns)))
        df = pd.DataFrame({"daily_net_values": daily_net_values}, index=self.date)
        return df

    def get_daily_gross_values(self):
        """
        return the daily total value of the portfolio without transaction fees
        """
        daily_t_cost = np.zeros(self.timeseries.shape[0])
        daily_t_cost[self.states_buy + self.states_sell] = self.transaction_cost_history
        daily_gross_values = self.get_daily_net_values().iloc[:,0].values + daily_t_cost.cumsum()
        df = pd.DataFrame({"daily_net_values": daily_gross_values}, index=self.date)
        return df

    def save_logger(self):
        logger = pd.DataFrame(self.logs,
                              columns=["Dates","Action","Ratio","Amount","Position","Price",
                                       "Cash","Market Value","Total Value","T Cost"])
        logger.to_csv(path + "results/logger.csv")

    def draw(self, sp500):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize = (15,5))
        plt.plot(self.total_date, self.total_timeseries, color='grey', lw=2.)
        plt.plot(self.total_date, self.total_timeseries, '^', alpha=0.5, markersize=8, color='g', label = 'buying signal', markevery = self.states_buy)
        plt.plot(self.total_date,self.total_timeseries, 'v', alpha=0.5, markersize=8, color='r', label = 'selling signal', markevery = self.states_sell)
        plt.title('total profits %.2f, market value %.2f'%(self.total_profit, self.market_value))
        plt.legend()
        plt.savefig(path + 'results/trading_signals.png')
        plt.close()
        try:
            fig = plt.figure(figsize = (15,5))
            plt.plot(self.get_daily_gross_values(), label='my portfolio - gross', alpha=0.5)
            plt.plot(self.get_daily_net_values(), label='my portfolio - net', alpha=0.5)
            stock_values = (self.init_money / self.OHLCV_df['Adj Close'].iloc[0]) * self.OHLCV_df['Adj Close']
            # sp500_values = (self.init_money / sp500.iloc[0]) * sp500
            # plt.plot(sp500_values, label='SP500')

            plt.plot(stock_values, label='Stock')
            plt.title('start from: $%d, SP500: $%.2f, my portfolio: $%.2f'%(self.init_money, stock_values[-1], self.total_value))
            plt.legend()
            plt.savefig(path + 'results/performance.png')
            plt.close()

            fig = plt.figure(figsize = (15,5))
            plt.plot(np.array(self.transaction_cost_history).cumsum(), label='transaction cost')
            plt.title('avg cost per trade: $%f'%(np.mean(np.array(self.transaction_cost_history))))
            plt.legend()
            plt.savefig(path + 'results/transaction_costs.png')
            plt.close()
        except:
            pass
