import os
import glob
import pandas as pd
from statistics import mean
from itertools import chain
from configs.inputs import path
import datetime
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


def data_dispose(ticket, Nmins=1080, Nsims=1):
    if ticket != 'simulation':
        try:
            os.chdir(path + "data/intraday/" + ticket)
        except:
            raise Exception("This ticker is not found in intradata set, try daily data!")
        extension = 'csv'
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

        combined_csv = pd.concat( [ datetime_converter(pd.read_csv(f, usecols = [0,6,8], names=["timestamp", "volume", "price"]), f)\
         for f in all_filenames ], ignore_index=True )
    elif ticket == 'simulation':
        combined_csv = SimulatePath(Nmins,Nsims)
        fig = plt.figure(figsize = (15,5))
        plt.plot(combined_csv['timestamp'], combined_csv['price'])
        plt.title('Price Trend before OHLCV')
        # plt.legend()
        plt.savefig(path + 'results/price_trend_before_OHLCV.png')

        OHLCV(combined_csv, 60)
        combined_csv.index = pd.to_datetime(combined_csv.index, unit = 'ms')

        fig = plt.figure(figsize = (15,5))
        plt.plot(combined_csv.index, combined_csv['Adj Close'])
        plt.title('Price Trend after OHLCV')
        # plt.legend()
        plt.savefig(path + 'results/price_trend_after_OHLCV.png')

    return combined_csv



def datetime_converter(df, file):
    try:
        time = file[-12:-4]
        date = datetime.datetime.strptime(time, "%Y%m%d")
        df["timestamp"] = df["timestamp"].apply(lambda x: int(x + date.timestamp() * 1000))
        return df
    except:
        raise Exception("Fail to convert datetime!")
        return 0


def OHLCV(df, n):
    # df["price"] = df["price"] / 100000;
    df.insert(2, "Volume",get_volumn(df["volume"].to_numpy(),n), True)
    df.insert(2, "Adj Close",get_close(df["price"].to_numpy(),n), True)
    df.insert(2, "Low",get_low(df["price"].to_numpy(),n), True)
    df.insert(2, "High",get_high(df["price"].to_numpy(),n), True)
    df.insert(2, "Open",get_open(df["price"].to_numpy(),n), True)
    df.index = df.index + 1
    df.drop(df[df.index % n != 0].index,inplace = True)
    df.drop(columns=["volume", "price"], inplace = True)
    df.set_index("timestamp", inplace=True, drop=True)

def get_open(col, n):
    return list(chain.from_iterable([col[i]]*n for i in range(0,len(col),n)))[0:len(col)]

def get_high(col, n):
    return list(chain.from_iterable([max(col[i:i+n])]*n for i in range(0,len(col),n)))[0:len(col)]

def get_low(col, n):
    return list(chain.from_iterable([min(col[i:i+n])]*n for i in range(0,len(col),n)))[0:len(col)]

def get_close(col, n):
    return list(chain.from_iterable([col[-1]]*n if len(col) <= (i+n-1) else [col[i+n-1]]*n for i in range(0,len(col),n)))[0:len(col)]

def get_volumn(col, n):
    return list(chain.from_iterable([sum(col[i:i+n])]*n for i in range(0,len(col),n)))[0:len(col)]


def SimulatePath(Nmins, Nsims):
    dt = 1/60 # measure time in minutes

    T = 60*dt*Nmins
    t = np.linspace(1,T,60*Nmins) # length is 1 less than that in MATLAB
    Nsteps = len(t)

    # volume model parameters
    kappa = 5
    theta = 1
    eta = np.array([[2,1], [1,2]])
    lot_size = 100

    # price movement
    b = 0.0001
    gamma = 0.05
    tick_size = 0.01

    # for output
    S = np.empty((Nsims, Nsteps))
    lambda_buy = np.empty((Nsims, Nsteps))
    lambda_sell = np.empty((Nsims, Nsteps))
    V_buy = np.empty((Nsims, Nsteps))
    V_sell = np.empty((Nsims, Nsteps))
    N_up = np.empty((Nsims, Nsteps))
    N_down = np.empty((Nsims, Nsteps))
    V = np.empty((Nsims, Nsteps,2))
    N = np.empty((Nsims, Nsteps,2))
    # initial asset price
    S[:,0] = np.random.uniform(9.5,10.5)

    lambda_buy[:,0] = theta
    lambda_sell[:,0] = theta

    for i in range(Nsteps-1):

        # generate buy and sell volume
        V_buy[:,i] = np.random.poisson(lot_size*lambda_buy[:,i] )
        V_sell[:,i] = np.random.poisson(lot_size*lambda_sell[:,i] )
        # print(V_buy[:,i],V_sell[:,i])
        # *** generate price movement ***
        # use buy volume to drive up moves + idiosyncratic
        N_up[:,i] = np.random.poisson( b * V_buy[:,i] ) + np.random.poisson(gamma, [1,Nsims])

        # use sell volume to drive up moves + idiosyncratic
        N_down[:,i] = np.random.poisson( b * V_sell[:,i] ) + np.random.poisson(gamma, [1,Nsims])

        S[:,i+1] = S[:,i] + tick_size*(N_up[:,i]-N_down[:,i])

        # ********************************

        # **** update intensity using Hawkes model ****
        lambda_tot = lambda_buy[:,i]+lambda_sell[:,i]
        p_buy = lambda_buy[:,i] / lambda_tot

        # decide if a jump occurs
        J = (np.random.uniform(0,1) < 1-np.exp(-lambda_tot*dt))
        # decide whether it is for buy
        H = (np.random.uniform(0,1) < p_buy)

        lambda_buy[:,i+1] = (lambda_buy[:,i]-theta)*np.exp(-kappa*dt) + theta+ J * ( eta[0,0] * H + eta[0,1] * (1-H) )

        lambda_sell[:,i+1] = (lambda_sell[:,i]-theta)*np.exp(-kappa*dt) + theta+ J * ( eta[1,0] * H + eta[1,1] * (1-H) )



        # lambda(:,:,0) = lambda_buy
        # lambda(:,:,1) = lambda_sell
    V[:,:,0] = V_buy
    V[:,:,1] = V_sell
    N[:,:,0] = N_up
    N[:,:,1] = N_down


    t=t.reshape(Nmins*60,1)
    t = pd.DataFrame(t,columns=['timestamp'])
    V_total = V[0,:,0]+V[0,:,1]
    V_total = pd.DataFrame(V_total,columns=['volume'])
    S1 = S[0,:]
    S1 = pd.DataFrame(S1,columns=['price'])
    df = t.join(S1).join(V_total)
    df.set_index("timestamp",inplace=True)

    for i in range(Nsims):
      if i > 0:
        # print(i)
        # print(S.shape)
        V_total = V[i,:,0]+V[i,:,1]
        V_total = pd.DataFrame(V_total,columns=['volume'])
        S_temp = S[i,:]
        S_temp = pd.DataFrame(S_temp,columns=['price'])
        df_temp = t.join(S_temp).join(V_total)
        df_temp.set_index("timestamp",inplace=True)
        df1 = pd.concat([df['price'], df_temp['price']], ignore_index=True)
        df2 = pd.concat([df['volume'], df_temp['volume']], ignore_index=True)
        df = pd.DataFrame([df1,df2]).T
    # return t,S,V
    df = df.reset_index()
    df.columns = ['timestamp','price','volume']
    return df
