import numpy as np
from configs.inputs import path

def moving_average(timeseries, window_size) :
    ret = np.cumsum(timeseries, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size


if __name__ == '__main__':
    import pandas as pd
    import os
    timeseries = pd.read_csv(path + "data/^GSPC.csv").Close.values
    ma_50 = moving_average(timeseries, 50)
    ma_20 = moving_average(timeseries, 20)
