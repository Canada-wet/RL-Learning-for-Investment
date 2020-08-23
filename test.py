import pandas as pd
import yfinance as yf
from datetime import date
import numpy as np
pd.options.plotting.backend = "plotly"
SPY_Dat = pd.read_csv("data/SPY.csv")
timeseries = SPY_Dat['Adj Close']
timeseries.plot()
# window_size = 50
# pre_action= 1
#
#
# df = SPY_Dat.set_index("Date")
# df[(df.index>'2000-01-04') & (df.index<'2020-03-01')]
#
#
# s = np.linspace(0, 1,n_action//2+1)
# s = np.append(s, -s[1:])
# s
#
# n_action=5
# for action in range(3, n_action, 2):
# n_action=21
# s = np.linspace(0, 1,n_action//2+1)
# s = np.append(s, -s[1:])
# print(s[11], s.shape)
#
# importing the libraries

import numpy as np
# for reading and displaying images
import matplotlib.pyplot as plt
#%matplotlib inline

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

class Net(Module):
    def __init__(self,K_size=2,stride_size=1,padding_size=0,hights=5,widths=5,outs=5):
        super(Net, self).__init__()

        self.K_h = K_size[0] if isinstance(K_size, tuple) else K_size
        self.K_w = K_size[1] if isinstance(K_size, tuple) else K_size
        self.Stride_h = stride_size[0] if isinstance(stride_size, tuple) else stride_size
        self.Stride_w = stride_size[1] if isinstance(stride_size, tuple) else stride_size
        self.Padding_h = padding_size[0] if isinstance(padding_size, tuple) else padding_size
        self.Padding_w = padding_size[1] if isinstance(padding_size, tuple) else padding_size

        self.H = hights
        self.W = widths
        self.out_size = outs

        self.cnn_output_h = int((self.H + 2 * self.Padding_h - (self.K_h - 1) - 1)/self.Stride_h + 1)
        self.cnn_output_w = int((self.W + 2 * self.Padding_w - (self.K_w - 1) - 1)/self.Stride_w + 1)

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 1, kernel_size = (self.K_h,self.K_w), stride = (self.Stride_h,self.Stride_w), padding = (self.Padding_h,self.Padding_w)),
            BatchNorm2d(1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size = (1, self.cnn_output_w),stride = (1, self.cnn_output_w))
        )

        self.linear_layers = Sequential(
            Linear(self.cnn_output_h, int(self.out_size))
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


# defining the model
model = Net(2,4,1,5,5,5);
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.07)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()


input = np.random.randn(1, 1, 5, 5)
print(model(torch.from_numpy(input).float()).detach())
print(model)

risk_free = pd.read_csv("data/US3M.csv")
risk_free
risk_free.index = pd.to_datetime(risk_free.index.strftime('%Y-%m'))
risk_free = risk_free.reindex(OHLCV_df.index, method='ffill')
