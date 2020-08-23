import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import torch.nn.functional as F
import numpy as np
from configs.inputs import path

np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
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


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_states,
            learning_rate,
            reward_decay,
            e_greedy,
            replace_target_iter,
            memory_size,
            batch_size,
            e_greedy_increment,
            hidden_layer,
            hidden_units
    ):
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = learning_rate
        self.gamma = torch.tensor(reward_decay, dtype=torch.float)
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0             # for target updating
        self.memory_counter = 0                 # for storing memory
        self.memory = np.zeros((self.memory_size, self.n_states * 2 + 2))    # initialize memory

        self.eval_net = Net(hights=5, widths=5, outs=self.n_actions).to(device)
        self.target_net = Net(hights=5, widths=5, outs=self.n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

        self.cost_his = []

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = transition
        self.memory_counter += 1

    def choose_action(self, state, train=True):
        if np.random.uniform() < self.epsilon or not train:
            with torch.no_grad():
                return torch.argmax(self.eval_net(torch.from_numpy(state.reshape((1,1,5,5))).float()))

        return np.random.randint(0, self.n_actions)

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.memory_size, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states]).to(device)
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:,self.n_states+1:self.n_states+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:]).to(device)

        q_eval = self.eval_net(b_s.reshape(self.batch_size,1,5,5)).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_.reshape(self.batch_size,1,5,5)).detach()     # detach from graph, don't backpropagate

        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        self.cost_his.append(loss)

    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('loss')
        plt.xlabel('training steps')
        plt.savefig('results/loss.png')
        plt.close()
