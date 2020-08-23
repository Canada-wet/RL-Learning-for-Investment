import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from configs.inputs import path

np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, n_states, n_actions, hidden_layer, hidden_units):
        super(Net, self).__init__()
        self.layer = hidden_layer
        self.fc1 = nn.Linear(n_states, hidden_units)
        self.hidden = nn.Linear(hidden_units, hidden_units)
        self.out = nn.Linear(hidden_units, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for _ in range(self.layer):
            x = F.relu(self.hidden(x))
        actions_value = self.out(x)
        return actions_value


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

        self.eval_net = Net(self.n_states, self.n_actions, hidden_layer, hidden_units).to(device)
        self.target_net = Net(self.n_states, self.n_actions, hidden_layer, hidden_units).to(device)
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
                # print (torch.argmax(self.eval_net(torch.from_numpy(state).float())))
                return torch.argmax(self.eval_net(torch.from_numpy(state).float()))
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

        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate

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
        print(self.cost_his)
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('loss')
        plt.xlabel('training steps')
        plt.savefig('results/loss.png')
        plt.close()
