# Reinforcement Learning Trading System - V1.1.0

Clone the git repo, then install the requirements with pip

```

git clone https://github.com/shin2suka/RL-Trading.git
cd RL-Trading
pip install -r requirements.txt

```

Run the trading system

```

python backtest_main.py

```

Current algorithm: Deep Q learning


### TO DO:
Urgent:
- Deploy CNN network
- Design a bette reward mechanism and the state space (NEED TO FIX *profit_per_episode* plot)
- Simulate several scenarios (e.g., up/down trend and mean reversion)
- Incorporate with performance engine to track stats

Non-urgent:

- Add more RL algorithms (policy gradient/PPO/A3C)
- Incorporate with dash visualization
- Build risk engine to control risk
- Build trading signals
- enable to hold multiple assets

## Reference
[Algorithm Trading using Q-Learning and Recurrent Reinforcement Learning](http://cs229.stanford.edu/proj2009/LvDuZhai.pdf)

[Learning to Trade via Direct Reinforcement](https://ieeexplore.ieee.org/document/935097)

&copy;2020 All Rights Reserved by Hongyi (Henry) Wu & [Tianyu (Shin) Ren](https://shin2suka.github.io/) & Zhizu (Zale) Li
