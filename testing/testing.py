import os
import time
import soccer_twos
import time
import re
import numpy as np
from example_team_agent.model import MAPOCA
from example_team_agent.agent import TeamAgent


import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

# usage python testing.py <pytorch model file 1> <pytorch model file 2>

def main():
    if len(sys.argv) != 3:
        print("no input pytorch model")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = MAPOCA().to(device)
    model1.load_state_dict(torch.load(sys.argv[1]))
    model2 = MAPOCA().to(device)
    model2.load_state_dict(torch.load(sys.argv[2]))
    agent1 = TeamAgent("blue_1",model1, device)
    agent2 = TeamAgent("blue_2",model1, device)
    agent3 = TeamAgent("orange_1",model2, device)
    agent4 = TeamAgent("orange_2",model2, device)

    env = soccer_twos.make(
        blue_team_name="blue_team",
        orange_team_name="orange_team",
        render=False,
        watch=True,
        flatten_branched=True,
        variation=soccer_twos.EnvType.multiagent_player,
        opponent_policy=lambda *_: 0,
    )
    while 1:
        state = env.reset()
        done = False
        with torch.no_grad():
            while not done:
                action1 = agent1.act(state[0])
                action2 = agent2.act(state[1])
                action3 = agent3.act(state[2])
                action4 = agent4.act(state[3])
                state, r, done, info = env.step({0: action1, 1: action2, 2: action3, 3: action4})

if __name__ == '__main__':
    main()