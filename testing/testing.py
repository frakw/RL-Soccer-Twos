import os
import time
import soccer_twos
import time
import re
import numpy as np
from model import MAPOCA
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

    env = soccer_twos.make(
        blue_team_name="blue_team",
        orange_team_name="orange_team",
        render=False,
        watch=True,
        flatten_branched=False,
        variation=soccer_twos.EnvType.multiagent_player,
        opponent_policy=lambda *_: 0,
    )

    team1_agent = TeamAgent(env, "blue_team", sys.argv[1])
    team2_agent = TeamAgent(env, "orange_team", sys.argv[2])
    while 1:
        state = env.reset()
        done = False
        with torch.no_grad():
            while not done:
                team1_action = team1_agent.act({0:state[0],1:state[1]})
                team2_action = team2_agent.act({0:state[2],1:state[3]})
                state, r, done, info = env.step({0: team1_action[0], 1: team1_action[1], 2: team2_action[0], 3: team2_action[1]})

if __name__ == '__main__':
    main()