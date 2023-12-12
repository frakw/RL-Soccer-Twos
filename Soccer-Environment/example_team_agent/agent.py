import os
import time
import soccer_twos
import time
import re
import numpy as np
from soccer_twos import AgentInterface


import torch
import torch.nn as nn
import torch.nn.functional as F

class SoccerModel(nn.Module):
    def __init__(self):
        super(SoccerModel, self).__init__()
        self.body_encoder = nn.Sequential(
            nn.Linear(336, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.branches = nn.ModuleList([
            nn.Linear(512, 3),
            nn.Linear(512, 3),
            nn.Linear(512, 3)
        ])

    def forward(self, x):
        x = self.body_encoder(x)
        outputs = [branch(x) for branch in self.branches]
        return outputs

model = SoccerModel()
print(model)
input = torch.randn(1, 336)
output = model(input)
print(output)
# 動作output是一個list，裡面有三個元素，每個元素是一個tensor，tensor的shape是(1, 3)
# 要再把她轉換回numpy array，shape是(27, )

# 選出每個list中最大的那個tensor，再把他轉換成numpy array
# 這樣就可以得到一個shape是(3, )的numpy array
for i in range(len(output)):
    output[i] = output[i].max(1)[1].numpy()
action = 0
for i in range(len(output)):
    action += output[i][0] * 3 * i
print("action:", action)

class Agent(AgentInterface):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def act(self, obs):
        obs = torch.tensor(obs).float().to(self.device)
        output = self.model(obs)
        # print(output)
        for i in range(len(output)):
            output[i] = output[i].argmax().cpu().detach().numpy()
        
        action = output[0]*9 + output[1]*3 + output[2]
        
        return action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SoccerModel().to(device)
model.load_state_dict(torch.load("SoccerTwos.pth"))
agent1 = Agent(model, device)
agent2 = Agent(model, device)
agent3 = Agent(model, device)
agent4 = Agent(model, device)


def environment():
    env = soccer_twos.make(
        render=True,
        watch=True,
        flatten_branched=True,
        variation=soccer_twos.EnvType.multiagent_player,
        opponent_policy=lambda *_: 0,
    )
    return env

env = environment()
while 1:
    
    state = env.reset()
    done = False
    with torch.no_grad():
        while 1:
            action1 = agent1.act(state[0])
            action2 = agent2.act(state[1])
            action3 = agent3.act(state[2])
            action4 = agent4.act(state[3])
            state, r, done, info = env.step({0: action1, 1: action2, 2: action3, 3: action4})
            
            #if done["__all__"]:
            #    env.close()
            #    break