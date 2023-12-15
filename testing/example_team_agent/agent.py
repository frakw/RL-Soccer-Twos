import os
import gym
from gym_unity.envs import ActionFlattener
import numpy as np
import torch
from soccer_twos import AgentInterface

from .model import MAPOCA

class SingleAgent:
    def __init__(self,name ,model,device):
        self.model = model
        self.device = device
        self.name = name

    def act(self, observation):
        observation = torch.tensor(observation).float().to(self.device)
        output = self.model(observation)
        for i in range(len(output)):
            output[i] = output[i].argmax().cpu().detach().numpy()
        return output


class TeamAgent(AgentInterface):
    """
    An agent definition for policies trained with DQN on `team_vs_policy` variation with `single_player=True`.
    """

    def __init__(self, env: gym.Env, team_name='unknown', team_model_file = "./models/best_model.pth"):
        self.name = team_name
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        team_model = MAPOCA().to(device)
        team_model.load_state_dict(torch.load(team_model_file))
        agent1 = SingleAgent(self.name + '_0', team_model, device)
        agent2 = SingleAgent(self.name + '_1', team_model, device)
        self.agents = {}
        self.agents[0] = agent1
        self.agents[1] = agent2
        

    def act(self, observation):
        """The act method is called when the agent is asked to act.
        Args:
            observation: a dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: a dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        actions = {}
        # for each team player
        for player_id in observation:
            action = self.agents[player_id].act(observation[player_id])
            actions[player_id] = action
        return actions