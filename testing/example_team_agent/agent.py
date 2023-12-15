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
        """The act method is called when the agent is asked to act.
        Args:
            observation: a dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: a dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        observation = torch.tensor(observation).float().to(self.device)
        output = self.model(observation)
        for i in range(len(output)):
            output[i] = output[i].argmax().cpu().detach().numpy()
        
        action = output[0]*9 + output[1]*3 + output[2]
        
        return action


class TeamAgent(AgentInterface):
    """
    An agent definition for policies trained with DQN on `team_vs_policy` variation with `single_player=True`.
    """

    def __init__(self, env: gym.Env, team1_name='unknown', team1_model_file = "./models/best_model.pth",team2_name='unknown', team2_model_file = "./models/best_model.pth"):
        super().__init__()
        self.team1_name = team1_name
        self.team2_name = team2_name
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        team1_model = MAPOCA().to(device)
        team1_model.load_state_dict(torch.load(team1_model_file))
        team2_model = MAPOCA().to(device)
        team2_model.load_state_dict(torch.load(team2_model_file))
        team1_agent1 = SingleAgent(team1_name + '_0', team1_model, device)
        team1_agent2 = SingleAgent(team1_name + '_1', team1_model, device)
        team2_agent1 = SingleAgent(team2_name + '_0', team2_model, device)
        team2_agent2 = SingleAgent(team2_name + '_1', team2_model, device)

        self.agents = {}
        self.agents[0] = team1_agent1
        self.agents[1] = team1_agent2
        self.agents[2] = team2_agent1
        self.agents[3] = team2_agent2
        

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
            print("player_id " + str(player_id))
            action = self.agents[player_id].act(observation[player_id])
            actions[player_id] = action

        print("actions: ")
        print(actions)
        return actions