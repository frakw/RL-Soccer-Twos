import os

from gym_unity.envs import ActionFlattener
import numpy as np
import torch
from soccer_twos import AgentInterface

from .model import MAPOCA


class TeamAgent(AgentInterface):
    """
    An agent definition for policies trained with DQN on `team_vs_policy` variation with `single_player=True`.
    """

    def __init__(self, name, model, device):
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