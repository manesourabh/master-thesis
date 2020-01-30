import torch

import numpy as np
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel


class GymFfModel(torch.nn.Module):
    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=None,
            dueling=False
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        self.dueling = dueling
        self._obs_ndim = len(observation_shape)
        if dueling:
            self.head = DuelingHeadModel(int(np.prod(observation_shape)), hidden_sizes, action_size)
        else:
            self.head = MlpModel(int(np.prod(observation_shape)), hidden_sizes, action_size)

    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        q = self.head(observation.view(T * B, -1))

        q = restore_leading_dims(q, lead_dim, T, B)
        return q
