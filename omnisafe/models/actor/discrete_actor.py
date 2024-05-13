# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of DiscreteActor."""

from __future__ import annotations

import gymnasium
import torch
import torch.nn as nn
from torch.distributions import Distribution, Categorical

from omnisafe.models.base import Actor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


# pylint: disable-next=too-many-instance-attributes
class DiscreteActor(Actor):
    """Implementation of DiscreteActor.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    _current_dist: Categorical

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        """Initialize an instance of :class:`GaussianLearningActor`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)

        self.net: nn.Module = build_mlp_network(
            sizes=[self._obs_dim, *self._hidden_sizes, self._act_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def _distribution(self, obs: torch.Tensor) -> Categorical:
        """Get the distribution of the actor.

        .. warning::
            This method is not supposed to be called by users. You should call :meth:`forward`
            instead.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The distribution over different actions.
        """
        logits_prob = self.net(obs)
        return Categorical(logits=logits_prob)

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict the action given observation.
        Currently deterministic has not been used.
        """
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        return self._current_dist.sample()

    def forward(self, obs: torch.Tensor) -> Distribution:
        """Forward method.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The prob of all actions.
        """
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        return self._current_dist

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of the action given the current distribution.

        .. warning::
            You must call :meth:`forward` or :meth:`predict` before calling this method.

        Args:
            act (torch.Tensor): Action from :meth:`predict` or :meth:`forward` .

        Returns:
            Log probability of the action.
        """
        assert self._after_inference, 'log_prob() should be called after predict() or forward()'
        self._after_inference = False
        return self._current_dist.log_prob(act).sum(axis=-1)

    def log_prob_all(self) -> torch.Tensor:
        """ compute the log probability of all actions"""
        assert self._after_inference, 'log_prob() should be called after predict() or forward()'
        self._after_inference = False
        return self._current_dist.logits

if __name__ == "__main__":
    obs_space = gymnasium.spaces.Discrete(4)
    action_space = gymnasium.spaces.Discrete(4)
    print(obs_space.sample())
    net = DiscreteActor(obs_space,
                        action_space,
                        hidden_sizes=[10])
    net(torch.tensor([2], dtype=torch.float))
    print(net.log_prob())
