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
"""Implementation of the RMPG algorithm."""

from __future__ import annotations

import torch

from omnisafe.common.logger import Logger
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.qpg import QPG
import torch.nn.functional as F
from torch import nn

@registry.register
class RMPG(QPG):
    """Regret Matching Policy Gradient

    References:
        - Title: Actor-Critic Policy Optimization in Partially Observable Multiagent Environments
        - Authors: Sriram Srinivasan, Marc Lanctot, Vinicius Zambaldi, Julien Pérolat, Karl Tuyls, Rémi Munosm,
        Michael Bowling
        - URL: `https://arxiv.org/abs/1810.09026_
    """

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        q_value: torch.Tensor,
        logger: Logger = None,
        player_id: int = None,
    ) -> torch.Tensor:
        r"""Computing pi/actor loss.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            q_value (torch.Tensor)
            logger, player_id: For multi-agent case.
        Returns:
            The loss of pi/actor.
        """
        distribution = self._actor_critic.actor(obs)
        logits = distribution.logits
        prob = nn.functional.softmax(logits, dim=1)
        q_value = q_value.detach()
        baseline = torch.sum(torch.mul(prob, q_value.detach()), dim=1)
        advantages = F.relu(q_value - torch.unsqueeze(baseline, 1))
        policy_advantages = torch.sum(-torch.mul(prob, advantages.detach()))

        loss = torch.mean(policy_advantages, dim=0)
        entropy = torch.distributions.Categorical(prob).entropy().mean().item()
        if logger is None:
            self._logger.store(
                {
                    'Train/Entropy': entropy,
                    'Loss/Loss_pi': loss.mean().item(),
                },
            )
        else:
            logger.store(
                {
                    f'Train/Entropy_{player_id}': entropy,
                    f'Loss/Loss_pi_{player_id}': loss.mean().item(),
                }
            )
        return loss
