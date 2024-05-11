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
"""Implementation of the PPO algorithm."""

from __future__ import annotations

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.qpg import QPG


@registry.register
class RMPG(QPG):
    """The Proximal Policy Optimization (PPO) algorithm.

    References:
        - Title: Proximal Policy Optimization Algorithms
        - Authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov.
        - URL: `PPO <https://arxiv.org/abs/1707.06347>`_
    """

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        q_value: torch.Tensor
    ) -> torch.Tensor:
        r"""Computing pi/actor loss.
        """
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)
        ratio_cliped = torch.clamp(
            ratio,
            1 - self._cfgs.algo_cfgs.clip,
            1 + self._cfgs.algo_cfgs.clip,
        )
        loss = -torch.min(ratio * adv, ratio_cliped * adv).mean()
        loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()
        # useful extra info
        entropy = distribution.entropy().mean().item()
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                'Train/PolicyStd': std,
                'Loss/Loss_pi': loss.mean().item(),
            },
        )
        return loss

def compute_advantages(policy,
                       action_values,
                       use_relu=False,):
    """Compute advantages using pi and Q."""
    # Compute advantage.
    # Avoid computing gradients for action_values.
    action_values = action_values.detach()

    baseline = compute_baseline(policy, action_values)

    advantages = action_values - torch.unsqueeze(baseline, 1)
    if use_relu:
        advantages = F.relu(advantages)
    # print(action_values.size(), policy.size(), baseline.size(), advantages.size())
    # print(torch.concatenate([action_values, policy, baseline.unsqueeze(1), advantages], dim=1))
    # Compute advantage weighted by policy.
    policy_advantages = -torch.mul(policy, advantages.detach())
    # print(policy_advantages)
    return torch.sum(policy_advantages, dim=1)

def compute_baseline(policy, action_values):
  # V = pi * Q, backprop through pi but not Q.
  return torch.sum(torch.mul(policy, action_values.detach()), dim=1)
