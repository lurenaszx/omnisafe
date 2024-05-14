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
"""MultiAgent OnPolicy Adapter for OmniSafe."""

from __future__ import annotations

import random
from typing import Any

import numpy
import numpy as np
import torch
from rich.progress import track

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils.config import Config
from omnisafe.algorithms.base_algo import BaseAlgo

class MultiOnPolicyAdapter(OnlineAdapter):
    """MultiOnPolicy Adapter for OmniSafe in the multi-agent environment.

    :class:`MultiOnPolicyAdapter` is used to adapt the Multi-agent environment to the on-policy training.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize an instance of :class:`OnPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._reset_log()

    @property
    def num_players(self) -> int:
        """The observation space of the environment."""
        return self._env.number_of_players

    def init_value(self):
        return torch.zeros((self._num_env,), dtype=torch.float32)

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: list[ConstraintActorCritic | ConstraintActorQCritic],
        buffer: list[VectorOnPolicyBuffer],
        logger: Logger,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            steps_per_epoch (int): Number of steps per epoch.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor , reward critic
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        # Only accept one environment
        def wrap_step(tem_agent, tem_obs):
            if isinstance(agent, ConstraintActorCritic):
                return tem_agent.step(tem_obs)
            else:
                return tem_agent.step(tem_obs), *(torch.tensor([0], dtype=torch.float) for _ in range(3))

        self._reset_log()

        obs, info = self.reset()
        obs_list, act_list, value_r_list, value_c_list, logp_list = ([None for _ in range(self.num_players)],
                            [None for _ in range(self.num_players)], [None for _ in range(self.num_players)],
                            [None for _ in range(self.num_players)], [None for _ in range(self.num_players)],)
        reward_list = [self.init_value() for _ in range(self.num_players)]
        cost_list = [self.init_value() for _ in range(self.num_players)]
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            current_players = info['current_players']
            total_act = []
            for player_id in current_players:
                if act_list[player_id] is not None:
                    buffer[player_id].store(
                        obs=obs_list[player_id],
                        act=act_list[player_id],
                        reward=reward_list[player_id],
                        cost=cost_list[player_id],
                        value_r=value_r_list[idx],
                        value_c=value_c_list[idx],
                        logp=logp_list[idx],
                    )
                    reward_list[player_id], cost_list[player_id] = self.init_value(), self.init_value()
                act, value_r, value_c, logp = wrap_step(agent[player_id], obs[:, player_id].unsqueeze(1))
                # Add random to ensure the exploration
                # print(act)
                if random.random() < 0.05:
                    act = torch.as_tensor([[self._env.action_space.sample()]])
                total_act.append(act)
                act_list[player_id], value_r_list[player_id], value_c_list[player_id], logp_list[player_id] = (
                    act, value_r, value_c, logp)
                obs_list[player_id] = obs[:, player_id]

            next_obs, reward, cost, terminated, truncated, info = self.step(torch.as_tensor(total_act))
            for player_id in range(self.num_players):
                # print(reward_list, reward[:, player_id])
                reward_list[player_id] += reward[:, player_id]
                cost_list[player_id] += cost[:, player_id]
            self._log_value(reward=reward, cost=cost, info=info)  #ToDo to add

            # for player_id in current_players:
            #     if self._cfgs.algo_cfgs.use_cost:
            #         logger[player_id].store({'Value/cost': value_c_list[player_id]})
            #     logger[player_id].store({'Value/reward': value_r_list[player_id]})

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            if epoch_end:
                num_dones = int(terminated.contiguous().sum())
                if self._env.num_envs - num_dones:
                    logger.log(
                        f'\nWarning: trajectory cut off when rollout by epoch\
                            in {self._env.num_envs - num_dones} of {self._env.num_envs} environments.',
                    )

            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    if done or time_out:
                        self._log_metrics(logger, agent, idx)
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0
                    for player_id in range(self.num_players):
                        last_value_r = torch.zeros(1)
                        last_value_c = torch.zeros(1)
                        if not done:
                            if epoch_end:
                                _, last_value_r, last_value_c, _ = wrap_step(agent[player_id], obs[idx][player_id])
                            if time_out:
                                _, last_value_r, last_value_c, _ = wrap_step(agent[player_id],
                                                                             info['final_observation'][idx][player_id])
                            if isinstance(agent, ConstraintActorCritic):
                                last_value_r = last_value_r.unsqueeze(0)
                                last_value_c = last_value_c.unsqueeze(0)
                        if act_list[player_id] is not None:
                            buffer[player_id].store(
                                obs=obs_list[player_id],
                                act=act_list[player_id],
                                reward=reward_list[player_id],
                                cost=cost_list[player_id],
                                value_r=value_r_list[player_id],
                                value_c=value_c_list[player_id],
                                logp=logp_list[player_id],
                            )
                            reward_list[player_id], cost_list[player_id] = self.init_value(), self.init_value()
                        buffer[player_id].finish_path(last_value_r, last_value_c, idx)
                    act_list = [None for _ in range(self.num_players)]

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward and cost will
            be stored in ``info['original_reward']`` and ``info['original_cost']``.

        Args:
            reward (torch.Tensor): The immediate step reward.
            cost (torch.Tensor): The immediate step cost.
            info (dict[str, Any]): Some information logged by the environment.
        """
        for player_id in range(self.num_players):
            self._ep_ret[player_id] += info.get('original_reward', reward).cpu()[:, player_id]
            self._ep_cost[player_id] += info.get('original_cost', cost).cpu()[:, player_id]
        self._ep_len += 1

    def _log_metrics(self, logger: Logger, agents: list[ConstraintActorQCritic], idx: int) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            idx (int): The index of the environment.
        """
        if hasattr(self._env, 'spec_log'):
            self._env.spec_log(logger, agents)
        logger.store({'Metrics/EpLen': self._ep_len[idx]})
        for player_id in range(self.num_players):
            logger.store(
                {
                    f'Metrics/EpRet_{player_id}': self._ep_ret[player_id][idx],
                    f'Metrics/EpCost_{player_id}': self._ep_cost[player_id][idx],
                },
            )

    def _reset_log(self, idx: int | None = None) -> None:
        """Reset the episode return, episode cost and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        if idx is None:
            self._ep_ret = torch.zeros((self.num_players, self._env.num_envs))
            self._ep_cost = torch.zeros((self.num_players, self._env.num_envs))
            self._ep_len = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[:, idx] = 0.0
            self._ep_cost[:, idx] = 0.0
            self._ep_len[idx] = 0.0



