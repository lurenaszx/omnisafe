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
"""Implementation of the Multi-Agent On-Policy Wrapper."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.adapter import MultiOnPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import distributed
from omnisafe.utils.config import Config
from omnisafe.utils.tools import get_device, seed_all

@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods,line-too-long
class MultiOnPolicyWrapper:
    """The Multi Agent Wrapper for different base algorithm.
    """
    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.MultiOnPolicyAdapter` to adapt the environment to the
        algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
        """
        self._env: MultiOnPolicyAdapter = MultiOnPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init_model(self) -> None:
        """Initialize the agents according to the number of agents and its algo type.

        OmniSafe uses :class:`omnisafe.models.actor_critic.constraint_actor_critic.ConstraintActorCritic`
        as the default model.

        User can customize the model by inheriting this method.

        Examples:
            >>> def _init_model(self) -> None:
            ...     self._actor_critic = CustomActorCritic()
        """
        algo_class = registry.get(self._algo)
        self._agents = [algo_class(
            env_id=self._env_id,
            cfgs=self._cfgs,
        ) for i in range(self._num_players)]
        self._models = [agent.actor_critic for agent in self._agents]

    def __init__(self, algo: str, env_id: str, cfgs: Config) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()
        """
        self._env_id: str = env_id
        self._cfgs: Config = cfgs
        self._algo = algo

        assert hasattr(cfgs, 'seed'), 'Please specify the seed in the config file.'
        self._seed: int = int(cfgs.seed) + distributed.get_rank() * 1000
        seed_all(self._seed)

        assert hasattr(cfgs.train_cfgs, 'device'), 'Please specify the device in the config file.'
        self._device: torch.device = get_device(self._cfgs.train_cfgs.device)

        distributed.setup_distributed()

        self._init_env()
        self._num_players = self._env.num_players
        self._init_model()

        self._init_log()
        self._num_players = self._cfgs.env_cfgs.num_players
        self._buf: list[VectorOnPolicyBuffer] = [agent.buf for agent in self._agents]

    def _init_log(self) -> None:
        """Log info about epoch.

        +-----------------------+----------------------------------------------------------------------+
        | Things to log         | Description                                                          |
        +=======================+======================================================================+
        | Train/Epoch           | Current epoch.                                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpCost_id     | Average cost of the epoch for every player.                          |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpRet_id      | Average return of the epoch for every player.                        |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpLen         | Average length of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_pi_id       | Loss of the policy network for every player.                         |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_cost_critic_id| Loss of the cost critic network for every player.                  |
        +-----------------------+----------------------------------------------------------------------+
        | Train/Entropy_id      | Entropy of the policy network for every player.                      |
        +-----------------------+----------------------------------------------------------------------+
        | Train/StopIters_id    | Number of iterations of the policy network for every player.         |
        +-----------------------+--------------------------------------------------------------------
        | Train/LR_id           | Learning rate of the policy network for every player.                |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/Seed             | Seed of the experiment.                                              |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/TotalEnvSteps    | Total steps of the experiment.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Time                  | Total time.                                                          |
        +-----------------------+----------------------------------------------------------------------+
        | FPS                   | Frames per second of the epoch.                                      |
        +-----------------------+----------------------------------------------------------------------+
        """
        self._logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: dict[str, Any] = {}
        for player_id in range(self._num_players):
            what_to_save[f'pi_{player_id}'] = self._models[player_id].actor
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        self._logger.register_key(
            'Metrics/EpLen',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )

        self._logger.register_key('Train/Epoch')

        self._logger.register_key('TotalEnvSteps')
        for player_id in range(self._num_players):
            self._logger.register_key(
                f'Metrics/EpRet_{player_id}',
                window_length=self._cfgs.logger_cfgs.window_lens
            )
            self._logger.register_key(
                f'Metrics/EpCost_{player_id}',
                window_length=self._cfgs.logger_cfgs.window_lens
            )
            self._logger.register_key(f'Train/Entropy_{player_id}')
            self._logger.register_key(f'Train/LR_{player_id}')
            self._logger.register_key(f'Train/KL_{player_id}')
            self._logger.register_key(f'Train/StopIter_{player_id}')
            # log information about actor
            self._logger.register_key(f'Loss/Loss_pi_{player_id}', delta=True)

            # log information about critic
            self._logger.register_key(f'Loss/Loss_reward_critic_{player_id}', delta=True)

            if self._cfgs.algo_cfgs.use_cost:
                # log information about cost critic
                self._logger.register_key(f'Loss/Loss_cost_critic_{player_id}', delta=True)

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')

        # register environment specific keys
        for env_spec_key in self._env.env_spec_keys:
            self._logger.register_key(env_spec_key)

    def learn(self) -> tuple[list[float], list[float], float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: Average episode return in final epoch.
            ep_cost: Average episode cost in final epoch.
            ep_len: Average episode length in final epoch.
        """
        start_time = time.time()
        self._logger.log('INFO: Start training')

        for epoch in range(self._cfgs.train_cfgs.epochs):
            epoch_time = time.time()

            rollout_time = time.time()
            self._env.rollout(
                steps_per_epoch=self._steps_per_epoch,
                agent=self._models,
                buffer=self._buf,
                logger=self._logger,
            )
            self._logger.store({'Time/Rollout': time.time() - rollout_time})

            update_time = time.time()
            self._update()
            self._logger.store({'Time/Update': time.time() - update_time})

            self._logger.store(
                {
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
                epoch + 1
            ) == self._cfgs.train_cfgs.epochs:
                self._logger.torch_save()
        ep_ret = [self._logger.get_stats(f'Metrics/EpRet_{i}')[0] for i in range(self._num_players)]
        ep_cost = [self._logger.get_stats(f'Metrics/EpCost_{i}')[0] for i in range(self._num_players)]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len

    def _update(self) -> None:
        # Update ActorCritic for all agents
        for player_id, agent in enumerate(self._agents):
            agent._update(self._logger, player_id)


    @property
    def logger(self):
        return self._logger
