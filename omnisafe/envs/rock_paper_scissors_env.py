from copy import copy
from itertools import dropwhile
from typing import List
import enum
import numpy as np
from gymnasium.spaces import Discrete, Box
from omnisafe.envs.core import CMDP, env_register
from typing import Any, ClassVar
import torch
from omnisafe.typing import DEVICE_CPU
import random

class ActionType(enum.Enum):
    Rock = 0
    Paper = 1
    Scissor = 2

@env_register
class RockPaperScissorsEnv(CMDP):
    '''
    Implementation of Kuhn's poker in accordance to OpenAI gym environment interface.
    '''

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False

    _support_envs: ClassVar[list[str]] = [
        'RockPaperScissors-v0',
    ]

    reward_table = [
        [[0, 0], [-1, 1], [1, -1]],
        [[1, -1], [0, 0], [-1, 1]],
        [[-1, 1], [1, -1], [0, 0]],
    ]

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,
    ) -> None:
        '''
        :param number_of_players: Number of players (Default 2).
        '''
        self._num_envs = num_envs
        self.done = False
        self.number_of_players = 2
        self._device = device
        self._action_space = Discrete(len(ActionType))
        self._action_space_size = len(ActionType)

        self._observation_space = Box(low=0, high=0, shape=(self.number_of_players, 1), dtype=np.float32)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if seed is not None:
            self.set_seed(seed)
        self.done = False
        obs = torch.as_tensor([0.0
                for i in range(self.number_of_players)])
        info = {"current_players": [0, 1]}
        return obs, info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action = action.detach().cpu().numpy()
        assert 0 <= action[0] <= len(ActionType) and 0 <= action[1] <= len(ActionType), \
            f"Action outside of valid range: [0,{len(ActionType)}]"
        assert not self.done, "Episode is over"
        # print(action[0], action[1])
        reward_vector = self.reward_table[action[0]][action[1]]
        obs = [self._observation_space.sample() for i in range(self.number_of_players)]
        self.done = True
        info = {"current_players": []}
        truncated = False
        cost = [0 for _ in range(self.number_of_players)]
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward_vector, cost, self.done, truncated)
        )
        return obs, reward, cost, terminated, truncated, info



    def render(self, mode='human', close=False):
        raise NotImplementedError('Rendering has not been coded yet')

    def close(self):
        pass

if __name__ == "__main__":
    env = RockPaperScissorsEnv(env_id='Rock-Paper-Scissors-v0')
    env.reset(seed=0)
    for i in range(20):
        action = [env.action_space.sample() for _ in range(2)]
        print(f'action:{ActionType(action[0]), ActionType(action[1])}')
        obs, reward, cost, terminated, truncated, info = env.step(torch.as_tensor(action))
        print('-' * 20)
        print(f'obs: {obs}')
        print(f'reward: {reward}')
        print(f'cost: {cost}')
        print(f'terminated: {terminated}')
        print(f'truncated: {truncated}')
        print('*' * 20)
        if terminated or truncated:
            env.reset(seed=0)
    env.close()
