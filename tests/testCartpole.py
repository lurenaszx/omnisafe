# 导入必要的包
from __future__ import annotations

from typing import Any, ClassVar
import gymnasium
import torch
import numpy as np
import omnisafe

from omnisafe.envs.core import CMDP, env_register, env_unregister
from omnisafe.typing import DEVICE_CPU

custom_cfgs = {
    'train_cfgs': {
        'total_steps': 200000,
    },
    'algo_cfgs': {
        'steps_per_epoch': 200,
        'update_iters': 5,
    },
}


@env_register
@env_unregister  # 避免重复运行单元格时产生"环境已注册"报错
class ExampleMuJoCoEnv(CMDP):
    _support_envs: ClassVar[list[str]] = ['CartPole-v0']  # 支持的任务名称

    need_auto_reset_wrapper = True  # 是否需要 `AutoReset` Wrapper
    need_time_limit_wrapper = True  # 是否需要 `TimeLimit` Wrapper

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_id)
        self._num_envs = num_envs
        self._env = gymnasium.make(id=env_id, autoreset=True, **kwargs)  # 实例化环境对象
        self._action_space = self._env.action_space  # 指定动作空间，以供算法层初始化读取
        self._observation_space = self._env.observation_space  # 指定观测空间，以供算法层初始化读取
        self._device = device  # 可选项，使用GPU加速。默认为CPU

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)  # 重置环境
        return (
            torch.as_tensor(obs, dtype=torch.float32, device=self._device),
            info,
        )  # 将重置后的观测转换为torch tensor。

    @property
    def max_episode_steps(self) -> int | None:
        return self._env.env.spec.max_episode_steps  # 返回环境每一幕的最大交互步数

    def set_seed(self, seed: int) -> None:
        self.reset(seed=seed)  # 设定环境的随机种子以实现可复现性

    def render(self) -> Any:
        return self._env.render()  # 返回环境渲染的图像

    def close(self) -> None:
        self._env.close()  # 训练结束后，释放环境实例

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        obs, reward, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )  # 读取与环境交互后的动态信息
        cost = np.zeros_like(reward)  # Gymnasium并显式包含安全约束，此处仅为占位。
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )  # 将动态信息转换为torch tensor。
        if 'final_observation' in info:
            info['final_observation'] = np.array(
                [
                    array if array is not None else np.zeros(obs.shape[-1])
                    for array in info['final_observation']
                ],
            )
            info['final_observation'] = torch.as_tensor(
                info['final_observation'],
                dtype=torch.float32,
                device=self._device,
            )  # 将info中记录的上一幕final observation转换为torch tensor。

        return obs, reward, cost, terminated, truncated, info


agent = omnisafe.Agent('RMPG', 'CartPole-v0', custom_cfgs=custom_cfgs)
agent.learn()

agent.plot(smooth=1)
# agent.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
