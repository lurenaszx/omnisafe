from copy import copy
from itertools import dropwhile
from typing import List
import enum
import numpy as np
# import gym
from gymnasium.spaces import Discrete, Tuple
import gymnasium
from omnisafe.envs.core import CMDP, env_register
from typing import Any, ClassVar
import torch
from omnisafe.typing import DEVICE_CPU, Box
import random
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.models.actor.discrete_actor import DiscreteActor

class ActionType(enum.Enum):
    PASS = 0
    BET = 1


class OneHotEncoding(gymnasium.Space):
    """
    {0,...,1,...,0}

    Example usage:
    self.observation_space = OneHotEncoding(size=4)
    """

    def __init__(self, size=None):
        assert isinstance(size, int) and size > 0
        self.size = size
        gymnasium.Space.__init__(self, (), np.int64)

    def sample(self):
        one_hot_vector = np.zeros(self.size)
        one_hot_vector[np.random.randint(self.size)] = 1
        return one_hot_vector

    def contains(self, x):
        if isinstance(x, (list, tuple, np.ndarray)):
            number_of_zeros = list(x).contains(0)
            number_of_ones = list(x).contains(1)
            return (number_of_zeros == (self.size - 1)) and (number_of_ones == 1)
        else:
            return False

    def __repr__(self):
        return "OneHotEncoding(%d)" % self.size

    def __eq__(self, other):
        return self.size == other.size

@env_register
class KuhnPokerEnv(CMDP):
    '''
    Implementation of Kuhn's poker in accordance to OpenAI gym environment interface.
    '''

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False

    _support_envs: ClassVar[list[str]] = [
        'KuhnPoker-v0',
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
        :param deck_size: Size of the deck from which cards will be drawn, one for each player (Default 3).
        :param betting_rounds: Number of times that (Default: 2).
        :param ante: Amount of utility that all players must pay at the beginning of an episode (Default 1).
        '''
        self._num_envs = num_envs
        self.done = False
        self.number_of_players = kwargs.get("num_players", 2)
        self.deck_size = kwargs.get("deck_size", self.number_of_players+1)
        self.betting_rounds = kwargs.get("betting_rounds", 2)
        self.ante = kwargs.get("ante", 1)

        assert self.number_of_players >= 2, "Game must be played with at least 2 players"
        assert self.deck_size >= self.number_of_players, "The deck of cards must contain at least one card per player"
        assert self.betting_rounds >= 2, "The deck of cards must contain at least one card per player"
        assert self.ante >= 1, "Minimun ante must be one"
        self._device = device
        self._action_space = Discrete(len(ActionType))
        self._action_space_size = len(ActionType)

        obs_len = self.calculate_observation_space()

        self._observation_space = Box(low=0, high=1, shape=[self.number_of_players, obs_len], dtype=np.float32)
        self.state_space_size = None  # TODO len(self.random_initial_state_vector())

        self.betting_history_index = self.number_of_players + self.number_of_players * self.deck_size
        self.betting_round_offset = self.number_of_players * len(ActionType)
        self.env_spec_log = {'Env/nash_conv': 0.0}
        self.log_count = 0

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        card: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if seed is not None:
            self.set_seed(seed)
        self.player_pot = [self.ante for _ in range(self.number_of_players)]
        self.done = False
        self.current_player = 0
        self.history: List[ActionType] = []
        self.state = self.random_initial_state_vector(card=card)
        self.first_to_bet, self.winner = None, None
        # Players that will face off in card comparison after betting ends
        self.elegible_players = [i for i in range(self.number_of_players)]
        obs = torch.as_tensor([self.observation_from_state(player_id=i)
                for i in range(self.number_of_players)], dtype=torch.float32)
        info = {'current_players': [self.current_player]}
        return obs, info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action = action.detach().cpu().numpy()
        assert 0 <= action <= len(ActionType), \
            f"Action outside of valid range: [0,{len(ActionType)}]"
        assert not self.done, "Episode is over"
        move = ActionType(action)

        self.state = self.update_state(self.current_player, move)

        if self.done:
            reward_vector = self.reward_vector_for_winner(self.winner)
        else:
            reward_vector = [0] * self.number_of_players
        obs = [self.observation_from_state(i) for i in range(self.number_of_players)]
        info = {"current_players": [self.current_player]}
        truncated = False
        cost = [0 for i in range(self.number_of_players)]
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward_vector, cost, self.done, truncated)
        )
        return obs, reward, cost, terminated, truncated, info

    def calculate_reward(self, state):
        if False:  # If is terminal
            pass
        return [0] * self.number_of_players

    def update_state(self, player_id, move):
        # print(self.state)
        self.history.append(move)

        if self.first_to_bet is None:
            player_move_index = self.betting_history_index
        else:
            player_move_index = self.betting_history_index + self.betting_round_offset
        player_move_index += self.current_player * len(ActionType)
        if move == ActionType.PASS:
            self.state[player_move_index] = 1
        if move == ActionType.BET:
            self.state[player_move_index + 1] = 1
            if self.first_to_bet is None:
                self.first_to_bet = player_id
                self.elegible_players = []
            self.elegible_players += [player_id]
            self.player_pot[player_id] += 1

        # Update current player
        self.state[self.current_player] = 0
        self.current_player = (self.current_player + 1) % self.number_of_players
        self.state[self.current_player] = 1

        if self.is_state_terminal():
            self.done = True
            self.winner = self.get_winner()

        return self.state

    def get_winner(self):
        base_i = self.number_of_players
        player_hand = lambda p_i: slice(base_i + p_i * self.deck_size,
                                        base_i + p_i * self.deck_size + self.deck_size)
        player_hands = [self.state[player_hand(i)] for i in self.elegible_players]
        card_values = [hand.index(1) for hand in player_hands]
        max_card = max(card_values)
        winner = self.elegible_players[card_values.index(max_card)]
        return winner

    def get_total_pot(self):
        return sum(self.player_pot)

    def get_player_pot(self, player_id):
        assert 0 <= player_id < self.number_of_players, \
            f"Player id must be in [0, {self.number_of_players}]"
        return self.player_pot[player_id]

    def reward_vector_for_winner(self, winner: int):
        assert 0 <= winner < self.number_of_players, \
            f"Player id must be in [0, {self.number_of_players}]"
        reward_vector = [- self.get_player_pot(i)
                         for i in range(self.number_of_players)]
        reward_vector[winner] += self.get_total_pot()
        return reward_vector

    def is_state_terminal(self):
        return self.all_players_passed() or self.betting_is_over()

    def all_players_passed(self):
        all_players_acted = len(self.history) == self.number_of_players
        return all_players_acted and all(map(lambda m: m == ActionType.PASS,
                                             self.history))

    def betting_is_over(self):
        after_bet_moves = dropwhile(lambda m: m != ActionType.BET,
                                    self.history)
        return len(list(after_bet_moves)) == self.number_of_players

    def random_initial_state_vector(self, card = None):
        # Player 1 always begins
        player_turn = [1] + [0] * (self.number_of_players - 1)
        # Deal 1 card to each player
        dealt_cards_per_player = np.random.choice(range(self.deck_size),
                                                  size=self.number_of_players,
                                                  replace=False).tolist()
        if card is not None:
            dealt_cards_per_player = card
        player_hands = self.vector_from_dealt_hands(dealt_cards_per_player)

        if card is not None:
            dealt_cards_per_player = card

        betting_history_vector = []
        for _ in range(self.betting_rounds):
            for _ in range(self.number_of_players):
                betting_history_vector += [0] * len(ActionType)

        return player_turn + player_hands + \
            betting_history_vector

    def vector_from_dealt_hands(self, dealt_cards_per_player):
        player_hands = [[0] * self.deck_size
                        for _ in range(self.number_of_players)]
        player_hands = []
        for dealt_card in dealt_cards_per_player:
            hand = [0] * self.deck_size
            hand[dealt_card] = 1
            player_hands += hand
        return player_hands

    def observation_from_state(self, player_id: int):
        '''
        Returns the observation vector for :param player_id:. It is strictly
        a sub-vector of the real state.

        Contains:
            - Player id
            - Card randomly dealt to :param player_id:
            - Betting history
            - Amount of utility on pot by each player

        :param player_id: Index from [0, self.number_of_players] of the player
                          whose observation vector is being generated
        :returns: Observation vector for :param player_id:
        '''
        encoded_id = [0] * self.number_of_players
        encoded_id[player_id] = 1

        dealt_cards_start_index = self.number_of_players
        player_card_index = dealt_cards_start_index + player_id*self.deck_size
        player_card = self.state[player_card_index: player_card_index + self.deck_size]

        betting_history_start_index = dealt_cards_start_index + (self.number_of_players * self.deck_size)
        betting_history_and_pot = self.state[betting_history_start_index:]

        return encoded_id + player_card + betting_history_and_pot

    def calculate_observation_space(self):
        single_len = self.number_of_players+self.deck_size+(len(ActionType))*self.number_of_players\
                     *self.betting_rounds
        return single_len

    def render(self, mode='human', close=False):
        raise NotImplementedError('Rendering has not been coded yet')

    def close(self):
        pass

    def spec_log(self, logger, policy: list[ConstraintActorQCritic | DiscreteActor]) -> dict[str, Any]:
        if self.log_count % 20 == 0:
            if isinstance(policy[0], DiscreteActor):
                model = policy
            else:
                model = [ac.actor for ac in policy]
            response_policy = [BestResponsePolicy(number_of_players=self.number_of_players,
                                           player_id=i,
                                           policy=model,
                                           root_state=root_state,
                                           state_transition=state_transition,
                                           infosets=obs_set) for i in range(self.number_of_players)]
            player_conv = [response.get_nash_conv() for response in response_policy]
            nash_conv = sum(player_conv)
            self.env_spec_log = {'Env/nash_conv': nash_conv}
            print(player_conv)
        logger.store({'Env/nash_conv': self.env_spec_log['Env/nash_conv']})
        self.log_count += 1

# To record the information of the environment
env = KuhnPokerEnv(env_id="", num_envs=1)
root_state = []
state_transition = {}
obs_set = {}

def record(bet, history):
    obs, info = env.reset(card=bet)
    for per_step in history:
        obs, rewards, _, terminal, truncated, info = env.step(torch.tensor(per_step))
    if len(history) == 0:
        root_state.append(tuple(env.state))
    for per_obs in obs:
        if tuple(per_obs.tolist()) in obs_set.keys():
            obs_set[tuple(per_obs.tolist())] += [tuple(env.state)]
        else:
            obs_set[tuple(per_obs.tolist())] = [tuple(env.state)]
    state_record = {}
    if len(history) == 0:
        state_record['obs'] = obs
        state_record['current_players'] = info['current_players']
        state_record['done'] = 0
    else:
        state_record['obs'], state_record['rew'], state_record['done'], state_record['current_players'] = obs, \
            rewards, terminal, info["current_players"]
    state_transition[tuple(env.state)] = state_record
    res = tuple(env.state)
    if len(history) == 0 or (not state_record['done']):
        state_record[0] = record(bet, [x for x in history] + [0])
        state_record[1] = record(bet, [x for x in history] + [1])
    return res


for i in range(3):
    for j in range(3):
        if i != j:
            record([i, j], [])

# print(state_transition, len(obs_set), len(root_state))
class BestResponsePolicy:
    """Computes the best response to a specified strategy."""

    def __init__(self,
                 number_of_players,
                 player_id,
                 policy,
                 root_state=None,
                 cut_threshold=0.0,
                 state_transition=None,
                 infosets=None
                 ):
        """Initializes the best-response calculation.

        Args:
          game: The game to analyze.
          player_id: The player id of the best-responder.
          root_state: The state of the game at which to start analysis. If `None`,
            the game root state is used.
          cut_threshold: The probability to cut when calculating the value.
            Increasing this value will trade off accuracy for speed.
        """
        self._num_players = number_of_players
        self._player_id = player_id
        self._policy = policy
        self._root_state = root_state
        self.v_values = {}
        self.ct_v_values = {}
        self.state_transition = state_transition
        self.infosets = infosets
        self.probs_cal = {}
        self._cut_threshold = cut_threshold
        self.actions = [0, 1]
        [self.dfs(state, 1) for state in self._root_state]

    def get_nash_conv(self):
        sum = 0
        for state in self._root_state:
            sum += self.get_value(state) - self.get_current_value(state)
        return sum/len(self._root_state)

    def dfs(self, state, prob):
        self.probs_cal[state] = prob
        current_players = self.state_transition[tuple(state)]['current_players']
        # print(self.state_transition[state])
        if self.state_transition[state]['done']:
            return
        if self._player_id in current_players:
            self.dfs(state_transition[state][0], prob)
            self.dfs(state_transition[state][1], prob)
        else:
            transtion = self.transitions(state)
            [self.dfs(state_transition[state][act], prob*pb) for act, pb in transtion]

    def transitions(self, state, is_best=True):
        """Returns a list of (action, cf_prob) pairs from the specified state."""
        current_players = self.state_transition[state]['current_players']
        if self._player_id in current_players and is_best:
            # Counterfactual reach probabilities exclude the best-responder's actions,
            # hence return probability 1.0 for every action.
            return [(action, 1.0) for action in self.actions]
        else:
            obs = self.state_transition[state]['obs'][current_players[0]].unsqueeze(0)
            probs = self._policy[current_players[0]](obs).probs.detach().tolist()[0]
            # print(probs)
            return [(action, prob) for action, prob in zip(self.actions, probs)]

    def get_current_value(self, state):
        """Returns the value of the specified state to the current player."""
        if state in self.ct_v_values.keys():
            return self.ct_v_values[tuple(state)]
        if self.state_transition[state]['done']:
            self.ct_v_values[state] = self.state_transition[state]['rew'][self._player_id]
            return self.ct_v_values[state]
        else:
            self.ct_v_values[state] = sum(p * self.get_current_q_value(state, a)
                                          for a, p in self.transitions(state, is_best=False)
                                          if p > self._cut_threshold)
            return self.ct_v_values[state]

    def get_current_q_value(self, state, action):
        """Returns the value of the (state, action) to the best-responder."""
        current_players = self.state_transition[state]['current_players']
        return self.get_current_value(self.state_transition[state][action])

    def get_value(self, state):
        """Returns the value of the specified state to the best-responder."""
        current_players = self.state_transition[state]['current_players']
        if state in self.v_values.keys():
            return self.v_values[state]
        if self.state_transition[state]['done']:
            self.v_values[state] = self.state_transition[state]['rew'][self._player_id]
            return self.v_values[state]
        elif (self._player_id in self.state_transition[state]['current_players']):
            action = self.best_response_action(
                tuple(self.state_transition[state]['obs'][self._player_id].tolist()))
            self.v_values[state] = self.get_q_value(state, action)
            return self.v_values[state]
        else:
            return sum(p * self.get_q_value(state, a)
                       for a, p in self.transitions(state)
                       if p > self._cut_threshold)

    def get_q_value(self, state, action):
        """Returns the value of the (state, action) to the best-responder."""
        current_players = self.state_transition[state]['current_players']
        return self.get_value(self.state_transition[state][action])

    def best_response_action(self, infostate):
        """Returns the best response for this information state."""
        infoset = self.infosets[infostate]
        # print(infostate, infoset)
        # Get actions from the first (state, cf_prob) pair in the infoset list.
        # Return the best action by counterfactual-reach-weighted state-value.
        return max(
            self.actions,
            key=lambda a: sum(self.probs_cal[s] * self.get_q_value(s, a) for s in infoset))


if __name__ == "__main__":
    env = KuhnPokerEnv(env_id='Example-v0')
    env.reset(seed=0)
    for _ in range(1000):
        action = env.action_space.sample()
        print(f'action:{action}')
        obs, reward, cost, terminated, truncated, info = env.step(torch.as_tensor(action))
        print('-' * 20)
        print(f'obs: {obs}')
        print(f'reward: {reward}')
        print(f'cost: {cost}')
        print(f'terminated: {terminated}')
        print(f'truncated: {truncated}')
        print('*' * 20)
        if terminated or truncated:
            env.reset(seed = 0)
    env.close()
