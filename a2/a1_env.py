import pickle
import pygame
import numpy as np
from tqdm import tqdm
from typing import List, Any

from gymnasium import Env
from gymnasium import spaces
from gymnasium.utils import env_checker

# Display set up
import os
import time
import math
import hashlib
import matplotlib.pyplot as plt
from itertools import combinations
from IPython.display import clear_output


os.environ["SDL_VIDEODRIVER"] = "dummy"


class StocasticGridEnvironment(Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
        "background_img": "./assets/background.jpg",
        "goal_img": "./assets/goal.png",
        "agent_img": "./assets/agent.png",
        "reward_img": "./assets/reward.png",
        "neg_reward_img": "./assets/neg_reward.png" 
    }

    def __init__(
        self,
        size: int = 4,
        max_time_steps:int = 20,
        render_mode=None,
        seed=88
    ):
        super().reset(seed=seed)
        self.size = size
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2, ), seed=self.np_random, dtype=int),
                "goal": spaces.Box(0, size - 1, shape=(2, ), seed=self.np_random, dtype=int),
                "reward": spaces.Box(
                    -1., 1.,
                    shape=(size**2, ),
                    seed=self.np_random,
                    dtype=np.float16
                )
            }
        )

        self.action_space = spaces.Discrete(4)
        self.action_mode_space = spaces.Discrete(3)
        self.max_timesteps = max_time_steps

        self._base_states = np.zeros((size, size), dtype=np.float16)

        # Add 6 negative rewards
        for _ in range(6):
            neg_reward_loc = self.observation_space['agent'].sample()
            while self._base_states[tuple(neg_reward_loc)] != 0:
                neg_reward_loc = self.observation_space['agent'].sample()

            self._base_states[tuple(neg_reward_loc)] = self.observation_space['reward'].low[0]

        # Add 3 positive rewards
        for _ in range(3):
            pos_reward_loc = self.observation_space['agent'].sample()
            while self._base_states[tuple(pos_reward_loc)] != 0:
                pos_reward_loc = self.observation_space['agent'].sample()

            self._base_states[tuple(pos_reward_loc)] = \
                .99 * self.observation_space['reward'].high[0]

        """The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: {'move': np.array([1, 0]), 'name': 'down'},
            1: {'move': np.array([-1, 0]), 'name': 'up'},
            2: {'move': np.array([0, 1]), 'name': 'right'},
            3: {'move': np.array([0, -1]), 'name': 'left'},
        }
        self._action_mode_mapping = {
            0: {'p': 2/3, 'name': 'as-is', 'score_mul': 1},
            1: {'p': 2/9, 'name': 'double', 'score_mul': 2},
            2: {'p': 1/9, 'name': 'mirror', 'score_mul': 3}
        }

        self.timestep = 0
        self.states = self._base_states.copy()
        # We will sample the goals's location randomly until it does not coincide
        # with the reward locations.
        self._goal_pos = self.observation_space['goal'].sample()
        while self.states[tuple(self._goal_pos)] != 0:
            self._goal_pos = self.observation_space['goal'].sample()

        self.states[tuple(self._goal_pos)] = self.observation_space['reward'].high[0]

        # We will sample the agent's location randomly until it does not coincide
        # with the goal and reward locations.
        self._agent_pos = self.observation_space['agent'].sample()
        while self.states[tuple(self._agent_pos)] != 0:
            self._agent_pos = self.observation_space['agent'].sample()

        # Check for render mode legality
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.window_size = 744  # The size of the PyGame window

    def reset(self, seed=None, options=None):
        # For compliance
        super().reset(seed=seed)

        # Reset env
        self.timestep = 0
        self.states = self._base_states.copy()
        self.states[tuple(self._goal_pos)] = self.observation_space['reward'].high[0]

        # Agent start pos changes everytime
        self._agent_pos = self.observation_space['agent'].sample()
        while self.states[tuple(self._agent_pos)] != 0:
            self._agent_pos = self.observation_space['agent'].sample()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Randomly select the action mode
        action_mode = self.np_random.choice(
            range(self.action_mode_space.n),
            p=[self._action_mode_mapping[i]['p']
               for i in range(self.action_mode_space.n)]
        )

        if self._action_mode_mapping[action_mode]['name'] in ('double', 'as-is'):
            direction = self._action_to_direction[action]['move'].copy()
            if self._action_mode_mapping[action_mode]['name'] == 'double':
                direction *= 2

            # Restricting the agent to the grid.
            new_pos = np.clip(
                self._agent_pos + direction,
                0, self.size - 1
            )

        else:
            new_pos = self._agent_pos[::-1]

        # Penalize reaching the goal before collecting all rewards
        block_exit = (self.states > 0).sum() > 1

        terminated = False
        if np.array_equal(new_pos, self._goal_pos):
            if block_exit:
                new_pos = self._agent_pos
            else:
                terminated = True

        # Calculate base bonus
        base_bonus = (self.states[tuple(new_pos)]>0) * \
            .1 * self.observation_space['reward'].high[0]

        # Give negative reward for actions that keep you in the same
        # place (boundary actions)
        if np.array_equal(self._agent_pos, new_pos):
            reward = self.observation_space['reward'].low[0]
        else:
            reward = self.states[tuple(new_pos)] + base_bonus * \
                self._action_mode_mapping[action_mode]['score_mul']

        # In all cases add a reward penalty of 0.1 * max_negative_reward
        # in other words, every action is penalized.
        reward += .1 * self.observation_space['reward'].low[0]

        self._agent_pos = new_pos

        # Record the action taken
        observation = self._get_obs()

        # Consume reward
        self.states[tuple(self._agent_pos)] = 0
        
        # An episode is done iff the agent has reached the goal
        self.timestep += 1

        truncated = self.timestep > self.max_timesteps

        info = self._get_info()
        info.update({
            'action': self._action_to_direction[action]['name'],
            'action_mode': self._action_mode_mapping[action_mode]['name']
        })

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        return {
            "agent": self._agent_pos,
            "goal": self._goal_pos,
            "reward": self.states.flatten()
        }

    # City block distance between goal and agent
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_pos - self._goal_pos, ord=1
            )
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Background image
        bg_img = pygame.image.load(self.metadata['background_img'])
        bg_img.set_alpha(188)
        bg_img = pygame.transform.scale(bg_img, (self.window_size, self.window_size))
        canvas.blit(
            bg_img,
            [0, 0]
        )

        # Goal image
        goal_img = pygame.image.load(self.metadata['goal_img'])
        goal_img = pygame.transform.scale(goal_img, (pix_square_size, pix_square_size))
        canvas.blit(
            goal_img,
            pix_square_size * self._goal_pos[::-1]
        )

        # Agent image
        agent_img = pygame.image.load(self.metadata['agent_img'])
        agent_img = pygame.transform.scale(agent_img, (pix_square_size, pix_square_size))
        canvas.blit(
            agent_img,
            pix_square_size * self._agent_pos[::-1]
        )

        # Reward image
        reward_img = pygame.image.load(self.metadata['reward_img'])
        # Negative reward image
        neg_reward_img = pygame.image.load(self.metadata['neg_reward_img'])

        # Add the reward and neg reward
        for x in range(self.size):
            for y in range(self.size):
                reward = self.states[x, y]
                if self.states[x, y] > 0 and self.states[x, y] < \
                    self.observation_space['reward'].high[0]:
                    sreward_img = pygame.transform.scale(
                        reward_img,
                        (pix_square_size*reward, pix_square_size*reward)
                    )
                    rew_sz = np.array(sreward_img.get_size()) # w x h
                    position = pix_square_size * (np.array([y, x])+0.5)
                    # To center to grid square
                    position -= rew_sz[::-1] / 2
                    canvas.blit(
                        sreward_img,
                        position
                    )
                elif self.states[x, y] < 0:
                    reward *= -1
                    sneg_reward_img = pygame.transform.scale(
                        neg_reward_img,
                        (pix_square_size*reward, pix_square_size*reward)
                    )
                    nrew_sz = np.array(sneg_reward_img.get_size()) # w x h
                    position = pix_square_size * (np.array([y, x])+0.5)
                    # To center to grid square
                    position -= nrew_sz[::-1] / 2
                    canvas.blit(
                        sneg_reward_img,
                        position
                    )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (255, 255, 255),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=2,
            )
            pygame.draw.line(
                canvas,
                (255, 255, 255),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=2,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
