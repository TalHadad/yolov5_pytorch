#!/usr/bin/env python

"""
Simulate the simplifie Mouse environment.
Each episode is trying to put the cat in the middle of the image.
"""

# Core Library
from enum import Enum
import random
import logging.config
import math
import random
from typing import Any, Dict, List, Tuple

# Third party
import gym
import numpy as np
from gym import spaces


class ActionType(Enum):
    stay = 'stay'

    forward = 'forward'
    forward_left = 'forward left'
    forward_right = 'forward right'

    backward = 'backward'
    backward_left = 'backward left'
    backward_right = 'backward right'

class MouseEnv(gym.Env):
    """
    Define a simple Mouse environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self) -> None:
        self.__version__ = "0.1.0"
        logging.info(f"MouseEnv - Version {self.__version__}")

        # Define what the agent can do
        # Move forward, forward-left, forward-right, backward, backward-left, backward-right.
        self.action_space = spaces.Discrete(4)

        # Observation is the cat location
        coordinate_low = np.array([-np.inf, -np.inf])
        coordinate_high = np.array([np.inf, np.inf])
        self.observation_space = spaces.Box(coordinate_low, coordinate_high, dtype=np.float32)

        # Store what the agent tried
        self.curr_episode = -1
        self.done = False

        # The observation is the cat location.
        # Cat location is in range of x=0-5 and y=0-5.
        # If cat isn't found, it's location is (0,0).
        self.location: List[int] = [random.randrange(11), random.randrange(11)] # the cat is somewere in the image.

        # switcher
        self.action_switcher = {ActionType.stay: (lambda l: l),
                    ActionType.forward: (lambda l: (l[0], l[1]+1)),
                    ActionType.forward_left: (lambda l: (l[0]-1, l[1]+1)),
                    ActionType.forward_right: (lambda l: (l[0]+1, l[1]+1)),
                    ActionType.backward: (lambda l: (l[0], l[1]-1)),
                    ActionType.backward_left: (lambda l: (l[0]-1, l[1]-1)),
                    ActionType.backward_right: (lambda l: (l[0]+1, l[1]-1))}

    def step(self, action: ActionType) -> Tuple[List[int], float, bool, Dict[Any, Any]]:
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob : List[int]
                an environment-specific object representing your observation of
                the environment.
            reward : float
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over : bool
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info : Dict
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        if (self._location_out_of_range()):
            # Cat is out of the frame. Episode is done.
            raise RuntimeError("Episode is done")

        self.curr_step += 1
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()
        return ob, reward, self.done, {}

    def _location_out_of_range(self) -> bool:
        # 0<=location<=10 is in the frame, otherwise out.
        return (self.location[0]<0 or self.location[1]<0 or self.location[0]>10 or self.location[1]>10)

    def _take_action(self, action: ActionType) -> None:
        self.location = self.action_switcher[action](self.location)

    def _get_reward(self) -> float:
        """Reward is given if location is in the middle (5,5)."""
        if self.location==(5,5):
            return 1.0
        else:
            return 0.0

    def reset(self) -> List[int]:
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation: List[int]
            The initial observation of the space.
        """
        self.done = False
        self.location: List[int] = [random.randrange(11), random.randrange(11)]
        self.curr_episode += 1
        return self._get_state()

    def _render(self, mode: str = "human", close: bool = False) -> None:
        print(f'current location: {self.location}')

    def _get_state(self) -> List[int]:
        """Get the observation."""
        ob = self.location
        return ob

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed