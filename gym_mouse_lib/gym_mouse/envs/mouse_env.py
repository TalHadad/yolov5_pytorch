#!/usr/bin/env python

"""
Simulate the simplifie Mouse environment.
Each episode is trying to put the cat in the middle of the image.
"""

# Core Library
import logging
from utils.logging_level import LOGGING_LEVEL
logging.basicConfig(level=LOGGING_LEVEL)
log = logging.getLogger('mouse_env')
log.setLevel(LOGGING_LEVEL)
import random
from typing import Any, Dict, List, Tuple

# Third party
import gym
import numpy as np
from gym import spaces

MAX_SINGLE_GAME_ITERATION = 1000
class MouseEnv(gym.Env):
    """
    Define a simple Mouse environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self) -> None:
        self.__version__ = "0.1.0"
        log.info(f"MouseEnv - Version {self.__version__}")
        self.action_space = spaces.Discrete(7)
        coordinate_low = np.array([-1, -1])
        coordinate_high = np.array([11, 11])
        self.observation_space = spaces.Box(coordinate_low, coordinate_high, dtype=np.float32)
        self.action_switcher = {0: (lambda l: l),
                                1: (lambda l: [l[0], l[1] + 1]),
                                2: (lambda l: [l[0] - 1, l[1] + 1]),
                                3: (lambda l: [l[0] + 1, l[1] + 1]),
                                4: (lambda l: [l[0], l[1] - 1]),
                                5: (lambda l: [l[0] - 1, l[1] - 1]),
                                6: (lambda l: [l[0] + 1, l[1] - 1])}

        self.location: List[int] = [random.randrange(1, 11), random.randrange(1, 11)]  # the cat is somewere in the image.
        self.game_step_counter = 0

    def controller_do_action(self, action: int):
        if action == -1: # last game is over
            self.game_step_counter = 0
            self.location: List[int] = [random.randrange(1, 11), random.randrange(1, 11)]  # the cat is somewere in the image.
            log.info(f'started at new location {self.location}')
        else:
            self.location = self.action_switcher[action](self.location)
            if self.is_done(self.location):
                log.info(f'location is out of bound, {self.location}, game over!')
                self.location = [0, 0]

    def is_done(self, state) -> bool:
        self.game_step_counter += 1
        return (state[0] == 0 and state[1] == 0) \
               or state[0] < 0 or state[1] < 0 \
               or state[0] > 10 or state[1] > 10 \
               or self.game_step_counter > MAX_SINGLE_GAME_ITERATION

    def detector_render(self, mode: str = "human", close: bool = False) -> None:
        print(f'current location: {self.location}')

    def detector_get_location(self) -> List[int]:
        return self.location

    def seed(self, seed: Any) -> None:
        random.seed(seed)
        np.random.seed

    def reset(self):
        pass

class MouseEnv_old(gym.Env):
    """
    Define a simple Mouse environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self) -> None:
        self.__version__ = "0.1.0"
        log.info(f"MouseEnv - Version {self.__version__}")

        # Define what the agent can do
        # Move forward, forward-left, forward-right, backward, backward-left, backward-right.
        self.action_space = spaces.Discrete(7)

        # Observation is the cat location
        # coordinate_low = np.array([-np.inf, -np.inf])
        # coordinate_high = np.array([np.inf, np.inf])
        coordinate_low = np.array([-1, -1])
        coordinate_high = np.array([11, 11])
        self.observation_space = spaces.Box(coordinate_low, coordinate_high, dtype=np.float32)

        # Store what the agent tried
        self.curr_episode = -1
        self.done = False

        # The observation is the cat location.
        # Cat location is in range of x=0-5 and y=0-5.
        # If cat isn't found, it's location is (0,0).
        self.location: List[int] = [random.randrange(11), random.randrange(11)]  # the cat is somewere in the image.

        # switcher
        # self.action_switcher = {ActionType.stay: (lambda l: l),
        #            ActionType.forward: (lambda l: (l[0], l[1]+1)),
        #            ActionType.forward_left: (lambda l: (l[0]-1, l[1]+1)),
        #            ActionType.forward_right: (lambda l: (l[0]+1, l[1]+1)),
        #            ActionType.backward: (lambda l: (l[0], l[1]-1)),
        #            ActionType.backward_left: (lambda l: (l[0]-1, l[1]-1)),
        #            ActionType.backward_right: (lambda l: (l[0]+1, l[1]-1))}
        self.action_switcher = {0: (lambda l: [l[0], l[1] + 1]),
                                1: (lambda l: [l[0] - 1, l[1] + 1]),
                                2: (lambda l: [l[0] + 1, l[1] + 1]),
                                3: (lambda l: [l[0], l[1] - 1]),
                                4: (lambda l: [l[0] - 1, l[1] - 1]),
                                5: (lambda l: [l[0] + 1, l[1] - 1]),
                                6: (lambda l: l)}
        self.iteration_counter = 0

    # not used, using controller_do_action to return only state
    def step(self, action: int) -> Tuple[List[int], float, bool, Dict[Any, Any]]:
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
        if self.is_done(self.get_state()):
            # Cat is out of the frame. Episode is done.
            # raise RuntimeError("Episode is done")
            # restart a game
            log.info(f'{self.__class__.__name__} game over, location {self.location}, done {self.done}')
            self.reset()
            log.info(f'{self.__class__.__name__} restart, location {self.location}, done {self.done}')

        self._take_action(action)
        reward = self._get_reward()
        state = self.get_state()
        return state, reward, self.done, {}

    def controller_do_action(self, action: int) -> List[int]:
        if self.is_done(self.get_state()):
            # Cat is out of the frame. Episode is done.
            # raise RuntimeError("Episode is done")
            # restart a game
            log.info(f'{self.__class__.__name__} game over, location {self.location}, done {self.done}')
            self.reset()
            log.info(f'{self.__class__.__name__} restart, location {self.location}, done {self.done}')

        self.controller_take_action(action)
        state = self.get_state()
        return state

    def is_done(self, state) -> bool:
        # 0<location<=10 is in the frame, otherwise out.
        self.iteration_counter += 1
        return state is None or \
               state[0] <= 0 or \
               state[1] <= 0 or \
               state[0] > 10 or \
               state[1] > 10 or \
               self.iteration_counter > MAX_SINGLE_GAME_ITERATION

    def controller_take_action(self, action: int) -> None:
        # print(f'action = {type(action)} {action}') # action = <class 'numpy.ndarray'> [-0.00392292  0.03153225  0.00023934  0.02393808 -0.0070587   0.0526591 0.04278992]
        self.location = self.action_switcher[action](self.location)
        if self.is_done(self.get_state()):
            self.done = True
            # TODO not make location None as end state, cause non vector to return as action
            # self.location = None
            self.location = [0,0]

    # not used, the new method is moved to agent env
    def _get_reward(self) -> float:
        """Reward is given if location is in the middle (5,5)."""
        if self.location == [5, 5]:
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
        self.location: List[int] = [random.randrange(1, 11), random.randrange(1, 11)]
        self.curr_episode += 1
        return self.get_state()

    def detector_render(self, mode: str = "human", close: bool = False) -> None:
        print(f'current location: {self.location}')

    def get_state(self) -> List[int]:
        """Get the observation."""
        state = self.location
        return state

    def detector_get_location(self) -> List[int]:
        return self.get_state()

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed
