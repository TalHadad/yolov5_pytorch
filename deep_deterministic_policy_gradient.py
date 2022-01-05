# Reinforcement learning in continuous action spaces
# (deep deterministic policy gradient - DDPG).
# We will need a:
# 1. class to encourage exploration (i.e. type of noise)
# (the policy is deterministic, it chooses some action with certainty, it it's fully deterministic it can't explore).
# 2. class to handle the replay memory, becase deep deterministic policy gradient works by combining the the magic of actor critic method with the magic of deep q-learning (which has a replay buffer)
# 3. class for our critic
# 4. class for our actor (as the agent)

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# 1. start with the noise
# OU stand for Ornstein Ullembeck
# that is a type of noise from physics that models the motion of a Brownian particle,
# meaning a particle subject to a random walk based on interactions with other nearby particles.
# It gives you a temporally correlated (in time set) type of noise that centers around a mean of zero.
class OUActionNoise(object):
      # the action noise will be used in actor class to add in some exploration noise to the action selection

      # dt, as in the differential with respect to time
      def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
            self.theta = theta
            self.mu = mu
            self.sigma = sigma
            self.dt = dt
            self.x0 = x0
            self.reset() # will reset the temporal correlation, you may want to do that from time to time, not necessary for our particular implementaion

      # noise = OUActionNoise()
      # noise() # allows you to call noise() instead of noise.get_noise()
      def __call__(self):
            x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
            self.x_prev = x
            return x

      def reset(self):
            '''
            check to make sure x0 exists,
            if it doesn't it sets it to some 0 value
            '''
            self.x_prev = self.x0 if self.x0 is None else np.zeros_like(self.mu)

# check out videos on deep q-learning (will make it more clear)
# (and videos on actor critic methods, because DDPG combines actor critic with deep q-learning)
class ReplayBuffer(object):
      '''
      straightforward, just set of numpy arrays in the shape of the action space, observation space, and rewards.
      That way we have a memory of event that happened, so we can sapmle them during the learning step
      '''
     def __init__(self, max_size, input_shape, n_actions):
           self.mem_size = max_size
           self.mem_cntr = 0

           # * idiom isn't a pointer, it is to unpack a tuple
           # makes our class extensible, pass list of a single element as in the case of continuous environment
           self.state_memory = np.zeros((self.mem_size, *input_shape))
           self.new_state_memory = np.zeros((self.mem_size, *input_shape))

           self.action_memory = np.zeros((self.mem_size, n_actions))

           self.reward_memory = np.zeros(self.mem_size)

           # we track when we transition into terminal states,
           # save the done flages from the open ai gym environment
           self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

      def store_transition(self, state, action, reward, state_, done): # state_ is new state, done is done flag
            index = self.mem_cntr % self.mem_size
            self.state_memory[index] = state
            self.action_memory[index] = action
            self.reward_memory[index] = reward
            self.new_state_memory[index] = state_

            # when we get to the update/bellman equation for our learning funcion,
            # you'll see we want to multiply by whether or not the episode is over,
            # and that gets facilitated by 1 - the quantity done
            self.terminal_memory[index] = 1 - done

            self.mem_cntr += 1

      def sample_buffer(self, batch_size):
            max_mem = min(self.mem_cntr, self.mem_size)
            batch = np.random.choice(max_mem, batch_size)

            # we want to get ahold of the respective states, actions, rewards, new states and terminal flages
            # and pass them back to the learning function
