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
            x = self.x_prev + self.theta *
