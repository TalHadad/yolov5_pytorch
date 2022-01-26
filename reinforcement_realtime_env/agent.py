# deep_deterministic_policy_gradient.py
# NOTE: $ mkdir tmp && makdir tmp/ddpg $$ mkdir plots
# NOTE: $ source ~/virtualenv_environments/py38_yolov5_pytorch/bin/activate
# NOTE: $ pip3 install gym Box2D pyglet
# activate with:
# NOTE: $ source ~/virtualenv_environments/py38_yolov5_pytorch/bin/activate
# NOTE: $ cd ~/Desktop/public_projects/yolov5_pytorch/
# NOTE: $ conda deactivate
# NOTE: $ python3 main_ddpg.py
# NOTE: $ python3 torch_lunar_lander.py

# Reinforcement learning in continuous action spaces
# (deep deterministic policy gradient - DDPG).
# We will need a:
# 1. class to encourage exploration (i.e. type of noise)
# (the policy is deterministic, it chooses some action with certainty, it it's fully deterministic it can't explore).
# 2. class to handle the replay memory, becase deep deterministic policy gradient works by combining the the magic of actor critic method with the magic of deep q-learning (which has a replay buffer)
# 3. class for our critic
# 4. class for our actor (as the agent)

from abc import ABC, abstractmethod
import random
import logging
logging.basicConfig(level=logging.INFO)
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import datetime

class Agent(ABC):
      @abstractmethod
      def choose_action(self, state):
            pass

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
            self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

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
            self.new_state_memory = np.zeros((self.mem_size, *input_shape)) # same shape as state_memeory
            self.action_memory = np.zeros((self.mem_size, n_actions))
            self.reward_memory = np.zeros(self.mem_size)

            # we track when we transition into terminal states,
            # save the done flages from the open ai gym environment
            self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32) # float32 becasue tesorflow is perticular with data types, be aware of that.

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
            states = self.state_memory[batch]
            new_states = self.new_state_memory[batch]
            rewards = self.reward_memory[batch]
            actions = self.action_memory[batch]
            terminal = self.terminal_memory[batch]

            return states, actions, rewards, new_states, terminal

class CriticNetwork(nn.Module): # nn.Module give access to important methods, e.g. train, eval and parameters for updating the weights of the neural network.
      def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
            super(CriticNetwork, self).__init__()
            self.input_dims = input_dims
            self.fc1_dims = fc1_dims
            self.fc2_dims = fc2_dims
            self.n_actions = n_actions
            self.checkpoint_file = os.path.join(chkpt_dir, f'{name}_ddpg')
            self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
            f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
            T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
            T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

            # The bach normalization helps with convergence of your model.
            # (you don't get good convergence if you don't have it, so leave it in)
            self.bn1 = nn.LayerNorm(self.fc1_dims)

            self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
            f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
            T.nn.init.uniform_(self.fc2.weight.data, -f2, f2) # the first parameter is the tensor you want to initialize, and then the lower and upper boundaries
            T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
            self.bn2 = nn.LayerNorm(self.fc2_dims)
            # The fact that we have a normalization layer (a batch norm type) means that we have to use
            # the eval and train funcions later

            # The critic network is also going to get a action value, because the action value function takes in the states and actions as input,
            # but we're going to add it in at the very end of the network
            self.action_value = nn.Linear(self.n_actions, fc2_dims)
            f3 = 0.003 # comes from the paper
            self.q = nn.Linear(self.fc2_dims, 1) # since the output is a scalar value, it has 1 output
            T.nn.init.uniform_(self.q.weight.data, -f3, f3)
            T.nn.init.uniform_(self.q.bias.data, -f3, f3)

            self.optimizer = optim.Adam(self.parameters(), lr=beta)

            # if you want to run this on GPU, because its expensive algorithm
            # I don't recommend runinig this on CPU
            self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

            # send the whole network to your device
            self.to(self.device)

      def forward(self, state, action):
            # keep in mind the actions are continuous so it's a vector
            # (in this case length 2, for the continuous 'lunar lander' game environment, 2 real numbers in a list or numpy array format)

            state_value = self.fc1(state)
            state_value = self.bn1(state_value)

            # activate it
            state_value = F.relu(state_value)
            # it is open for debate whether or not you want to do the value before or after the batch normalization.
            # I think it make more sense to do the batch normalization first, because when you calculating batch statistics,
            # if you apply the relu first, then your lopping off everything below zero, that means your statistics are going to
            # be skewed toward the positive end when perhaps the real distibution has a mean of zero or negative (instead of positive)
            # which you wouldn't see if you used the value funcion before the batch normalization.

            state_value = self.fc2(state_value)
            state_value = self.bn2(state_value)

            # so we've done the batch normalization, we don't want to activate it yet.
            # what we want to do first is take into account the action value.
            # we activate (relu) the action through the action value layer.
            # we're not going to calculate batch statistics on this, so we don't need to worry about that.
            action_value = F.relu(self.action_value(action))
            # but what we want to do is add the two values together
            state_action_value = F.relu(T.add(state_value, action_value))
            # (another thing that wonky, play with it if you want) I'm double reluing the action value function/quantity.
            # one relu on action_value and second relu on the add of the action value.
            # This is what worked for me, alternative is to do relu after the add.
            # relu is a non commutative function with add, that means that if you do an addition first and then a relu,
            # that is different than doning relu first on the two values and the add them.
            # (i.e.: relu(-10)+5 = 0+5 = 5. relu(-10+5) = relu(-5) = 0)
            # I've seen other implementation that do it differently. This is what worked for me.

            # get the actual state action value, by passing the additive quantity through our final layer of the network
            state_action_value = self.q(state_action_value)

            return state_action_value

      # book keeping
      def save_checkpoint(self):
            print('... saving checkpoint ...')
            T.save(self.state_dict(), self.checkpoint_file)

      def load_checkpoint(self):
            print('... loading checkpoint ...')
            self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
      # pretty similar to the critic network it will just have a different structure.
      # in particular we don't have the actions.

      def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
            super(ActorNetwork, self).__init__()
            self.input_dims = input_dims
            self.n_actions = n_actions
            self.fc1_dims = fc1_dims
            self.fc2_dims = fc2_dims
            self.checkpoint_file = os.path.join(chkpt_dir, f'{name}_ddpg')

            self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # * is to unpack the tuple
            # select initialization interval value
            f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
            # initialize the first layer uniformly within that interval
            T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
            T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

            self.bn1 = nn.LayerNorm(self.fc1_dims) # don't need to initialize

            self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
            f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
            T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
            T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

            self.bn2 = nn.LayerNorm(self.fc2_dims) # don't need to initialize

            f3 = 0.003 # came from the paper
            # mu is the representation of the policy in this case.
            # it's real vector of shape n actions, it's the actual action not the probability.
            # because it's deterministic it's linear layer
            self.mu = nn.Linear(self.fc2_dims, self.n_actions)
            T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
            T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

            self.optimizer = optim.Adam(self.parameters(), lr=alpha)

            self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
            self.to(self.device)
      def forward(self, state):
            state_value = self.fc1(state)
            state_value = self.bn1(state_value)
            state_value = F.relu(state_value)

            state_value = self.fc2(state_value)
            state_value = self.bn2(state_value)
            state_value = F.relu(state_value)

            state_value = T.tanh(self.mu(state_value))
            # tanh will bound it between -1 and +1 (important for many environment).
            # later we can multiply it by the actual action bounds, because some evnironment have a max action of +/-2
            return state_value

      # book keeping
      def save_checkpoint(self):
            print('... saving checkpoint ...')
            T.save(self.state_dict(), self.checkpoint_file)

      def load_checkpoint(self):
            print('... loading checkpoint ...')
            self.load_state_dict(T.load(self.checkpoint_file))

class Agent_DDPG(Agent):
      # env to get the action space
      # gamma (agent discount factor) = 0.99 is 1% to value reward now that the future, because there's uncertainty in the future, typically is 0.95 to 0.99
      # n_action is 2, becase many environment has only 2 action
      # layers size came from the paper

      def __init__(self, alpha:float=0.0001, beta:float=0.001, input_dims:tuple=(2,),
                   tau:float=0.001, gamma:float=0.99, n_actions:int=7, max_size:int=1000000,
                   layer1_size:int=400, layer2_size:int=300, batch_size:int=64):
            self.alpha = alpha
            self.beta = beta
            self.input_dims = input_dims
            self.gamma = gamma
            self.n_actions = n_actions
            self.tau = tau
            self.memory = ReplayBuffer(max_size, input_dims, n_actions)
            self.batch_size = batch_size
            self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='Actor')

            # much like the deep Q network algorithm, this uses taget networks as well as the base network,
            # so it's an off policy method.
            # This will allow us to have multiple different agents with similar names.
            self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='TargetActor')

            self.critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='Critic')
            self.target_critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='TargetCritic')
            # This is very similar to key learning, where you have Q eval and Q next (or Q target, whatever you want to call it).
            # It's the same concept.

            self.noise = OUActionNoise(mu=np.zeros(n_actions))

            self.update_network_parameters(tau=1)
            # what this does is to solve a problem of a moving target,
            # so in Q learning if you use one network to calculate both the action as well as the value of that action,
            # then you're really chasing a moving target,
            # beacuse you're updating that estimate turn, so you end up using the same parameters for both, and it can lead to divergence.
            # The solution to that is to use a target network, that learns the values of these states and action combinations and then the other network is what learns the policy.
            # Then periodically you have to override the target networks parameters with the evaluation network parameters,
            # and this function will do precisely that.
            # Except that we have 4 networks instead of 2.

            # me: do main init commands to keep elegant use of agent
            self._preparation_init()

      def choose_action(self, state):
            # very important: you have to put the actor into evaluation mode.
            # this doesn't perform an evaluation step,
            # this just tells pytorch that you don't want to calculate statistics for the batch normalization.
            # This is very critical, if you don't do this the agent will not learn,
            # and it doesn't do what you think the name implies it would do.
            # The complementary function 'train', it doesn't perform a training step, it puts it in training mode where it does store
            # those statistics in the graph for the batch normalization.
            # If you don't do batch norm then you don't need to do this (call eval and train)
            # (Dropout has the same 'tick' where you have to call the eval and train funcions)
            self.actor.eval()

            # to turn to a cuda float tensor
            state = T.tensor(state, dtype=T.float).to(self.actor.device)

            # get the actual action from the actor network
            mu = self.actor(state).to(self.actor.device)

            # NOTE: if not in train mode (but in real time mode) you should comment the bottom line (noise).
            mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
            # NOTE cont: and replace it with:
            # mu_prime = mu

            self.actor.train()

            # This is an idiom within pytorch, where you have to detach, otherwise it doesn't give you the actual numpy value.
            # Otherwise it will pass out a tensor, which doesn't work, beacuse you can't pass a tensor into the open ai gym.
            self.last_action_probs  = mu_prime.cpu().detach().numpy()  # action probabilities, type ndarry: (7,)]
            return self.last_action_probs

      ############################################

      def choose_action_and_prep(self, state, done) -> int:
            # me: do main commands to keep elegant use of agent
            if state is not None: # state is None only at the beginning
                  self._preparation_selection(state, done)

            action = None
            if not done:
                  action_probs = self.choose_action(state)
                  # TODO compair action[0] to (is or ==) np.nan don't work, it is of type float32, fix comparison.
                  if str(action_probs[0]) == 'nan':
                        logging.error(f'action is None {action_probs}')
                        raise RuntimeError('action is nan')
            else:
                  #action = mp.zeros(self.n_actions)
                  #action = random.randrange(self.n_actions)
                  action_probs = np.random.randint(0, 10, size=self.n_actions)
            self.last_action_int = int(np.argmax(action_probs)) # this is a numpy of single value that is passed as int
            #logging.info(f'{self.__class__.__name__} choosen action {action_int}')
            return self.last_action_int

      def _preparation_init(self):
            # init for all games
            self.game_index = 0
            self.best_score = 0
            self.score_history = []
            self._preparation_game_init()

      def _preparation_selection(self, state, done):
            if self.last_state is None:
                  self.last_state = state
            elif self.last_action_probs is None:
                  logging.warning(f'rememberring None')
            else:
                  reward = self._get_reward(state)
                  # If the game is over (done=True), arg state should be None.
                  # so the model remember and learn that the last action in last state leads to None state.

                  self.remember(self.last_state, self.last_action_probs, reward, state, done)
                  self.learn()
                  self.score += reward
                  self.last_state = state
                  if done:
                        logging.info(f'{self.__class__.__name__}._preparation_selection: game over, state {self.last_state}, score {self.score}, last_action {self.last_action_int}')
                        self._preparation_game_over()
                        self._preparation_game_init()
                        logging.info(f'{self.__class__.__name__}._preparation_selection: restart, state {self.last_state}, score {self.score}, last_action {self.last_action_int}')

      def _preparation_game_over(self):
            self.game_index += 1
            self.score_history.append(self.score)
            self.avg_score = np.mean(self.score_history[-100:])
            if self.avg_score > self.best_score:
                  self.best_score = self.avg_score
                  self.save_models()
            logging.info(f'{self.__class__.__name__}._preparation_game_over: score {self.score}, average score {self.avg_score}')

      def exit_clean(self):
            self._plot_learning_curve()

      def _plot_learning_curve(self):
            running_avg = np.zeros(len(self.score_history))
            for i in range(len(running_avg)):
                  running_avg[i] = np.mean(self.score_history[max(0, i-100):(i+1)])

            x = [i+1 for i in range(self.game_index)]
            plt.plot(x, running_avg)
            plt.title('Running average of previous 100 scores')
            filename = f'Mouse_alpha_{self.alpha}_beta_{self.beta}_{self.game_index}_games_time_{datetime.datetime.now()}'
            figure_file = f'plots/{filename}.png'
            plt.savefig(figure_file)

      def _preparation_game_init(self):
            # init for each game/done=True
            self.last_state = None
            self.last_action_probs = None
            self.last_action_int = None
            self.done = False
            self.score = 0
            self.noise.reset()

      def _get_reward(self, state):
            # the grid is 10x10, and location ~(5,5) which is the middle is where we want to be.
            # so state which is location of 4-6 in both axes (x and y, state[0] and state[1]).
            reward = 0.0
            if state is not None and 4<=state[0] and state[0]<=6 and 4<=state[1] and state[1]<=6:
                  reward += 0.3
            if state is not None and 4.5<=state[0] and state[0]<=6.5 and 4.5<=state[1] and state[1]<=6.5:
                  reward += 0.3
            if state is not None and state[0]==5 and state[1]==5:
                  reward += 0.3
            return reward

      ############################################

      def remember(self, state, action, reward, new_state, done):
            # to store state transitions (kind of interface for replay memory class)
            self.memory.store_transition(state, action, reward, new_state, done)

      def learn(self):
            # you don't want to learn untill you filled up at least batch size of your memory buffer
            if self.memory.mem_cntr < self.batch_size:
                  return
            # sample your memory
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

            # Turn all of those into tensors, because they come back as numpy arrays.
            # You can put them on criric.device, as long as you're on the same device it doesn't matter
            # (I do this for consistency, because these values will be used in the critic network).
            reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
            done = T.tensor(done).to(self.critic.device)
            new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
            action = T.tensor(action, dtype=T.float).to(self.critic.device)
            state = T.tensor(state, dtype=T.float).to(self.critic.device)

            # send eveything to eval mode
            # for the targets it may not be that important, I did it for consistency
            self.target_actor.eval()
            self.target_critic.eval()
            self.critic.eval()

            # we want to calculate the target action much like you do in the bellman equation for deep Q learning.
            target_actions = self.target_actor.forward(new_state)
            # the new states
            critic_value_ = self.target_critic.forward(new_state, target_actions)
            # we getting the target actions from the target actor network, i.e. what actions it should take based on the target actors estimates
            # and then plugging that into the state value function for the target critic network

            critic_value = self.critic.forward(state, action)
            # i.e., what was the estimate of the values of the states and actions we actually encountered in our subset of the replay buffer.

            # the target that we gonna move towards.
            # I use a loop instead of a vectorized implementation, because the vectorized implementation is a little bit tricky.
            # If you don't do it properly you can end up with something of shape batch size by batch size, which won't flag an error, but gives you the wrong answer, and you don't get learning.
            target = []
            for j in range(self.batch_size):
                  target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
                  # this is what the done flags are for. if the episode is over then the value of the resulting state is multiplied by zero,
                  # so you don't take it into account (you only take into account the reward from the current state).
            target = T.tensor(target).to(self.critic.device)
            # reshape
            target = target.view(self.batch_size, 1)

            # calculation of the loss funcion
            # (set the critic back into trainig mode, because we performed the evaluation, now we want to calculate the values for batch normalization)
            self.critic.train()
            # in pytorch, whenever you calculate the loss function you want to zero your gradients,
            # that's so gradients from previous steps don't accumulate and interfere with the calculation
            # (it can slow stuff down)
            self.critic.optimizer.zero_grad()
            critic_loss = F.mse_loss(target, critic_value)
            critic_loss.backward()
            self.critic.optimizer.step()

            # set the critic into evaluation mode for the calculation of the loss for our actor network.
            self.critic.eval()
            self.actor.optimizer.zero_grad()
            # It is confusing, this is one of the ways in which tensorflow is superior to pytorch, you don't have this quirk of eval and train.
            # I tend to like tensorflow a little bit better.

            mu = self.actor.forward(state)
            self.actor.train()
            actor_loss = -self.critic.forward(state, mu)
            actor_loss = T.mean(actor_loss)
            actor_loss.backward()
            self.actor.optimizer.step()
            # finished learning

            # update the network parameters for your target actor and target critic networks.
            self.update_network_parameters()

      def update_network_parameters(self, tau=None):

            # tau is parameter that allows the update of the target network to gradually approach the evaluation networks.
            # This is important for a nice slow convergence, you don't want to take too large steps in between updates.
            # (tau is a small number, much much less than 1)
            if tau is None:
                  tau = self.tau
                  # This seem mysterious, the reason I'm doing this is because at __init__ we say update_network_parameters(tau=1),
                  # this is because in the beginning we want all the networks to start with the same weights, and we call it with tau=1
                  # and in that case tau is not None and we get the update rule in the next lines of code in this method.
                  # This is more hocus-pocus with pytorch.

            # it'll get all the names of the parameters from these networks
            actor_params = self.actor.named_parameters()
            critic_params = self.critic.named_parameters()
            target_actor_params = self.target_actor.named_parameters()
            target_critic_params = self.target_critic.named_parameters()

            # now that we have the parametes, let's make them into a dictionary
            # (that makes iterating them much easier, because this is actually a generator - I belive)
            critic_state_dict = dict(critic_params)
            actor_state_dict = dict(actor_params)
            target_critic_dict = dict(target_critic_params)
            target_actor_dict = dict(target_actor_params)

            # now we want to iterate over these dictionaries and copy parameters.
            # It iterates ovet this dictionary, and update the values from this particular network
            # (you can see when tau is 1, you get only the the critic state (identity) without target critic).
            for name in critic_state_dict:
                  critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_dict[name].clone()
            # Loads the target critic with that parameters.
            # At the beginning it'll load it with the parameters from the initial critic network
            self.target_critic.load_state_dict(critic_state_dict)

            for name in actor_state_dict:
                  actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_dict[name].clone()
            self.target_actor.load_state_dict(actor_state_dict)

      # book keeping (because it take a long time to train)
      def save_models(self):
            self.actor.save_checkpoint()
            self.critic.save_checkpoint()
            self.target_actor.save_checkpoint()
            self.target_critic.save_checkpoint()

      def load_models(self):
            self.actor.load_checkpoint()
            self.critic.load_checkpoint()
            self.target_actor.load_checkpoint()
            self.target_critic.load_checkpoint()
