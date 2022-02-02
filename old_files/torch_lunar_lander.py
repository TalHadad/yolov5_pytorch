from deep_deterministic_policy_gradient import Agent
import gym
import numpy as np

# from utils import plotLearning
import matplotlib.pyplot as plt

def main():
    env = gym.make('LunarLanderContinuous-v2')
    # I didn't multiply by the action space high in the function for the choose action,
    # don't worry, that'll be in the tensorflow implementation,
    # or I can leave it as an exercise to the reader, it doesn't matter for this environment.
    # when we get to an environment where it is matter, I'll be nore diligent about that.
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env, batch_size=64, layer1_size=400, layer2_size=300, n_actions=2) # 25*10^-5, 25*10^-4

    # another intersting thing is that we have to set the random seed, this is not something I've done before,
    # but this is a highly sensitive learning method, so if you read the original paper they do averages over 5 runs,
    # and that's because every run is a little bit different, and I suspect that's why they had to initialize
    # the weights and biases within such a narrow range, you don't what to go all the way from +1 and -1,
    # when you constrain to something much more narrow.
    # so we have to set the numpy random seed to some value instead of none, I use 0, there are more value that is been used
    # (see what happen when you try other seed values)
    np.random.seed(0)

    score_history = []
    for i in range(1000):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            act = agent.choose_action(obs)
            env.render()
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state

        score_history.append(score)
        print(f'episode {i} score {score} 100 game average {np.mean(score_history[-100:])}')
        if i % 25 == 0:
            agent.save_models()

    env.close()
    #filename = 'lunar-lander.png'
    #plotLearning(score_history, filename, window=100)

def openai_gym_demo_random_agent():
    #gym.envs.box2d.lunar_lander.demo_heuristic_lander()
    env = gym.make('LunarLanderContinuous-v2')
    obs = env.reset()
    for _ in range(1000):
        env.render()
        new_state, reward, done, info = env.step(env.action_space.sample()) # take a random action
    env.close()

if __name__=="__main__":
    main()
