import gym
import numpy as np
from deep_deterministic_policy_gradient import Agent

#from utils import plot_learning_curve
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def main():
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()
    # def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=3, max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64):
    agent = Agent(alpha=0.0001, beta=0.001,
                    input_dims=env.observation_space.shape, tau=0.001, env=env,
                    batch_size=64, layer1_size=400, layer2_size=300,
                    n_actions=env.action_space.shape[0])
    n_games = 10
    filename = 'LunarLander_alpha_' + str(agent.alpha) + '_beta_' + \
                str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            env.render()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
    x = [i+1 for i in range(n_games)]
    env.close()
    plot_learning_curve(x, score_history, figure_file)

def openai_gym_demo_random_agent():
    #gym.envs.box2d.lunar_lander.demo_heuristic_lander()
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) # take a random action
    env.close()

if __name__ == '__main__':
    main()
