import datetime
import logging

logging.basicConfig(level=logging.INFO)
import gym
import gym_mouse_lib.gym_mouse
from agent import Agent_DDPG

# from utils import plot_learning_curve
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def realtime():
    env = gym.make('Mouse-v0')
    env.reset()
    agent = Agent_DDPG()
    # no for n_games (rl)
    try:
        location = [5,5]
        while True:
            # no receive image (server)
            action = agent.choose_action_and_prep_with_env(location)
            # TODO unmark line below to render env in every turn
            #env.render()
            # location, done, info = env.step(action)
            location = env.step_realtime(action)  # location should be a list with 2 numbers in range -1-11
            # no send action (server)
    except Exception as e:
        print(f'main ddpg stopped, exit clean.\ne = {e}')
        agent.exit_clean()
        env.close()

def main():
    # env = gym.make('LunarLanderContinuous-v2')
    env = gym.make('Mouse-v0')
    agent = Agent_DDPG()

    n_games = 10000
    filename = 'Mouse_alpha_' + str(agent.alpha) + '_beta_' + \
               str(agent.beta) + '_' + str(n_games) + '_games time_' + str(datetime.datetime.now())
    figure_file = 'plots/' + filename + '.png'

    best_score = 0  # env.reward_range[0]
    score_history = []
    # agent.load_models()
    for i in range(n_games):
        state = env.reset()
        done = False
        score = 0
        agent.noise.reset()

        while not done:
            action = agent.choose_action(state)
            env.render()
            new_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, new_state, done)
            agent.learn()
            score += reward
            state = new_state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(f'episode {i}, score {score}, average score {avg_score}')
    x = [i + 1 for i in range(n_games)]
    env.close()
    plot_learning_curve(x, score_history, figure_file)


def openai_gym_demo_random_agent():
    # gym.envs.box2d.lunar_lander.demo_heuristic_lander()
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()


if __name__ == '__main__':
    # main()
    realtime()
