
import gym
import numpy as np
import os
from ppo_agent import Agent
from datetime import datetime
from utils import plot_learning_curve

def main():
    env = gym.make("Pendulum-v0")  # Continuous obs and actions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = Agent(
        n_actions=act_dim,
        input_dims=obs_dim,
        alpha=3e-4,
        batch_size=64,
        n_epochs=10
    )

    N = 2048  # learning every N steps
    n_episodes = 1000
    n_steps = 0
    learn_iters = 0
    best_score = -np.inf
    score_history = []

    figure_file = f'plots/pendulum_agent_learning_curve.png'

    for i in range(n_episodes):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            action, prob, val = agent.choose_action(obs)
            scaled_action = np.clip(action*2, -2.0, 2.0)  # Pendulum expects [-2, 2]

            obs_, reward, done, _= env.step(scaled_action)
            

            agent.remember(obs, action, prob, val, reward, done)

            obs = obs_
            score += reward
            n_steps += 1

            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(f"Episode {i+1} | Score: {score:.1f} | Avg Score: {avg_score:.1f} | Steps: {n_steps} | Learns: {learn_iters}")

    x = [i+1 for i in range(len(score_history))]
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    plot_learning_curve(x, score_history, figure_file)

if __name__ == "__main__":
    main()
