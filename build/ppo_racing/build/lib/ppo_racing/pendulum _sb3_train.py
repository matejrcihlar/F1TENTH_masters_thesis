import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os
import pandas as pd
import matplotlib.pyplot as plt
from custom_policy import CustomActorCriticPolicy
import torch as T
from f110_gym.envs.base_classes import Simulator, Integrator

# Create and wrap the environment
env = gym.make('Pendulum-v1')

#env = Monitor(env, filename='logs/monitor.csv')

# Create a folder for logs and models
log_dir = "sb3_logs"
os.makedirs(log_dir, exist_ok=True)

# Define evaluation environment and callback for monitoring
eval_env = gym.make('Pendulum-v1')
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=20000,
    deterministic=False,
    render=True,
)

policy_kwargs = dict(activation_fn=T.nn.ReLU,
                     net_arch=dict(pi=[64, 64], vf=[64, 64]))

# Initialize PPO agent with MLP policy
model = PPO(
    policy='MlpPolicy',
    env=env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=log_dir,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    learning_rate=3e-4,
    clip_range=0.2,
)

# Train the model
model.learn(total_timesteps=200_000, callback=eval_callback)

# Save the final model
model.save(os.path.join(log_dir, "ppo_pendulum_final"))

# Optional: Test the trained policy
'''obs = eval_env.reset()
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = eval_env.step(action)
    eval_env.render()
    if done:
        obs = eval_env.reset()

eval_env.close()

# Load Monitor log
monitor_file = os.path.join(log_dir, 'monitor.csv')
data = pd.read_csv(monitor_file, skiprows=1)  # Skip first header line
rewards = data['r'].tolist()

# Plot rolling average
window = 100
averaged_rewards = pd.Series(rewards).rolling(window).mean()

plt.figure(figsize=(10, 6))
plt.plot(rewards, label='Episode Reward')
plt.plot(averaged_rewards, label=f'{window}-Episode Moving Avg', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('PPO on Pendulum-v1')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(log_dir, 'reward_plot.png'))
plt.show()'''

