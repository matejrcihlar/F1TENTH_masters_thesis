import numpy as np
import scipy.interpolate as sp_int
import csv
from ppo_agent import Agent
from utils import plot_learning_curve
import gym
import rclpy
import os
from reward_node import RewardNode
import time
from f110_gym.envs.base_classes import Integrator
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv

NUM_ENVS = 8  # Number of parallel environments

def make_env(map_path, flat_waypoints):
    def _init():
        env = gym.make('f110_gym:f110-v0',
                       map=map_path,
                       map_ext='.png',
                       num_agents=2,
                       integrator=Integrator.RK4,
                       timestep=0.05,
                       render_mode=None)
        return env
    return _init

def main():
    rclpy.init()
    map_path = '/home/mrc/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_map'

    '''# Load and interpolate waypoints
    with open('/home/mrc/sim_ws/src/f1tenth_lab6_template/waypoints.csv', mode='r') as file:
        reader = csv.reader(file)
        waypoints = [list(map(float, row)) for row in reader]
    waypoints = np.transpose(waypoints)
    tck, _ = sp_int.splprep(waypoints, s=0)
    u_fine = np.linspace(0, 1, 100)
    x_vals, y_vals = sp_int.splev(u_fine, tck)
    flat_waypoints = np.vstack((x_vals, y_vals)).T.flatten()'''

    waypoints = np.loadtxt('/home/mrc/sim_ws/src/ppo_racing/ppo_racing/Spielberg_waypoints.csv', delimiter=',')

    # Keep only the first two columns
    waypoints = waypoints[::5,:]
    flat_waypoints = waypoints[:, :2].flatten()

    env_fns = [make_env(map_path, flat_waypoints) for _ in range(NUM_ENVS)]
    env = DummyVecEnv(env_fns)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    figure_file = f'plots/ros2_agent_learning_curve_{timestamp}.png'

    # PPO hyperparameters
    N = 4096
    batch_size = 64
    n_epochs = 10
    alpha = 3e-4
    n_games = 10000
    beta = 0.3

    input_size = 108 + 6 * 2 + 60 + 1 + 2
    agent = Agent(n_actions=2, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=input_size)

    best_score = -np.inf
    score_history = []
    learn_iters = 0
    n_steps = 0
    reward_nodes = [RewardNode() for _ in range(NUM_ENVS)]
    prev_actions = np.zeros((NUM_ENVS, 2), dtype=np.float32)
    scores = np.zeros(NUM_ENVS)

    # Initial reset
    poses_batch = np.array([reset_poses(flat_waypoints)[0] for _ in range(NUM_ENVS)])
    obs_batch = [env.envs[i].reset(poses_batch[i])[0] for i in range(NUM_ENVS)]
    wp_start_ids = [reset_poses(flat_waypoints)[1] for _ in range(NUM_ENVS)]
    for rn, idx in zip(reward_nodes, wp_start_ids):
        rn.remember_start(idx)
    states = np.array([preprocess_observation(obs_batch[i], flat_waypoints, prev_actions[i]) for i in range(NUM_ENVS)])

    for i in range(n_games):
        done_envs = np.array([False] * NUM_ENVS)
        durations = np.zeros(NUM_ENVS)

        while not np.all(done_envs):
            actions, probs, vals = agent.choose_action_batch(states)
            smoothed = beta * prev_actions + (1 - beta) * actions
            prev_actions = smoothed

            # Clip actions and prepare full 2-agent action arrays
            clipped = np.stack([
                np.clip(smoothed[:, 0], -0.34, 0.34),
                np.clip(smoothed[:, 1], 0.0, 3.0)
            ], axis=-1)
            action_full = np.zeros((NUM_ENVS, 2, 2))
            action_full[:, 0, :] = clipped

            obs_, rewards, dones, infos = [], [], [], []
            for i_env in range(NUM_ENVS):
                obs_next, reward, done, info = env.envs[i_env].step(action_full[i_env])
                obs_.append(obs_next)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)

            rewards_np = np.zeros(NUM_ENVS)
            for j in range(NUM_ENVS):
                duration_exceeded = durations[j] >= 10.0
                if obs_[j]['collisions'][0] == 1 or duration_exceeded:
                    done_envs[j] = True
                rewards_np[j] = reward_nodes[j].get_reward(obs_[j], actions[j])
                if duration_exceeded and obs_[j]['collisions'][0] == 0:
                    rewards_np[j] += 0.2

                next_state = preprocess_observation(obs_[j], flat_waypoints, prev_actions[j])
                agent.remember(states[j], actions[j], probs[j], vals[j], rewards_np[j], done_envs[j])
                states[j] = next_state
                scores[j] += rewards_np[j]
                durations[j] += 0.05

            n_steps += NUM_ENVS
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

        score_history.extend(scores.tolist())
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(f'episode {i} | avg score: {avg_score:.1f} | learn iters: {learn_iters} | total steps: {n_steps}')

        # Reset environments after episode
        poses_batch = np.array([reset_poses(flat_waypoints)[0] for _ in range(NUM_ENVS)])
        obs_batch = [env.envs[k].reset(poses_batch[k])[0] for k in range(NUM_ENVS)]
        wp_start_ids = [reset_poses(flat_waypoints)[1] for _ in range(NUM_ENVS)]
        for rn, idx in zip(reward_nodes, wp_start_ids):
            rn.remember_start(idx)
        states = np.array([preprocess_observation(obs_batch[i], flat_waypoints, np.zeros(2)) for i in range(NUM_ENVS)])
        prev_actions = np.zeros((NUM_ENVS, 2))
        scores = np.zeros(NUM_ENVS)

    x = [i+1 for i in range(len(score_history))]
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    plot_learning_curve(x, score_history, figure_file)

# Keep preprocess_observation, reset_poses, get_next_waypoints_flat exactly as before

def preprocess_observation(obs, WAYPOINTS, prev_action, lidar_max_range=30.0, max_speed=3.0):
    ego = obs['ego_idx']
    opp = 1 - ego  # assuming 2 agents

    # Ego scan normalized
    scan = np.clip(obs['scans'][ego], 0.0, 30.0) / 30.0
    scan_short = []

    for k in range(len(scan)):
        if k%10==0:
            scan_short.append(scan[k])
    

    # Ego pose
    ego_pose = np.array([
        obs['poses_x'][ego] / 100.0,
        obs['poses_y'][ego] / 100.0,
        (obs['poses_theta'][ego]) / np.pi
    ])

    # Ego velocity
    ego_vel = np.array([
        obs['linear_vels_x'][ego] / max_speed,
        obs['linear_vels_y'][ego] / max_speed,
        obs['ang_vels_z'][ego] / 3.2
    ])

    # Opponent relative pose
    rel_pose = np.array([
        (obs['poses_x'][opp] - obs['poses_x'][ego]) / 100.0,
        (obs['poses_y'][opp] - obs['poses_y'][ego]) / 100.0,
        (obs['poses_theta'][opp] - obs['poses_theta'][ego]) / np.pi
    ])

    # Opponent velocity
    opp_vel = np.array([
        obs['linear_vels_x'][opp] / max_speed,
        obs['linear_vels_y'][opp] / max_speed,
        obs['ang_vels_z'][opp] / 3.2
    ])

    # Collision (convert to scalar float)
    collision = np.array([float(obs['collisions'][ego])])

    WAYPOINTS = WAYPOINTS / 100.0

    next_waypoints = get_next_waypoints_flat((ego_pose[0],ego_pose[1]), WAYPOINTS)

    prev_action_array = np.array(prev_action)





    # Combine all
    state_vector = np.concatenate([
        scan_short,
        ego_pose,
        ego_vel,
        rel_pose,
        opp_vel,
        collision,
        next_waypoints,
        prev_action_array
    ])

    state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=1e3, neginf=-1e3)
    state_vector = np.clip(state_vector, -1e3, 1e3)


    return state_vector.astype(np.float32)

def reset_poses(flat_waypoints):
    """
    Resets the poses of the ego and opponent car using looped, flattened waypoints.

    Args:
        flat_waypoints (np.ndarray): 1D array [x0, y0, x1, y1, ..., xN, yN]

    Returns:
        poses (np.ndarray): shape (2, 3), each row is [x, y, yaw] for ego and opponent.
    """
    assert len(flat_waypoints) % 2 == 0, "Waypoint list must be even-length."
    waypoints = flat_waypoints.reshape(-1, 2)
    num_waypoints = waypoints.shape[0]

    # Choose a random index
    ego_idx = np.random.randint(0, num_waypoints)

    # Circular indexing for yaw calculation
    wp_prev = waypoints[(ego_idx - 1) % num_waypoints]
    wp_curr = waypoints[ego_idx]
    wp_next = waypoints[(ego_idx + 1) % num_waypoints]
    ego_yaw = np.arctan2(wp_next[1] - wp_prev[1], wp_next[0] - wp_prev[0])

    # Opponent car placed 2 waypoints ahead
    opp_idx = (ego_idx + 15) % num_waypoints
    wp_prev_opp = waypoints[(opp_idx - 1) % num_waypoints]
    wp_curr_opp = waypoints[opp_idx]
    wp_next_opp = waypoints[(opp_idx + 1) % num_waypoints]
    opp_yaw = np.arctan2(wp_next_opp[1] - wp_prev_opp[1], wp_next_opp[0] - wp_prev_opp[0])

    poses = np.array([
        [wp_curr[0], wp_curr[1], ego_yaw],
        [wp_curr_opp[0], wp_curr_opp[1], opp_yaw]
    ])

    #poses = np.array([[0.0, 0.0, 0.0], [2.0, 0.5, 0.0]])
    #ego_idx = 0

    return poses, ego_idx

def get_next_waypoints_flat(car_pos, flat_waypoints, num_points=30):
    """
    Find the closest waypoint to the car and return the next `num_points` as a flat array.
    
    Args:
        car_pos (tuple): (x, y) position of the car
        flat_waypoints (list or np.ndarray): Flattened list of waypoints [x0, y0, x1, y1, ...]
        num_points (int): Number of future waypoints to return
    
    Returns:
        np.ndarray: Flattened array of the next `num_points` waypoints
    """
    # Reshape into [N, 2]
    waypoints = np.array(flat_waypoints).reshape(-1, 2)

    # Find closest index
    dists = np.linalg.norm(waypoints - np.array(car_pos), axis=1)
    closest_idx = np.argmin(dists)

    # Select next waypoints with wrap-around
    total = waypoints.shape[0]
    indices = [(closest_idx + i) % total for i in range(num_points)]
    next_waypoints = waypoints[indices]

    # Flatten and return
    return next_waypoints.flatten()



if __name__ == '__main__':
    main()
