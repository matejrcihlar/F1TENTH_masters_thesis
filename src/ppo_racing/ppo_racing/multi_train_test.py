import numpy as np
import scipy.interpolate as sp_int
import csv
from ppo_agent import Agent, MultiAgentTrainer
from utils import plot_learning_curve_multi
import gym
import rclpy
import os
from reward_node import RewardNode
import time
from f110_gym.envs.base_classes import Simulator, Integrator
from datetime import datetime

def main():
    rclpy.init()
    env = gym.make('f110_gym:f110-v0',
                            map='/home/mrc/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_map',
                            map_ext='.png',
                            num_agents=2,
                            integrator = Integrator.RK4,
                            timestep = 0.05,
                            render_mode=None)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # PPO hyperparameters
    N = 4096             # steps before update
    batch_size = 64
    n_epochs = 6
    alpha = 1e-4
    n_games = 10000
    figure_file = f'plots/ros2_agent_learning_curve_multi_{N}_{batch_size}_{n_epochs}_{timestamp}.png'
    beta = 0.0
    episode_length = 30.0 #seconds
    

    scan_size = 108
    odom_size = 6
    waypoint_size = 20
    collision_size = 1
    prev_action_size = 2


    input_size = scan_size + odom_size *2 -1+ waypoint_size *2+ collision_size + prev_action_size
    
    load_model = False

    if not load_model:
        parent_path = "/home/mrc/sim_ws/src/ppo_racing/ppo_racing/tmp"  # Change this to your target directory
        ppo_dir = create_unique_ppo_dir(parent_path)
        chkpt_dir = ppo_dir
    else:
        chkpt_dir = 'tmp/ppo_multi5'

    agent = Agent(
        n_actions=2,
        batch_size=batch_size,
        alpha=alpha,
        n_epochs=n_epochs,
        input_dims=input_size,
        chkpt_dir=chkpt_dir,
    )

    

    if load_model:
        agent.load_models()
    

    



    best_score_ego = -np.inf
    score_history_ego = []
    score_history_opp = []

    learn_iters = 0
    avg_score_ego = 0
    n_steps = 0
    

    # Load waypoints from CSV at module level (only once)
    #WAYPOINTS = np.loadtxt('raceline.csv', delimiter=',', skiprows=1)[:, :2]  # shape (N, 2)
    '''WAYPOINTS = []
    with open('/home/mrc/sim_ws/src/f1tenth_lab6_template/waypoints.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            WAYPOINTS.append([float(i) for i in row])

    
    WAYPOINTS= np.transpose(WAYPOINTS)
    
    tck, _ = sp_int.splprep(WAYPOINTS, s=0)
    u_fine = np.linspace(0, 1, 100)
    interpolated = sp_int.splev(u_fine, tck)
    x_vals, y_vals = interpolated
    WAYPOINTS = np.vstack((x_vals, y_vals)).T.flatten()'''
    waypoints = np.loadtxt('/home/mrc/sim_ws/src/ppo_racing/ppo_racing/Spielberg_waypoints.csv', delimiter=',')

    # Keep only the first two columns
    waypoints = waypoints[::5,:]
    waypoints = waypoints[:, :2]
    WAYPOINTS = waypoints.flatten()

    reward_node_ego = RewardNode(reward_function='combined')
    reward_node_opp = RewardNode(reward_function='combined')

    start_time = time.time()

    trainer = MultiAgentTrainer(agent)


    for i in range(n_games):
        poses, wp_start_i_ego, wp_start_i_opp = reset_poses(WAYPOINTS)
        #print("poses", poses)
        #time.sleep(1.5)
        observation, _, _, _ = env.reset(poses)
        reward_node_ego.remember_start(wp_start_i_ego)
        reward_node_opp.remember_start(wp_start_i_opp)
        prev_action_ego = np.array([0.0, 0.0])
        prev_action_opp = np.array([0.0, 0.0])
        #action_opp = np.array([0.0,0.0])#get_opp_action(observation, WAYPOINTS)
        observation_ego = preprocess_observation(observation, WAYPOINTS, prev_action_ego, ego=0)
        observation_opp = preprocess_observation(observation, WAYPOINTS, prev_action_opp, ego=1)
        done = False
        score_ego = 0
        score_opp = 0
        duration = 0.

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            elapsed_minutes = elapsed / 60
            print(f"[Episode {i+1}] Time elapsed: {elapsed_minutes:.2f} minutes")



        while not done:

            #if i % 100 == 0:
            #    env.render()
            
            action_ego, prob_ego, val_ego = agent.choose_action(observation_ego)
            action_ego = np.array(action_ego)
            action_sized_ego = np.array([np.clip(action_ego[0], -0.34, 0.34), np.clip(action_ego[1], 0.0, 3.0)])
            smoothed_action_ego = beta * prev_action_ego + (1 - beta) * action_sized_ego
            prev_action_ego = smoothed_action_ego
            action_opp, prob_opp, val_opp = agent.choose_action(observation_opp)
            action_opp = np.array(action_opp)
            action_sized_opp = np.array([np.clip(action_opp[0], -0.34, 0.34), np.clip(action_opp[1], 0.0, 3.0)])
            smoothed_action_opp = beta * prev_action_opp + (1 - beta) * action_sized_opp
            prev_action_opp = smoothed_action_opp
            #smoothed_action_opp = np.array([0.0, 0.0])
            action_full =np.array([[smoothed_action_ego[0], smoothed_action_ego[1]], [smoothed_action_opp[0], smoothed_action_opp[1]]])

            #if (i+1)%100==0: print(action_sized)
            obs_, reward, done, info = env.step(action_full)
            #print(obs_)
            #opp_action = np.array([0.0,0.0])#get_opp_action(obs_, WAYPOINTS)
            
            #if (i+1)%200==0:visualizer.draw_frame([(obs_['poses_x'][0], obs_['poses_y'][0] , obs_['poses_theta'][0]), \
            #                             (obs_['poses_x'][1], obs_['poses_y'][1] , obs_['poses_theta'][1])])
            #time.sleep(0.5)
            #print("collisions:", obs_['collisions'], "done:", done)
            
            if obs_['collisions'][0] == 1 or duration >= episode_length or obs_['collisions'][1] == 1:
                done = True
            else:
                done = False

            reward_ego = reward_node_ego.get_reward(obs_, np.array(action_sized_ego), ego=0)
            reward_opp = reward_node_opp.get_reward(obs_, np.array(action_sized_opp), ego=1)
            if (duration >= episode_length and obs_['collisions'][0] == 0) or (obs_['collisions'][1] == 1 and obs_['collisions'][0] == 0):
                reward_ego = reward_ego + 0.5
            if (duration >= episode_length and obs_['collisions'][1] == 0) or (obs_['collisions'][0] == 1 and obs_['collisions'][1] == 0):
                reward_opp = reward_opp + 0.5
            #done = check_done()
            #print("observation:", obs_)
            #time.sleep(0.5)
            #print("reward:", reward)
            obs_ego = preprocess_observation(obs_, WAYPOINTS, prev_action_ego, ego=0)
            obs_opp = preprocess_observation(obs_, WAYPOINTS, prev_action_opp, ego=1)
            
            n_steps += 1
            score_ego += reward_ego
            score_opp += reward_opp
            #agent.remember(observation_ego, action_ego, prob_ego, val_ego, reward_ego, done)
            trainer.remember('agent1', observation_ego, action_ego, prob_ego, val_ego, reward_ego, done)
            trainer.remember('agent2', observation_opp, action_opp, prob_opp, val_opp, reward_opp, done)

            if n_steps % N == 0:
                #agent.learn()
                trainer.learn('agent1')
                trainer.clear_memory()
                learn_iters += 1

            observation_ego = obs_ego
            observation_opp = obs_opp
            duration = duration + 0.05

        



        score_history_ego.append(score_ego)
        avg_score_ego = np.mean(score_history_ego[-100:])
        score_history_opp.append(score_opp)
        avg_score_opp = np.mean(score_history_opp[-100:])

        if avg_score_ego > best_score_ego:
            
            best_score_ego = avg_score_ego
            agent.save_models()

        if i%300 == 0:
            figure_file_rewards = f'reward_plots/Reward multi of episode:{i+1}.png'
            reward_node_ego.plot_logs(figure_file_rewards,i+1)

        print(f'episode {i} | score: {score_ego:.1f}|{score_opp:.1f} | avg score: {avg_score_ego:.1f}|{avg_score_opp:.1f} | '
              f'time steps: {n_steps} | learn iters: {learn_iters}')
        
    #visualizer.save_video(path=f"videos/training_run{timestamp}.mp4", fps=20)

    x = [i+1 for i in range(len(score_history_ego))]
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)
    plot_learning_curve_multi(x, score_history_ego, score_history_opp, figure_file)
    agent.plot_logs()

def preprocess_observation(obs, WAYPOINTS, prev_action, lidar_max_range=30.0, max_speed=3.0, ego=0):
    opp = 1 - ego

    # Normalize Lidar
    scan = np.clip(obs['scans'][ego], 0.0, lidar_max_range) / lidar_max_range
    scan_short = scan[::10]  # downsample

    # Ego pose
    ego_x = obs['poses_x'][ego]
    ego_y = obs['poses_y'][ego]
    ego_theta = obs['poses_theta'][ego]

    # Ego velocity
    ego_vel = np.array([
        obs['linear_vels_x'][ego] / max_speed,
        obs['linear_vels_y'][ego] / max_speed,
        obs['ang_vels_z'][ego] / 3.2
    ])

    # Opponent pose relative in map frame
    dx = obs['poses_x'][opp] - ego_x
    dy = obs['poses_y'][opp] - ego_y
    dtheta = obs['poses_theta'][opp] - ego_theta

    # Rotate to ego frame
    cos_yaw = np.cos(-ego_theta)
    sin_yaw = np.sin(-ego_theta)
    rel_x = dx * cos_yaw - dy * sin_yaw
    rel_y = dx * sin_yaw + dy * cos_yaw
    rel_theta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

    rel_pose = np.array([rel_x / 100.0, rel_y / 100.0, rel_theta / np.pi])

    # Transform opponent velocity into ego frame
    opp_vx = obs['linear_vels_x'][opp]
    opp_vy = obs['linear_vels_y'][opp]
    vx_rel = opp_vx * cos_yaw - opp_vy * sin_yaw
    vy_rel = opp_vx * sin_yaw + opp_vy * cos_yaw
    opp_ang = obs['ang_vels_z'][opp]

    opp_vel = np.array([
        (vx_rel - obs['linear_vels_x'][ego]) / max_speed,
        (vy_rel - obs['linear_vels_y'][ego]) / max_speed,
        (opp_ang - obs['ang_vels_z'][ego]) / 3.2
    ])

    # Raceline waypoints
    WAYPOINTS = WAYPOINTS / 100.0  # convert to meters
    next_waypoints = get_next_waypoints_flat((ego_x,ego_y), WAYPOINTS)
    next_opp_waypoints = get_next_waypoints_flat((dx+ego_x, dy+ego_y), WAYPOINTS)
    wp_ego = waypoints_to_ego_frame(next_waypoints, [ego_x / 100.0, ego_y / 100.0, ego_theta])
    wp_opp = waypoints_to_ego_frame(next_opp_waypoints, [ego_x / 100.0, ego_y / 100.0, ego_theta])

    # Ego lateral error and heading error
    WAYPOINTS = WAYPOINTS.reshape(-1, 2)
    diffs = WAYPOINTS - np.array([[ego_x / 100.0, ego_y / 100.0]])
    dists = np.linalg.norm(diffs, axis=1)
    nearest_idx = np.argmin(dists)
    if nearest_idx < len(WAYPOINTS) - 1:
        raceline_vec = WAYPOINTS[nearest_idx + 1] - WAYPOINTS[nearest_idx]
    else:
        raceline_vec = WAYPOINTS[nearest_idx] - WAYPOINTS[nearest_idx - 1]

    raceline_yaw = np.arctan2(raceline_vec[1], raceline_vec[0])
    heading_error = np.arctan2(np.sin(ego_theta - raceline_yaw), np.cos(ego_theta - raceline_yaw))

    # Lateral error = signed distance to raceline
    dx_vec = np.array([ego_x / 100.0, ego_y / 100.0]) - WAYPOINTS[nearest_idx]
    lateral_error = np.cross(raceline_vec / np.linalg.norm(raceline_vec), dx_vec)

    ego_pose = np.array([
        lateral_error,
        heading_error / np.pi
    ])

    # Collision + prev action
    collision = np.array([float(obs['collisions'][ego])])
    prev_action_array = np.array(prev_action)

    # Final input vector
    state_vector = np.concatenate([
        scan_short,
        ego_pose,
        ego_vel,
        rel_pose,
        opp_vel,
        collision,
        wp_ego,
        wp_opp,
        prev_action_array
    ])

    state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=1e3, neginf=-1e3)
    state_vector = np.clip(state_vector, -1e3, 1e3)

    return state_vector.astype(np.float32)

def waypoints_to_ego_frame(waypoints, ego_pose):
    """
    Transforms a set of 2D waypoints from global map frame into the ego (car) frame.

    Args:
        waypoints (np.ndarray): Array of shape (N, 2) in global frame.
        ego_pose (tuple or list): Ego pose (x, y, theta) in global frame.

    Returns:
        np.ndarray: Transformed waypoints in ego frame, shape (N * 2,)
    """
    waypoints =waypoints.reshape(-1, 2)
    ego_x, ego_y, ego_theta = ego_pose

    dx = waypoints[:, 0] - ego_x
    dy = waypoints[:, 1] - ego_y

    cos_yaw = np.cos(-ego_theta)
    sin_yaw = np.sin(-ego_theta)

    x_ego = dx * cos_yaw - dy * sin_yaw
    y_ego = dx * sin_yaw + dy * cos_yaw

    return np.stack((x_ego, y_ego), axis=1).flatten()

'''
def preprocess_observation(obs, WAYPOINTS, prev_action, lidar_max_range=30.0, max_speed=3.0, ego=0):
    #ego = obs['ego_idx']
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
        (obs['linear_vels_x'][opp] - obs['linear_vels_x'][ego]) / max_speed,
        (obs['linear_vels_y'][opp] - obs['linear_vels_y'][ego]) / max_speed,
        (obs['ang_vels_z'][opp] - obs['ang_vels_z'][ego]) / 3.2
    ])

    # Collision (convert to scalar float)
    collision = np.array([float(obs['collisions'][ego])])

    WAYPOINTS = WAYPOINTS / 100.0

    next_waypoints = get_next_waypoints_flat((ego_pose[0],ego_pose[1]), WAYPOINTS)
    next_opp_waypoints = get_next_waypoints_flat((rel_pose[0]+ego_pose[0], rel_pose[1]+ego_pose[1]), WAYPOINTS)

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
        next_opp_waypoints,
        prev_action_array
    ])

    state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=1e3, neginf=-1e3)
    state_vector = np.clip(state_vector, -1e3, 1e3)


    return state_vector.astype(np.float32)
'''
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

    # Opponent car placed 15 waypoints ahead
    rand_dist = np.random.randint(3, 15)
    opp_idx = (ego_idx + rand_dist) % num_waypoints
    wp_prev_opp = waypoints[(opp_idx - 1) % num_waypoints]
    wp_curr_opp = waypoints[opp_idx]
    wp_next_opp = waypoints[(opp_idx + 1) % num_waypoints]
    opp_yaw = np.arctan2(wp_next_opp[1] - wp_prev_opp[1], wp_next_opp[0] - wp_prev_opp[0])

    poses = np.array([
        [wp_curr[0], wp_curr[1], ego_yaw],
        [wp_curr_opp[0], wp_curr_opp[1], opp_yaw]
    ])

    change = np.random.randint(0,2)
    #print(change)
    if change == 1:
        #print(ego_idx, opp_idx)
        ego_idx , opp_idx = opp_idx , ego_idx
        #print(ego_idx, opp_idx)
        #print(poses)
        poses = poses[::-1]

    #poses = np.array([[0.0, 0.0, 0.0], [2.0, 0.5, 0.0]])
    #ego_idx = 0

    return poses, ego_idx, opp_idx

def get_next_waypoints_flat(car_pos, flat_waypoints, num_points=10):
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

def create_unique_ppo_dir(parent_dir, base_name="ppo_multi"):
    i = 1
    while True:
        dir_name = f"{base_name}{i}"
        full_path = os.path.join(parent_dir, dir_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            print(f"Created directory: {full_path}")
            return full_path
        i += 1


if __name__ == '__main__':
    main()
