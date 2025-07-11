import numpy as np


waypoints = np.loadtxt('/home/mrc/sim_ws/src/ppo_racing/ppo_racing/Spielberg_waypoints.csv', delimiter=',')

# Keep only the first two columns
waypoints = waypoints[::5,:]
WAYPOINTS = waypoints[:, :2].flatten()/100.0

def get_waypoints(pose, flat_waypoints, num_points=10):

        waypoints = np.array(flat_waypoints).reshape(-1, 2)

        dists = np.linalg.norm(waypoints - np.array(pose), axis=1)
        closest_idx = np.argmin(dists)

        # Select next waypoints with wrap-around
        total = waypoints.shape[0]
        indices = [(closest_idx + i) % total for i in range(num_points)]
        next_waypoints = waypoints[indices]

        # Flatten and return
        return np.array(next_waypoints)

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

        return np.stack((x_ego, y_ego), axis=1)


ego_pose = np.array([0.0,0.0,-0.0])
way = get_waypoints([ego_pose[0]/100.0,ego_pose[1]/100.0], WAYPOINTS)
print(f"way:{way*100.0}")
way.flatten()
trans_way = waypoints_to_ego_frame(way, [ego_pose[0]/100.0,ego_pose[1]/100.0, ego_pose[2]])
print(f"trnasformed{trans_way}")