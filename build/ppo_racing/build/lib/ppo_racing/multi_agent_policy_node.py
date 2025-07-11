import rclpy
from rclpy.node import Node
import numpy as np
import torch
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from .ppo_agent import ActorNetwork  # Your trained policy class
import tf_transformations
import csv
import scipy.interpolate as sp_int


class PolicyNode(Node):
    def __init__(self):
        super().__init__('multi_agent_policy_node')

        self.actor = ActorNetwork(input_dims=163, n_actions=2, alpha=3e-4,
                                  chkpt_dir='/home/mrc/sim_ws/src/ppo_racing/ppo_racing/tmp/ppo_multi29')
        self.actor.load_checkpoint()
        self.actor.eval()

        # ROS I/O
        self.create_subscription(LaserScan, '/scan', self.ego_lidar_callback, 10)
        self.create_subscription(LaserScan, '/opp_scan', self.opp_lidar_callback, 10)
        self.create_subscription(Odometry, '/ego_racecar/odom', self.ego_pose_callback, 10)
        self.create_subscription(Odometry, '/opp_racecar/odom', self.opp_pose_callback, 10)
        self.control_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.opp_control_pub = self.create_publisher(AckermannDriveStamped, '/opp_drive', 10)

        # State tracking
        self.ego_pose = None
        self.opp_pose = None
        self.ego_scan_short = None
        self.opp_scan_short = None
        self.ego_last_action = np.array([0.0, 0.0])
        self.opp_last_action = np.array([0.0, 0.0])
        self.ego_collision = np.array([0.])
        self.opp_collision = np.array([0.])
        self.beta = 0.9

        waypoints = np.loadtxt('/home/mrc/sim_ws/src/ppo_racing/ppo_racing/Spielberg_waypoints.csv', delimiter=',')
        self.WAYPOINTS = waypoints[::5, :2].flatten() / 100.0

    def ego_lidar_callback(self, msg):
        self.ego_scan_short = np.array(msg.ranges[::10]) / 30.0

    def opp_lidar_callback(self, msg):
        self.opp_scan_short = np.array(msg.ranges[::10]) / 30.0

    def ego_pose_callback(self, msg):
        self.ego_pose = self.extract_pose(msg)
        self.step_policy()

    def opp_pose_callback(self, msg):
        self.opp_pose = self.extract_pose(msg)
        self.step_policy()

    def extract_pose(self, msg):
        ori = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        return np.array([
            msg.pose.pose.position.x / 100.0,
            msg.pose.pose.position.y / 100.0,
            yaw / np.pi,
            msg.twist.twist.linear.x / 3.0,
            msg.twist.twist.linear.y / 3.0,
            msg.twist.twist.angular.z / np.pi
        ])

    def step_policy(self):
        if any(x is None for x in (self.ego_pose, self.opp_pose, self.ego_scan_short, self.opp_scan_short)):
            return

        # Get next waypoints
        ego_way = self.get_waypoints(self.ego_pose[:2], self.WAYPOINTS)
        opp_way = self.get_waypoints(self.opp_pose[:2], self.WAYPOINTS)

        # --- Ego Observation ---
        opp_rel_to_ego = self.get_relative_pose(self.opp_pose, self.ego_pose)
        ego_obs = np.concatenate((
            self.ego_scan_short,
            self.ego_pose[:3], self.ego_pose[3:],
            opp_rel_to_ego[:3], self.opp_pose[3:],
            self.ego_collision,
            ego_way, opp_way,
            self.ego_last_action
        ))
        ego_action, self.ego_last_action = self.generate_action(ego_obs, self.ego_last_action)
        self.publish_drive(self.control_pub, ego_action)

        # --- Opponent Observation ---
        ego_rel_to_opp = self.get_relative_pose(self.ego_pose, self.opp_pose)
        opp_obs = np.concatenate((
            self.opp_scan_short,
            self.opp_pose[:3], self.opp_pose[3:],
            ego_rel_to_opp[:3], self.ego_pose[3:],
            self.opp_collision,
            opp_way, ego_way,
            self.opp_last_action
        ))
        opp_action, self.opp_last_action = self.generate_action(opp_obs, self.opp_last_action)
        self.publish_drive(self.opp_control_pub, opp_action)

    def get_relative_pose(self, target, reference):
        dx, dy = target[0] - reference[0], target[1] - reference[1]
        theta = reference[2] * np.pi
        rel_x = np.cos(-theta) * dx - np.sin(-theta) * dy
        rel_y = np.sin(-theta) * dx + np.cos(-theta) * dy
        return np.array([
            rel_x,
            rel_y,
            (target[2] - reference[2]),  # heading diff
            target[3], target[4], target[5]
        ])

    def generate_action(self, obs, last_action):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.actor.device)
        with torch.no_grad():
            dist = self.actor(obs_tensor)
            raw_action = dist.sample().cpu().numpy().flatten()
            smoothed = self.beta * last_action + (1 - self.beta) * raw_action
            clipped = [np.clip(smoothed[0], -0.34, 0.34), np.clip(smoothed[1], 0.0, 3.0)]
            return clipped, smoothed

    def publish_drive(self, publisher, action):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(action[0])
        msg.drive.speed = float(action[1])
        publisher.publish(msg)

    def get_waypoints(self, pose, flat_waypoints, num_points=10):
        waypoints = np.array(flat_waypoints).reshape(-1, 2)
        dists = np.linalg.norm(waypoints - np.array(pose), axis=1)
        closest_idx = np.argmin(dists)
        total = waypoints.shape[0]
        indices = [(closest_idx + i) % total for i in range(num_points)]
        return waypoints[indices].flatten()


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()