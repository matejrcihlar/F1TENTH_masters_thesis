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
        super().__init__('policy_node_local_obs')

        # Load model
        self.actor = ActorNetwork(input_dims=162, n_actions=2, alpha=3e-4, chkpt_dir = '/home/mrc/sim_ws/src/ppo_racing/ppo_racing/tmp/ppo23_dobar')
        self.actor.load_checkpoint(weights_only=True)
        self.get_logger().info("loaded model")
        self.actor.eval()

        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)

        self.pose_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.pose_callback, 10)
        
        self.opp_sub = self.create_subscription(Odometry, '/opp_racecar/odom', self.opp_pose_callback, 10)

        self.control_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)
        
        #self.opp_control_pub = self.create_publisher(AckermannDriveStamped, '/opp_drive', 10)

        self.lidar = None
        self.scan_short = None
        self.pose = None
        self.vel = None
        self.opp_pose = None
        self.opp_vel = None
        self.collisions = np.array([0.])
        self.last_action = np.array([0.0,0.0])
        self.beta = 0.9
        self.max_speed = 3.0
        
        WAYPOINTS = []
        '''
        with open('/home/mrc/sim_ws/src/f1tenth_lab6_template/waypoints.csv', mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                WAYPOINTS.append([float(i) for i in row])

        
        WAYPOINTS= np.transpose(WAYPOINTS)
        
        tck, _ = sp_int.splprep(WAYPOINTS, s=0)
        u_fine = np.linspace(0, 1, 100)
        interpolated = sp_int.splev(u_fine, tck)
        x_vals, y_vals = interpolated
        self.WAYPOINTS = np.vstack((x_vals, y_vals)).T.flatten()
        self.WAYPOINTS = self.WAYPOINTS / 100.0
        '''
        
        waypoints = np.loadtxt('/home/mrc/sim_ws/src/ppo_racing/ppo_racing/Spielberg_waypoints.csv', delimiter=',')

        # Keep only the first two columns
        waypoints = waypoints[::5,:]
        self.WAYPOINTS = waypoints[:, :2].flatten() / 100.0
        #print(self.WAYPOINTS)
        '''
        waypoints = np.loadtxt('/home/mrc/sim_ws/src/ppo_racing/ppo_racing/raceline.csv', delimiter=',')
        
        # Keep only the first two columns
        self.WAYPOINTS = waypoints[:, :2].flatten()/100.0
        '''
        
        
        '''
        waypoints = np.loadtxt('/home/mrc/sim_ws/src/f1tenth_lab6_template/waypoints_crta_krug.csv', delimiter=',')
        waypoints = np.array([[y*1.3 + 2.7,x*1.45 + 0.4] for [x,y] in waypoints])
        self.WAYPOINTS = waypoints.flatten()/100.0
        '''


    def lidar_callback(self, msg):
        self.lidar = msg.ranges[:1080] # Assuming 1080 LIDAR points
        self.scan_short = []
        for k in range(len(self.lidar)):
            if k%10==0:
                self.scan_short.append(self.lidar[k])
        self.scan_short = np.array(self.scan_short) / 30.0


    def pose_callback(self, msg):
        ori = msg.pose.pose.orientation
        _, _ ,yaw = tf_transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.pose = np.array([msg.pose.pose.position.x, 
                              msg.pose.pose.position.y,
                              yaw])
        self.vel = np.array([msg.twist.twist.linear.x,
                              msg.twist.twist.linear.y,
                              msg.twist.twist.angular.z])  # Example
        
        if self.lidar is None or self.pose is None or self.opp_pose is None:
            return
        
        # Opponent pose relative in map frame
        dx = self.opp_pose[0] - self.pose[0]
        dy = self.opp_pose[1] - self.pose[1]
        dtheta = self.opp_pose[2] - self.pose[2]

        # Rotate to ego frame
        cos_yaw = np.cos(-self.pose[2])
        sin_yaw = np.sin(-self.pose[2])
        rel_x = dx * cos_yaw - dy * sin_yaw
        rel_y = dx * sin_yaw + dy * cos_yaw
        rel_theta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

        rel_pose = np.array([rel_x / 100.0, rel_y / 100.0, rel_theta / np.pi])

        # Transform opponent velocity into ego frame
        vx_rel = self.opp_vel[0] * cos_yaw - self.opp_vel[1] * sin_yaw
        vy_rel = self.opp_vel[0] * sin_yaw + self.opp_vel[1] * cos_yaw

        opp_vel = np.array([
            (vx_rel - self.vel[0]) / self.max_speed,
            (vy_rel - self.vel[1]) / self.max_speed,
            (self.opp_vel[2] - self.vel[2]) / np.pi
        ])

        vel = np.array([self.vel[0]/100.0, self.vel[1]/100.0, self.vel[2]/np.pi])
        curr_way = self.get_waypoints((self.pose[0]/100.0, self.pose[1]/100.0), self.WAYPOINTS)
        
        curr_way = self.waypoints_to_ego_frame(curr_way, [self.pose[0]/100.0, self.pose[1]/100.0, self.pose[2]])
        #self.get_logger().info(f"{curr_way}")
        curr_opp_way = self.get_waypoints((self.opp_pose[0]/100.0,self.opp_pose[1]/100.0), self.WAYPOINTS)
        curr_way = self.waypoints_to_ego_frame(curr_opp_way, [self.pose[0]/100.0, self.pose[1]/100.0, self.pose[2]])

        WAYPOINTS = self.WAYPOINTS.reshape(-1, 2)
        diffs = WAYPOINTS - np.array([[self.pose[0] / 100.0, self.pose[1] / 100.0]])
        dists = np.linalg.norm(diffs, axis=1)
        nearest_idx = np.argmin(dists)
        if nearest_idx < len(WAYPOINTS) - 1:
            raceline_vec = WAYPOINTS[nearest_idx + 1] - WAYPOINTS[nearest_idx]
        else:
            raceline_vec = WAYPOINTS[nearest_idx] - WAYPOINTS[nearest_idx - 1]

        raceline_yaw = np.arctan2(raceline_vec[1], raceline_vec[0])
        heading_error = np.arctan2(np.sin(self.pose[2] - raceline_yaw), np.cos(self.pose[2] - raceline_yaw))

        # Lateral error = signed distance to raceline
        dx_vec = np.array([self.pose[0] / 100.0, self.pose[1] / 100.0]) - WAYPOINTS[nearest_idx]
        lateral_error = np.cross(raceline_vec / np.linalg.norm(raceline_vec), dx_vec)

        pose = np.array([
            lateral_error,
            heading_error / np.pi
        ])



        obs = np.concatenate((self.scan_short, pose, vel, rel_pose, opp_vel, self.collisions, curr_way, curr_opp_way, self.last_action))
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.actor.device)
        with torch.no_grad():
            
            dist = self.actor(obs_tensor)
            action = dist.sample()  # Sample from the distribution
            action = action.cpu().detach().numpy().flatten()
            

            action_sized = np.array([np.clip(action[0], -0.34, 0.34), np.clip(action[1], 0.0, self.max_speed)])
            smoothed_action = self.beta * self.last_action + (1 - self.beta) * action_sized
            self.last_action = smoothed_action
        

        # Publish action
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(smoothed_action[0])   # Steering
        drive_msg.drive.speed = float(smoothed_action[1])  # Velocity
        self.control_pub.publish(drive_msg)
        #print("published drive")
        opp_drive_msg = AckermannDriveStamped()
        #self.opp_control_pub.publish(opp_drive_msg)



    def opp_pose_callback(self, msg):
        if self.pose is None or msg is None:
            return
        opp_ori = msg.pose.pose.orientation
        _, _ ,opp_yaw = tf_transformations.euler_from_quaternion([opp_ori.x, opp_ori.y, opp_ori.z, opp_ori.w])
        self.opp_pose = np.array([(msg.pose.pose.position.x),  
                              (msg.pose.pose.position.y), 
                              (opp_yaw)])
        self.opp_vel = np.array([(msg.twist.twist.linear.x),
                              (msg.twist.twist.linear.y),
                              (msg.twist.twist.angular.z)])

    def get_waypoints(self, pose, flat_waypoints, num_points=10):

        waypoints = np.array(flat_waypoints).reshape(-1, 2)

        dists = np.linalg.norm(waypoints - np.array(pose), axis=1)
        closest_idx = np.argmin(dists)

        # Select next waypoints with wrap-around
        total = waypoints.shape[0]
        indices = [(closest_idx + i) % total for i in range(num_points)]
        next_waypoints = waypoints[indices]

        # Flatten and return
        return np.array(next_waypoints.flatten())

    def waypoints_to_ego_frame(self, waypoints, ego_pose):
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

def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
