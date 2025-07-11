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
        super().__init__('policy_node')

        # Load model
        self.actor = ActorNetwork(input_dims=183, n_actions=2, alpha=3e-4, chkpt_dir = '/home/mrc/sim_ws/src/ppo_racing/ppo_racing/tmp/ppo_1-dobar')
        self.actor.load_checkpoint()
        self.actor.eval()

        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)

        self.pose_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.pose_callback, 10)
        
        self.opp_sub = self.create_subscription(Odometry, '/opp_racecar/odom', self.opp_pose_callback, 10)

        self.control_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)
        
        self.opp_control_pub = self.create_publisher(AckermannDriveStamped, '/opp_drive', 10)

        self.lidar = None
        self.scan_short = None
        self.pose = None
        self.opp_pose = None
        self.collisions = np.array([0.])
        self.last_action = np.array([0.0,0.0])
        self.beta = 0.9

        WAYPOINTS = []
        
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
        self.pose = np.array([msg.pose.pose.position.x / 100.0, 
                              msg.pose.pose.position.y / 100.0,
                              yaw / np.pi,
                              msg.twist.twist.linear.x / 3.0,
                              msg.twist.twist.linear.y / 3.0,
                              msg.twist.twist.angular.z / np.pi])  # Example
        
        if self.lidar is None or self.pose is None or self.opp_pose is None:
            return
        
        self.curr_way = self.get_waypoints((self.pose[0], self.pose[1]), self.WAYPOINTS)
        #self.curr_opp_way = self.get_waypoints((self.opp_pose[0],self.opp_pose[1]), self.WAYPOINTS)
        
        obs = np.concatenate((self.scan_short, self.pose, self.opp_pose, self.collisions, self.curr_way, self.last_action))

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.actor.device)
        with torch.no_grad():
            
            dist = self.actor(obs_tensor)
            action = dist.sample()  # Sample from the distribution
            action = action.cpu().detach().numpy().flatten()
            

            action_sized = np.array([np.clip(action[0], -0.34, 0.34), np.clip(action[1], 0.0, 3.0)])
            smoothed_action = self.beta * self.last_action + (1 - self.beta) * action_sized
            self.last_action = smoothed_action
        

        # Publish action
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(smoothed_action[0])   # Steering
        drive_msg.drive.speed = float(smoothed_action[1])  # Velocity
        self.control_pub.publish(drive_msg)
        #print("published drive")
        opp_drive_msg = AckermannDriveStamped()
        self.opp_control_pub.publish(opp_drive_msg)



    def opp_pose_callback(self, msg):
        if self.pose is None or msg is None:
            return
        opp_ori = msg.pose.pose.orientation
        _, _ ,opp_yaw = tf_transformations.euler_from_quaternion([opp_ori.x, opp_ori.y, opp_ori.z, opp_ori.w])
        self.opp_pose = np.array([(msg.pose.pose.position.x / 100.0 - self.pose[0]),  
                              (msg.pose.pose.position.y / 100.0 - self.pose[1]), 
                              (opp_yaw / np.pi - self.pose[2]) ,
                              (msg.twist.twist.linear.x / 3.0 - self.pose[3]),
                              (msg.twist.twist.linear.y / 3.0 - self.pose[4]),
                              (msg.twist.twist.angular.z / np.pi - self.pose[5])])

    def get_waypoints(self, pose, flat_waypoints, num_points=30):

        waypoints = np.array(flat_waypoints).reshape(-1, 2)

        dists = np.linalg.norm(waypoints - np.array(pose), axis=1)
        closest_idx = np.argmin(dists)

        # Select next waypoints with wrap-around
        total = waypoints.shape[0]
        indices = [(closest_idx + i) % total for i in range(num_points)]
        next_waypoints = waypoints[indices]

        # Flatten and return
        return np.array(next_waypoints.flatten())


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
