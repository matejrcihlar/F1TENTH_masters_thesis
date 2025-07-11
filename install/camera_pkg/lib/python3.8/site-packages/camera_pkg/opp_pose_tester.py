#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, PoseStamped
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan

import numpy as np
from scipy.spatial import cKDTree
import tf_transformations

from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
import std_msgs.msg
import struct



class LidarOpponentTracker(Node):
    def __init__(self):
        super().__init__('opp_pose_tester')

        self.publisher_ = self.create_publisher(Odometry, '/opp_pose', 10)
        self.cloud_pub = self.create_publisher(PointCloud2, '/feature_points', 10)
        self.radius_pub = self.create_publisher(Marker, '/extraction_radius', 1)
        self.window_pub = self.create_publisher(Marker, '/search_window', 1)

        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 1)

        self.scan = []
        self.angle_min = 0.0
        self.angle_increment = 0.0
        self.feature_points = []
        self.has_initial_pose = False
        self.last_pose = [0.0, 0.0, 0.0]

        self.framerate = 30.0
        self.max_speed = 3.0
        self.extraction_radius = 0.5
        self.search_window = [(-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6)]

        self.timer = self.create_timer(1.0 / self.framerate, self.timer_callback)

    def goal_callback(self, msg):
        self.last_pose = [msg.pose.position.x, msg.pose.position.y,
                          tf_transformations.euler_from_quaternion([
                              msg.pose.orientation.x,
                              msg.pose.orientation.y,
                              msg.pose.orientation.z,
                              msg.pose.orientation.w
                          ])[2]]
        self.has_initial_pose = True
        self.feature_points = self.extract_opponent_points()
        self.get_logger().info(f"{self.feature_points}")
        self.get_logger().info("Initial pose received and feature points extracted.")
        self.publish_feature_markers()

    def lidar_callback(self, msg):
        self.scan = msg.ranges
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment

    def extract_opponent_points(self):
        if not self.scan:
            return []

        angles = self.angle_min + np.arange(len(self.scan)) * self.angle_increment
        xs = np.array(self.scan) * np.cos(angles)
        ys = np.array(self.scan) * np.sin(angles)

        px, py, _ = self.last_pose
        dists = np.sqrt((xs - px)**2 + (ys - py)**2)
        mask = dists < self.extraction_radius
        return np.vstack((xs[mask], ys[mask])).T

    def match_scan_safe(self, resolution=0.01, max_pose_delta=1, max_yaw_delta=2):
        if not self.scan or len(self.feature_points) == 0:
            return []

        px, py, pyaw = self.last_pose

        angles = self.angle_min + np.arange(len(self.scan)) * self.angle_increment
        xs = np.array(self.scan) * np.cos(angles)
        ys = np.array(self.scan) * np.sin(angles)
        scan_points = np.vstack((xs, ys)).T

        scan_tree = cKDTree(scan_points)

        best_score = float('inf')
        best_pose = self.last_pose

        dx_range = np.arange(*self.search_window[0], resolution)
        dy_range = np.arange(*self.search_window[1], resolution)
        dyaw_range = np.arange(*self.search_window[2], resolution / 3)

        for dx in dx_range:
            for dy in dy_range:
                for dyaw in dyaw_range:
                    cx = px + dx
                    cy = py + dy
                    cyaw = pyaw + dyaw

                    cos_yaw = np.cos(cyaw)
                    sin_yaw = np.sin(cyaw)
                    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
                    transformed = (rot @ self.feature_points.T).T
                    transformed += np.array([cx, cy])

                    dists, _ = scan_tree.query(transformed, k=1)
                    score = np.median(dists)

                    if score < best_score:
                        best_score = score
                        best_pose = (cx, cy, cyaw)

        dx = best_pose[0] - px
        dy = best_pose[1] - py
        dyaw = abs((best_pose[2] - pyaw + np.pi) % (2 * np.pi) - np.pi)
        self.get_logger().info(f"{best_pose}")

        if np.sqrt(dx**2 + dy**2) > max_pose_delta or dyaw > max_yaw_delta or best_score > 3.0:
            return []

        return best_pose

    def timer_callback(self):
        if not self.has_initial_pose or not self.scan:
            return

        new_pose = self.match_scan_safe()
        if not new_pose:
            self.get_logger().info("No new pose found")
            return

        x, y, yaw = new_pose
        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw)

        odom_msg = Odometry()
        odom_msg.header = Header()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = '/ego_racecar/laser'
        odom_msg.child_frame_id = 'opp_pose'
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        odom_msg.twist.twist.linear.x = (x - self.last_pose[0]) * self.framerate / self.max_speed
        odom_msg.twist.twist.linear.y = (y - self.last_pose[1]) * self.framerate / self.max_speed
        odom_msg.twist.twist.angular.z = (yaw - self.last_pose[2]) * self.framerate / 3.2

        self.last_pose = [x, y, yaw]
        self.publisher_.publish(odom_msg)


    def publish_feature_markers(self):
        # Publish feature point cloud

        # Publish extraction radius marker
        px, py, _ = self.last_pose
        marker = Marker()
        marker.header.frame_id = "/ego_racecar/laser"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "extraction_radius"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = px
        marker.pose.position.y = py
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.extraction_radius * 2
        marker.scale.y = self.extraction_radius * 2
        marker.scale.z = 0.01
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 0.4
        self.radius_pub.publish(marker)

        # Publish search window cube
        win_marker = Marker()
        win_marker.header.frame_id = "/ego_racecar/laser"
        win_marker.header.stamp = self.get_clock().now().to_msg()
        win_marker.ns = "search_window"
        win_marker.id = 1
        win_marker.type = Marker.CUBE
        win_marker.action = Marker.ADD
        center_x = px + (self.search_window[0][0] + self.search_window[0][1]) / 2
        center_y = py + (self.search_window[1][0] + self.search_window[1][1]) / 2
        win_marker.pose.position.x = center_x
        win_marker.pose.position.y = center_y
        win_marker.pose.position.z = 0.0
        win_marker.pose.orientation.w = 1.0
        win_marker.scale.x = abs(self.search_window[0][1] - self.search_window[0][0])
        win_marker.scale.y = abs(self.search_window[1][1] - self.search_window[1][0])
        win_marker.scale.z = 0.01
        win_marker.color.r = 0.0
        win_marker.color.g = 0.0
        win_marker.color.b = 1.0
        win_marker.color.a = 0.3
        self.window_pub.publish(win_marker)

def main(args=None):
    rclpy.init(args=args)
    node = LidarOpponentTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()