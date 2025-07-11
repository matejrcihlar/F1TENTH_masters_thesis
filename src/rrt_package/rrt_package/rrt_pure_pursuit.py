#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import scipy.interpolate as sp_int
import csv
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
import tf_transformations  # Handles quaternion → euler
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64MultiArray


class RRTPurePursuit(Node):
    def __init__(self):
        super().__init__('rrt_pure_pursuit_node')

        # ROS Topics
        odom_topic = '/ego_racecar/odom'
        drive_topic = '/drive'
        self.pose_subscription = self.create_subscription(Odometry, odom_topic, self.pose_callback, 10)
        self.waypoint_subscription = self.create_subscription(Float64MultiArray, '/path',self.path_callback, 10)
        self.acker_publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        #self.marker_pub = self.create_publisher(Marker, '/waypoints_marker', 10)
        

        # Pure Pursuit parameters
        self.L = 1.0  # Lookahead distance
        self.K = 0.5  # Curvature gain
        self.waypoints = []

        # Load and interpolate waypoints
          # List of (x, y) tuples
        
        #self.publish_waypoint_markers()

    def pose_callback(self, pose_msg):
        # Get current position
        pos = pose_msg.pose.pose.position
        x, y = pos.x, pos.y

        # Find first waypoint at least L distance away
        if self.waypoints == []:
            #self.get_logger().warn("No waypoints message received yet, skipping pose callback.")
            return
            

        # Get yaw from quaternion
        ori = pose_msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])

        waypoint , dist_min = self.find_best_waypoint(x, y, yaw)


        # Transform waypoint into vehicle frame
        dx = waypoint[0] - x
        dy = waypoint[1] - y
        x_car = np.cos(-yaw) * dx - np.sin(-yaw) * dy
        y_car = np.sin(-yaw) * dx + np.cos(-yaw) * dy

        # Compute steering angle
        curvature = 2 * y_car / (self.L ** 2)
        steering_angle = np.clip(self.K * curvature, -0.34, 0.34)
        velocity = 0.5

        # Publish Ackermann drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = steering_angle
        self.acker_publisher.publish(drive_msg)
        #self.get_logger().info("done pose")


    def find_best_waypoint(self, x, y, yaw):
        min_diff = float('inf')
        best_wp = self.waypoints[0]
        dist_min = self.L
    
        for wp in self.waypoints:
            dx = wp[0] - x
            dy = wp[1] - y
            distance = np.hypot(dx, dy)

            # Check if waypoint is in front of the car
            heading_vec = np.array([np.cos(yaw), np.sin(yaw)])
            wp_vec = np.array([dx, dy])
            dot = np.dot(heading_vec, wp_vec)

            if dot > 0:  # In front
                diff = abs(distance - self.L)
                if diff < min_diff:
                    min_diff = diff
                    best_wp = wp
                    dist_min = distance

        return best_wp, dist_min


    def path_callback(self, path_msg):

        data = path_msg.data
        points = [(data[i], data[i+1]) for i in range(0, len(data), 2)]
        self.waypoints = points
        #self.get_logger().info("Got waypoints")

    def publish_waypoint_markers(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "waypoints"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2  # width of the point
        marker.scale.y = 0.2  # height
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        for wp in self.waypoints:
            p = Point()
            p.x = wp[0]
            p.y = wp[1]
            p.z = 0.0
            marker.points.append(p)

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    rrt_pure_pursuit_node = RRTPurePursuit()
    rclpy.spin(rrt_pure_pursuit_node)
    rrt_pure_pursuit_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()