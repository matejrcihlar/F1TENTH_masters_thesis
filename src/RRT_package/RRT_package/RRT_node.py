#!/usr/bin/env python3


import rclpy
from rclpy.node import Node
import yaml
from PIL import Image
import numpy as np
import os
import math
import random
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import tf_transformations
import csv
class RRT(Node):
    
    def __init__(self):
        super().__init__('RRT_node')

        # ROS Topics
        odom_topic = '/ego_racecar/odom'
        drive_topic = '/drive'
        scan_topic = '/scan'
        self.pose_subscription = self.create_subscription(Odometry, odom_topic, self.pose_callback, 10)
        self.scan_subscription = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.acker_publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        # Get waypoints
        self.waypoints = []
        with open('/home/mrc/sim_ws/src/rrt_package/waypoints.csv', mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.waypoints.append([float(i) for i in row])
            
        # Load map yaml file
        map_yaml_path = '/home/mrc/sim_ws/src/rrt_package/maps/levine.yaml'
        with open(map_yaml_path, 'r') as f:
            map_metadata = yaml.safe_load(f)

        self.get_logger().info(f"Loaded map metadata: {map_metadata}")

        # Load the PGM image
        image_path = os.path.join(os.path.dirname(map_yaml_path), map_metadata['image'])
        map_image = Image.open(image_path)
        self.map = np.array(map_image)
        self.map_array = self.map
        self.get_logger().info(f"Map image shape: {self.map_array.shape}")

        self.resolution = map_metadata['resolution']
        self.origin_x, self.origin_y, self.origin_yaw = map_metadata['origin']
        self.negate = map_metadata['negate']
        self.occupied_thresh = map_metadata['occupied_thresh']
        self.free_thresh = map_metadata['free_thresh']
        self.height, self.width = self.map_array.shape

        self.L = 0.5
        self.Lookahead = 5
        self.short_lookahead = 1 
        self.K = 0.5

        self.V = []
        self.E = []

        self.get_logger().info("RRT Node initialized.")


    def nearest(self, x_rand):
        dist_min = self.height**2 + self.width**2
        for v in range(len(self.V)):
            dist = (x_rand[0]-self.V[v][0])**2 + (x_rand[1]-self.V[v][1])**2
            if dist < dist_min:
                dist_min = dist
                x_nearest = self.V[v]
                parent_index = v
        return x_nearest, parent_index


    def expand(self, x_nearest, x_rand):
        

        dx = x_rand[0] - x_nearest[0]
        dy = x_rand[1] - x_nearest[1]

        length = np.hypot(dx, dy)
    
        unit_dx = dx / length
        unit_dy = dy / length

        x_new = [0,0]

        x_new[0] = int(x_nearest[0] + unit_dx * self.L/self.resolution)
        x_new[1] = int(x_nearest[1] + unit_dy * self.L/self.resolution)

        return x_new


    def check_collision(self, x_nearest, x_new):

        dx = abs(x_new[0] - x_nearest[0])
        dy = abs(x_new[1] - x_nearest[1])
        sx = 1 if x_nearest[0] < x_new[0] else -1
        sy = 1 if x_nearest[1] < x_new[1] else -1
        err = dx - dy

        x, y = x_nearest[0], x_nearest[1]

        # Continue until x and y both reach x1, y1
        while (x != x_new[0]) or (y != x_new[1]):

            if self.map_array[y, x] == 0:
                return True  # collision detected

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return False  # clear path


    def random_free(self):
        
        free_indices = np.argwhere(self.map_array >= self.free_thresh)
        selected_cell = random.choice(free_indices)
        selected_cell.reverse()

        return selected_cell


    def find_best_waypoint(self, x, y, yaw, lookahead, waypoints):
        min_diff = float('inf')
        best_wp = waypoints[0]
    
        for wp in waypoints:
            dx = wp[0] - x
            dy = wp[1] - y
            distance = np.hypot(dx, dy)

            # Check if waypoint is in front of the car
            heading_vec = np.array([np.cos(yaw), np.sin(yaw)])
            wp_vec = np.array([dx, dy])
            dot = np.dot(heading_vec, wp_vec)

            if dot > 0:  # In front
                diff = abs(distance - lookahead)
                if diff < min_diff:
                    min_diff = diff
                    best_wp = wp

        return best_wp


    def extract_path(self, goal_index):
        path = []
        current_index = goal_index
        while current_index is not None:
            path.append(self.V[current_index])
            current_index = self.E[current_index]
        path.reverse()  # from start to goal
        return path


    def pose_callback(self, pose_msg):

        self.V = [[msg.pose.pose.position.x, msg.pose.pose.position.y]]

        pos = pose_msg.pose.pose.position
        x, y = pos.x, pos.y

        # Find first waypoint at least L distance away
        

        # Get yaw from quaternion
        ori = pose_msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])

        self.map_array = self.scan_on_map(x, y, yaw)

        waypoint = self.find_best_waypoint(x, y, yaw, self.Lookahead, self.waypoints)


        # Transform waypoint into vehicle frame
        dx = waypoint[0] - x
        dy = waypoint[1] - y
        x_car = np.cos(-yaw) * dx - np.sin(-yaw) * dy
        y_car = np.sin(-yaw) * dx + np.cos(-yaw) * dy

        goal_point = [x_car, y_car]


        for k in range(150):

            x_rand = self.random_free()
            x_nearest, parent_index = self.nearest(x_rand)
            x_new = self.expand(x_nearest, x_rand)
            x_collision = self.check_collision(x_nearest, x_new)
            if not x_collision:
                self.V = self.V + [x_new]
                self.E = self.E + [parent_index]

        nearest_goal_point, goal_point_index = self.nearest(goal_point)
        path = self.extract_path(goal_point_index)
        close_waypoint = self.find_best_waypoint(x, y, yaw, self.short_lookahead, path)

        # Transform waypoint into vehicle frame
        dx = close_waypoint[0] - x
        dy = close_waypoint[1] - y
        x_car = np.cos(-yaw) * dx - np.sin(-yaw) * dy
        y_car = np.sin(-yaw) * dx + np.cos(-yaw) * dy

        # Compute steering angle
        curvature = 2 * y_car / (self.short_lookahead ** 2)
        steering_angle = np.clip(self.K * curvature, -0.34, 0.34)
        velocity = 3 - 0.5 * abs(steering_angle)

        # Publish Ackermann drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = steering_angle
        self.acker_publisher.publish(drive_msg)


    def scan_callback(self, scan_msg):
        self.scan_msg = scan_msg


    def scan_on_map(self, x, y, yaw):
        
        robot_x, robot_y, robot_theta = x, y, yaw

        angle = self.scan_msg.angle_min

        scan_map_array = self.map

        for r in self.scan_msg.ranges:
            if self.scan_msg.range_min < r < self.scan_msg.range_max:
                # Convert scan range and angle to local coordinates
                local_x = r * math.cos(angle)
                local_y = r * math.sin(angle)

                # Transform to world frame
                world_x = robot_x + local_x * math.cos(robot_theta) - local_y * math.sin(robot_theta)
                world_y = robot_y + local_x * math.sin(robot_theta) + local_y * math.cos(robot_theta)

                # Convert world frame to map array indices
                map_x = int((world_x - self.origin_x) / self.resolution)
                map_y = int(self.height - (world_y - self.origin_y) / self.resolution)

                # Check if indices are within map bounds
                if 0 <= map_x < self.width and 0 <= map_y < self.height:
                    scan_map_array[map_y, map_x] = 0  # set cell as occupied

            angle += self.scan_msg.angle_increment

        return scan_map_array

def main(args=None):
    rclpy.init(args=args)
    
    RRT_node = RRT()
    rclpy.spin(RRT_node)
    RRT_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


