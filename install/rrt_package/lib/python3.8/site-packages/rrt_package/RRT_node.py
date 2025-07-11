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
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Float64MultiArray
from geometry_msgs.msg import Point
from scipy.spatial import KDTree
import time
from scipy.ndimage import binary_dilation



class RRT(Node):
    
    def __init__(self):
        super().__init__('RRT_node')

        # ROS Topics
        odom_topic = '/ego_racecar/odom'
        drive_topic = '/drive'
        scan_topic = '/scan'
        path_topic = '/path'
        self.pose_subscription = self.create_subscription(Odometry, odom_topic, self.pose_callback, 1)
        self.scan_subscription = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        #self.acker_publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.path_publisher = self.create_publisher(Float64MultiArray, path_topic, 10)
        self.tree_pub = self.create_publisher(MarkerArray, '/rrt_tree', 10)


        # Get waypoints
        self.waypoints = []
        with open('/home/mrc/sim_ws/src/rrt_package/waypoints.csv', mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.waypoints.append([float(i) for i in row])
        '''        
        waypoints = np.loadtxt('/home/mrc/sim_ws/src/ppo_racing/ppo_racing/Spielberg_waypoints.csv', delimiter=',')
        # Keep only the first two columns
        waypoints = waypoints[::5,:]
        waypoints = waypoints[:, :2]
        self.waypoints = [tuple(row) for row in waypoints] 
        '''
        #self.get_logger().info(f"Loaded waypoint data: {self.waypoints}")
        # Load map yaml file
        map_yaml_path = '/home/mrc/sim_ws/src/rrt_package/maps/levine_blocked.yaml'
        with open(map_yaml_path, 'r') as f:
            map_metadata = yaml.safe_load(f)

        self.get_logger().info(f"Loaded map metadata: {map_metadata}")

        # Load the PGM image
        image_path = os.path.join(os.path.dirname(map_yaml_path), map_metadata['image'])
        map_image = Image.open(image_path)
        self.map = np.array(map_image)
        
        


        self.resolution = map_metadata['resolution']
        self.origin_x, self.origin_y, self.origin_yaw = map_metadata['origin']
        self.negate = map_metadata['negate']
        self.occupied_thresh = map_metadata['occupied_thresh']
        self.free_thresh = map_metadata['free_thresh']
        
        self.inflation_radius = int(0.2/self.resolution)
        self.map_array = self.inflate_obstacles()
        self.get_logger().info(f"Map image shape: {self.map_array.shape}")
        self.occupied_cells = set(zip(np.where(self.map_array == 0)[1], np.where(self.map_array == 0)[0]))
        self.height, self.width = self.map_array.shape
        self.free_indices = np.argwhere(self.map_array >= self.free_thresh)

        self.L = 0.6
        self.Lookahead = 5
        self.short_lookahead = 1.5 
        self.K = 0.5

        self.scan_msg = None
        self.scan = {}

        self.V = [[0,0]]
        self.E = []
        self.kdtree = KDTree(np.array(self.V))
        self.paused = False

        self.get_logger().info("RRT Node initialized.")


    def nearest(self, x_rand):
        dist, index = self.kdtree.query(x_rand)
        x_nearest = self.V[index]
        
        return x_nearest, index


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

        x = int(x)  # Convert x to integer
        y = int(y)

        # Continue until x and y both reach x1, y1
        while (x != x_new[0]) or (y != x_new[1]):

            if (x, y) in self.occupied_cells or (x, y) in self.scan:
                return True  # collision detected

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        

        return False  # clear path


    def world_to_map_coords(self, x, y):
        map_x = int((x - self.origin_x) / self.resolution)
        map_y = int(self.height - (y - self.origin_y) / self.resolution)
        return map_x, map_y


    def random_free(self, goal, bias_prob = 0.20):

        if random.random() < bias_prob:
            return goal

        
        selected_cell = random.choice(self.free_indices)
        return [selected_cell[1], selected_cell[0]]



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
        
        while current_index != 0:
            path.append(self.V[current_index])
            current_index = self.E[current_index]
        path.reverse()  # from start to goal
        return path


    def publish_path(self, points):

        msg = Float64MultiArray()
        flat_points = [coord for point in points for coord in point]  # flatten [[x1,y1],[x2,y2],...] to [x1,y1,x2,y2,...]
        msg.data = flat_points
        self.path_publisher.publish(msg)

    def toggle_pause(self):
        self.paused = not self.paused
        self.get_logger().info(f"{'Paused' if self.paused else 'Resumed'} node.")

    def pose_callback(self, pose_msg):

        if self.paused:
            return

        pos = pose_msg.pose.pose.position
        x, y = pos.x, pos.y

        x_map, y_map = self.world_to_map_coords(x, y)

        self.V = [[x_map, y_map]]
        self.E = [0]
        self.kdtree = KDTree(np.array(self.V))


        # Find first waypoint at least L distance away
        

        # Get yaw from quaternion
        ori = pose_msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])

        if not hasattr(self, 'scan_msg'):
            self.get_logger().warn("No scan message received yet, skipping pose callback.")
            return


        self.scan = self.scan_on_map(x, y, yaw)

        waypoint = self.find_best_waypoint(x, y, yaw, self.Lookahead, self.waypoints)

        

        x_car, y_car = self.world_to_map_coords(waypoint[0], waypoint[1])


        # Transform waypoint into vehicle frame
        #dx = waypoint[0] - x
        #dy = waypoint[1] - y
        #x_car = np.cos(-yaw) * dx - np.sin(-yaw) * dy
        #y_car = np.sin(-yaw) * dx + np.cos(-yaw) * dy

        goal_point = [x_car, y_car]

        k = 1
        found_goal = False
        total_time = time.perf_counter()

        while k < 100 and not found_goal:
            #self.get_logger().info(f"iter: {k}")
            t_iter_start = time.perf_counter()

            t_rand = time.perf_counter()
            x_rand = self.random_free(goal_point)
            t_rand_end = time.perf_counter()

            t_nearest = time.perf_counter()
            #self.get_logger().info(f"random_point: {x_rand}")
            x_nearest, parent_index = self.nearest(x_rand)
            t_nearest_end = time.perf_counter()

            t_expand = time.perf_counter()
            x_new = self.expand(x_nearest, x_rand)
            t_expand_end = time.perf_counter()

            t_collision = time.perf_counter()
            x_collision = self.check_collision(x_nearest, x_new)
            t_collision_end = time.perf_counter()

            t_path = time.perf_counter()
            path_iter = self.extract_path(parent_index)
            t_path_end = time.perf_counter()

            

            if not x_collision: #and len(path_iter) < 12:
                self.V = self.V + [x_new]
                self.E = self.E + [parent_index]
                self.kdtree = KDTree(np.array(self.V))
                x_global_new = x_new[0] * self.resolution + self.origin_x
                y_global_new = (self.height - x_new[1]) * self.resolution + self.origin_y
                dx_new = x_global_new - waypoint[0]
                dy_new = y_global_new - waypoint[1]
                goal_distance = np.hypot(dx_new, dy_new)
                if goal_distance < 1:
                    found_goal = True

            t_iter_end = time.perf_counter()
            self.get_logger().info(
                f"Iter {k}: Rand {1000*(t_rand_end-t_rand):.3f}ms | "
                f"Nearest {1000*(t_nearest_end-t_nearest):.3f}ms | "
                f"Expand {1000*(t_expand_end-t_expand):.3f}ms | "
                f"Collision {1000*(t_collision_end-t_collision):.3f}ms | "
                f"Path {1000*(t_path_end-t_path):.3f}ms | "
                f"Total {1000*(t_iter_end-t_iter_start):.3f}ms"
            )
            k = k+1
            

        

        nearest_goal_point, goal_point_index = self.nearest(goal_point)
        
        path = self.extract_path(goal_point_index)

        

        #self.get_logger().info(f"Path: {path}")

        #path = path + [goal_point]

        #self.get_logger().info(f"Path: {path}")

        self.publish_tree(path)

        for node in path:
            x_index = node[0]
            y_index = node[1]
            x_global = x_index * self.resolution + self.origin_x
            y_global = (self.height - y_index) * self.resolution + self.origin_y
            node[0] = x_global
            node[1] = y_global

        self.publish_path(path)

        time.sleep(5.0)
        #close_waypoint = self.find_best_waypoint(x, y, yaw, self.short_lookahead, path)
        

        

        # Transform waypoint into vehicle frame
        #dx = close_waypoint[0] - x
        #dy = close_waypoint[1] - y
        #x_car = np.cos(-yaw) * dx - np.sin(-yaw) * dy
        #y_car = np.sin(-yaw) * dx + np.cos(-yaw) * dy

        # Compute steering angle
        #curvature = 2 * y_car / (self.short_lookahead ** 2)
        #steering_angle = np.clip(self.K * curvature, -0.34, 0.34)
        #velocity = 0.5

        # Publish Ackermann drive message
        #drive_msg = AckermannDriveStamped()
        #drive_msg.drive.speed = velocity
        #drive_msg.drive.steering_angle = steering_angle
        #self.acker_publisher.publish(drive_msg)


    def scan_callback(self, scan_msg):
        self.scan_msg = scan_msg


    def scan_on_map(self, x, y, yaw):

        if self.scan_msg is None:
            return {}
        
        robot_x, robot_y, robot_theta = x, y, yaw

        angle = self.scan_msg.angle_min

        points = []

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
                    # Inflate around this point
                    for dx in range(-self.inflation_radius, self.inflation_radius + 1):
                        for dy in range(-self.inflation_radius, self.inflation_radius + 1):
                            # Only add cells within a circle around the point
                            if dx**2 + dy**2 <= self.inflation_radius**2:
                                inflated_x = map_x + dx
                                inflated_y = map_y + dy
                                if 0 <= inflated_x < self.width and 0 <= inflated_y < self.height:
                                    points.append((inflated_x, inflated_y))

            angle += self.scan_msg.angle_increment

        points = set(points)

        return points


    def publish_tree(self, path):
        marker_array = MarkerArray()

        # Tree edges
        line_marker = Marker()
        line_marker.header.frame_id = "map"
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = "rrt_tree"
        line_marker.id = 0
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.05  # line width in meters
        line_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)

        for i, parent_index in enumerate(self.E):
            child = self.V[i]
            parent = self.V[parent_index]

            p1 = self.point_msg_from_map_coords(parent)
            p2 = self.point_msg_from_map_coords(child)

            line_marker.points.append(p1)
            line_marker.points.append(p2)

        marker_array.markers.append(line_marker)

        # Optional: final path
        if path:
            path_marker = Marker()
            path_marker.header.frame_id = "map"
            path_marker.header.stamp = self.get_clock().now().to_msg()
            path_marker.ns = "rrt_path"
            path_marker.id = 1
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            path_marker.scale.x = 0.08
            path_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

            for p in path:
                pt = self.point_msg_from_map_coords(p)
                path_marker.points.append(pt)

            marker_array.markers.append(path_marker)

        self.tree_pub.publish(marker_array)



    def point_msg_from_map_coords(self, point):
        x = self.origin_x + point[0] * self.resolution
        y = self.origin_y + (self.height - point[1]) * self.resolution
        return Point(x=x, y=y, z=0.0)


    def inflate_obstacles(self):
        # Create a binary obstacle map: True = obstacle
        obstacle_mask = self.map< self.occupied_thresh

        # Define structuring element size (square or disk-shaped)
        structuring_element = np.ones((2 * self.inflation_radius + 1, 2 * self.inflation_radius + 1))

        # Perform binary dilation to inflate obstacles
        inflated_mask = binary_dilation(obstacle_mask, structure=structuring_element)

        # Create a new inflated map: set inflated areas as occupied (0)
        inflated_map = np.where(inflated_mask, 0, self.map)

        return inflated_map

def main(args=None):
    rclpy.init(args=args)
    
    RRT_node = RRT()
    rclpy.spin(RRT_node)
    RRT_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


