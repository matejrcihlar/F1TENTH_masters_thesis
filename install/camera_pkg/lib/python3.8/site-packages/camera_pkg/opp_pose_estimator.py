#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs

import pyrealsense2 as rs
import numpy as np
from scipy.spatial import cKDTree
import cv2
from pupil_apriltags import Detector
import tf_transformations
from tf_transformations import quaternion_from_matrix

class AprilTagPosePublisher(Node):
    def __init__(self):
        super().__init__('opp_pose_publisher')

        self.publisher_ = self.create_publisher(Odometry, '/opp_pose', 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


        # AprilTag detector setup
        self.detector = Detector(families='tag36h11')

        # RealSense pipeline
        self.pipeline = rs.pipeline()
        print(self.pipeline)
        config = rs.config()
        print(config)

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        print(device_product_line)

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        profile = self.pipeline.get_active_profile()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy, self.cx, self.cy = intr.fx, intr.fy, intr.ppx, intr.ppy
        self.tag_size = 0.05  # meters

        # Timer callback at 30 Hz
        self.framerate = 30
        self.timer = self.create_timer(1.0 / self.framerate, self.timer_callback)
        
        self.last_pose = [0.0, 0.0, 0.0]
        self.max_speed = 3.0
        self.scan = []
        self.angle_min = 0.0
        self.angle_increment = 0.0
        self.extraction_radius = 0.3
        self.search_window = [(-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)]
        self.feature_points = []

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        tags = self.detector.detect(gray,
                                    estimate_tag_pose=True,
                                    camera_params=[self.fx, self.fy, self.cx, self.cy],
                                    tag_size=self.tag_size)
        
        if not tags:
            if not self.feature_points:
                self.feature_points = self.extract_opponent_points()
            lidar_pose = self.match_scan_safe()
            if not lidar_pose:
                odom_msg = Odometry()
                odom_msg.header = Header()
                odom_msg.header.stamp = self.get_clock().now().to_msg()
                odom_msg.header.frame_id = 'laser'
                odom_msg.child_frame_id = 'opp_pose'

                odom_msg.pose.pose.position.x = -3.0
            else:
                odom_msg = Odometry()
                odom_msg.header = Header()
                odom_msg.header.stamp = self.get_clock().now().to_msg()
                odom_msg.header.frame_id = 'laser'
                odom_msg.child_frame_id = 'opp_pose'

                odom_msg.pose.pose.position.x = lidar_pose[0]
                odom_msg.pose.pose.position.y = lidar_pose[1]
                odom_msg.pose.pose.position.z = 0.0

                yaw = lidar_pose[2]

                qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw)

                odom_msg.pose.pose.orientation = Quaternion(
                    x=qx,
                    y=qy,
                    z=qz,
                    w=qw
                )

                odom_msg.twist.twist.linear.x = (odom_msg.pose.pose.position.x - self.last_pose[0]) * self.framerate / self.max_speed
                odom_msg.twist.twist.linear.y = (odom_msg.pose.pose.position.y - self.last_pose[1]) * self.framerate / self.max_speed
                odom_msg.twist.twist.angular.z = (yaw - self.last_pose[2]) * self.framerate / 3.2

                self.last_pose[0] = lidar_pose[0]
                self.last_pose[1] = lidar_pose[1]
                self.last_pose[2] = yaw


            transformed_odom = self.transform_opp_odometry(odom_msg, self.tf_buffer, target_frame='base_link')
            if transformed_odom:
                self.publisher_.publish(transformed_odom)

            #self.get_logger().info(f"Published pose for tag {tag.tag_id}")

        else:
            for tag in tags:
                rmat = tag.pose_R
                tvec = tag.pose_t

                # 3x4 pose matrix for quaternion conversion
                pose_mat = np.hstack((rmat, tvec))
                pose_mat = np.vstack((pose_mat, [0, 0, 0, 1]))
                q = quaternion_from_matrix(pose_mat)

                odom_msg = Odometry()
                odom_msg.header = Header()
                odom_msg.header.stamp = self.get_clock().now().to_msg()
                odom_msg.header.frame_id = 'camera_link'
                odom_msg.child_frame_id = f"tag_{tag.tag_id}"

                odom_msg.pose.pose.position.x = float(tvec[0])
                odom_msg.pose.pose.position.y = float(tvec[1])
                odom_msg.pose.pose.position.z = float(tvec[2])

                _, _ ,yaw = tf_transformations.euler_from_quaternion(q)
                
                odom_msg.pose.pose.orientation = Quaternion(
                    x=float(q[0]),
                    y=float(q[1]),
                    z=float(q[2]),
                    w=float(q[3])
                )

                odom_msg.twist.twist.linear.x = (odom_msg.pose.pose.position.x - self.last_pose[0]) * self.framerate / self.max_speed
                odom_msg.twist.twist.linear.y = (odom_msg.pose.pose.position.y - self.last_pose[1]) * self.framerate / self.max_speed
                odom_msg.twist.twist.angular.z = (yaw - self.last_pose[2]) * self.framerate / 3.2

                self.last_pose[0] = float(tvec[0])
                self.last_pose[1] = float(tvec[1])
                self.last_pose[2] = yaw

                transformed_odom = self.transform_opp_odometry(odom_msg, self.tf_buffer, target_frame='base_link')
                if transformed_odom:
                    self.publisher_.publish(transformed_odom)

                #self.get_logger().info(f"Published pose for tag {tag.tag_id}")
                self.feature_points = []

    def lidar_callback(self, msg):
        self.scan = msg.ranges
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment

    def extract_opponent_points(self):
        """
        Extract opponent points around given pose from raw laser scan data.
        
        Inputs:
        - ranges: list or np.array of lidar ranges (in meters)
        - angle_min: start angle of scan (in radians)
        - angle_increment: angular resolution of scan (in radians)
        - pose: previous opponent pose (x, y, yaw)
        - extraction_radius: distance in meters to extract points around pose

        Output:
        - feature_points: numpy array (N, 2) of extracted points
        """
        # Build angles array
        angles = self.angle_min + np.arange(len(self.ranges)) * self.angle_increment

        # Convert scan to cartesian
        xs = np.array(self.ranges) * np.cos(angles)
        ys = np.array(self.ranges) * np.sin(angles)

        # Previous pose
        px, py, pyaw = self.last_pose

        # Compute distance from previous pose
        dx = xs - px
        dy = ys - py
        dists = np.sqrt(dx**2 + dy**2)

        mask = dists < self.extraction_radius
        points = np.vstack((xs[mask], ys[mask])).T

        return points

    def match_scan_safe(self, resolution=0.05, max_pose_delta=0.3, max_yaw_delta=0.4):
        """
        Match feature points against new scan.

        Inputs:
        - feature_points: saved opponent points (from extract_opponent_points)
        - ranges, angle_min, angle_increment: new scan data
        - prev_pose: previous opponent pose (x, y, yaw)
        - search_window: ((dx_min, dx_max), (dy_min, dy_max), (dyaw_min, dyaw_max))
        - resolution: search step in meters and radians
        - max_pose_delta: maximum allowed movement in meters (safety check)
        - max_yaw_delta: maximum allowed rotation in radians (safety check)

        Outputs:
        - (new_pose, score) or (None, None) if exceeded limits
        """
        px, py, pyaw = self.last_pose

        # Convert new scan to cartesian
        angles = self.angle_min + np.arange(len(self.ranges)) * self.angle_increment
        xs = np.array(self.ranges) * np.cos(angles)
        ys = np.array(self.ranges) * np.sin(angles)
        scan_points = np.vstack((xs, ys)).T

        scan_tree = cKDTree(scan_points)

        best_score = float('inf')
        best_pose = self.last_pose

        dx_range = np.arange(self.search_window[0][0], self.search_window[0][1] + resolution, resolution)
        dy_range = np.arange(self.search_window[1][0], self.search_window[1][1] + resolution, resolution)
        dyaw_range = np.arange(self.search_window[2][0], self.search_window[2][1] + resolution/3, resolution/3)

        for dx in dx_range:
            for dy in dy_range:
                for dyaw in dyaw_range:
                    candidate_x = px + dx
                    candidate_y = py + dy
                    candidate_yaw = pyaw + dyaw

                    # Apply candidate transform
                    cos_yaw = np.cos(candidate_yaw)
                    sin_yaw = np.sin(candidate_yaw)
                    
                    rot = np.array([[cos_yaw, -sin_yaw],
                                    [sin_yaw,  cos_yaw]])
                    
                    transformed_points = (rot @ self.feature_points.T).T
                    transformed_points += np.array([candidate_x, candidate_y])

                    # Compute match score
                    dists, _ = scan_tree.query(transformed_points, k=1)
                    score = np.mean(dists)

                    if score < best_score:
                        best_score = score
                        best_pose = (candidate_x, candidate_y, candidate_yaw)

        # Safety check: did it move too much?
        delta_x = best_pose[0] - px
        delta_y = best_pose[1] - py
        delta_dist = np.sqrt(delta_x**2 + delta_y**2)
        delta_yaw = abs((best_pose[2] - pyaw + np.pi) % (2 * np.pi) - np.pi)  # wrap-around safe

        if delta_dist > max_pose_delta or delta_yaw > max_yaw_delta or best_score > 3.0:
            # Pose moved too much, likely lost track
            return []

        return best_pose

    def transform_opp_odometry(odom_msg: Odometry, tf_buffer: tf2_ros.Buffer, target_frame: str = 'base_link') -> Odometry:

        try:
            transform = tf_buffer.lookup_transform(
                target_frame=target_frame,
                source_frame=odom_msg.header.frame_id,
                time=odom_msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            # Pose transform
            pose_stamped = PoseStamped()
            pose_stamped.header = odom_msg.header
            pose_stamped.pose = odom_msg.pose.pose
            transformed_pose = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)

            # Velocity transform
            rot_q = transform.transform.rotation
            q = [rot_q.x, rot_q.y, rot_q.z, rot_q.w]
            R = quaternion_matrix(q)[:3, :3]

            lin = odom_msg.twist.twist.linear
            ang = odom_msg.twist.twist.angular
            lin_vec = np.array([lin.x, lin.y, lin.z])
            ang_vec = np.array([ang.x, ang.y, ang.z])
            lin_rot = R @ lin_vec
            ang_rot = R @ ang_vec

            # New odometry
            new_odom = Odometry()
            new_odom.header.stamp = odom_msg.header.stamp
            new_odom.header.frame_id = target_frame
            new_odom.child_frame_id = odom_msg.child_frame_id
            new_odom.pose.pose = transformed_pose.pose
            new_odom.twist.twist.linear.x = lin_rot[0]
            new_odom.twist.twist.linear.y = lin_rot[1]
            new_odom.twist.twist.linear.z = lin_rot[2]
            new_odom.twist.twist.angular.x = ang_rot[0]
            new_odom.twist.twist.angular.y = ang_rot[1]
            new_odom.twist.twist.angular.z = ang_rot[2]
            return new_odom

        except Exception as e:
            print(f"[TF Error] {e}")
            return None


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pipeline.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
