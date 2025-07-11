import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import message_filters

import csv
import os
import matplotlib.pyplot as plt
import numpy as np

class PoseLogger(Node):
    def __init__(self):
        super().__init__('test_logger')

        # File paths
        self.output_csv = '/home/f1tenth/f1tenth_mrc_ws/poses.csv'
        self.racetrack_csv = '/home/f1tenth/f1tenth_mrc_ws/raceline.csv'
        self.output_png = '/home/f1tenth/f1tenth_mrc_ws/pose_plots.png'

        # Subscribers using message_filters for synchronization
        self.pose1_sub = message_filters.Subscriber(self, PoseStamped, '/optitrack/f1tenth/pose')
        self.pose2_sub = message_filters.Subscriber(self, PoseStamped, '/optitrack/astro/pose')

        # Synchronize messages (Approximate time)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.pose1_sub, self.pose2_sub], queue_size=50, slop=0.1
        )
        self.ts.registerCallback(self.pose_callback)

        self.data = []

        self.get_logger().info('PoseLogger (PoseStamped) initialized and listening...')

    def pose_callback(self, pose1: PoseStamped, pose2: PoseStamped):
        # Use header timestamps for timing (in seconds)
        t_sec = pose1.header.stamp.sec + pose1.header.stamp.nanosec * 1e-9

        x1, y1 = pose1.pose.position.x, pose1.pose.position.y
        x2, y2 = pose2.pose.position.x, pose2.pose.position.y

        self.data.append([t_sec, x1, y1, x2, y2])
        self.get_logger().info(f"t={t_sec:.2f}: Car1({x1:.2f},{y1:.2f}) | Car2({x2:.2f},{y2:.2f})")

    def save_data(self):
        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'x1', 'y1', 'x2', 'y2'])
            writer.writerows(self.data)
        self.get_logger().info(f"CSV saved to '{self.output_csv}'.")

    def plot_data(self):
        if not os.path.exists(self.output_csv) or not os.path.exists(self.racetrack_csv):
            self.get_logger().error("Missing CSV file(s). Ensure pose and racetrack CSVs exist.")
            return

        data = np.loadtxt(self.output_csv, delimiter=',', skiprows=1)
        time = data[:,0]
        x1 = data[:,1]
        y1 = data[:,2]
        x2 = data[:,3]
        y2 = data[:,4]

        track = np.loadtxt(self.racetrack_csv, delimiter=',')
        tx = track[:,0]
        ty = track[:,1]

        fig, axs = plt.subplots(3, 1, figsize=(12, 10))

        # 1. XY Path
        axs[0].plot(x1, y1, label='Ego', marker='o', markersize=2)
        axs[0].plot(x2, y2, label='Opp', marker='x', markersize=2)
        axs[0].set_title('XY Path')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].legend()
        axs[0].axis('equal')

        # 2. X vs Time
        axs[1].plot(time, x1, label='Ego')
        axs[1].plot(time, x2, label='Opp')
        axs[1].set_title('X over Time')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('X')
        axs[1].legend()

        # 3. Y vs Time
        axs[2].plot(time, y1, label='Ego')
        axs[2].plot(time, y2, label='Opp')
        axs[2].set_title('Y over Time')
        axs[2].set_xlabel('Time [s]')
        axs[2].set_ylabel('Y')
        axs[2].legend()

        '''
        # 4. Cars on Racetrack
        axs[1, 1].plot(tx, ty, 'k--', label='Track')
        axs[1, 1].plot(x1, y1, label='Ego')
        axs[1, 1].plot(x2, y2, label='Opp')
        axs[1, 1].set_title('Cars on Racetrack')
        axs[1, 1].set_xlabel('X')
        axs[1, 1].set_ylabel('Y')
        axs[1, 1].legend()
        axs[1, 1].axis('equal')
        '''

        plt.tight_layout()
        plt.savefig(self.output_png)
        self.get_logger().info(f"Plot saved as '{self.output_png}'")

    def destroy_node(self):
        if self.data:
            self.save_data()
            self.plot_data()
        else:
            self.get_logger().warn('No data collected to save or plot.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PoseLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down via KeyboardInterrupt.')
    finally:
        node.destroy_node()
        rclpy.shutdown()
