#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header

class ParticleVisualizer(Node):
    def __init__(self):
        super().__init__('amcl_viz_node')

        self.subscription = self.create_subscription(
            PoseArray,
            '/particlecloud',
            self.particle_callback,
            10
        )

        self.publisher = self.create_publisher(
            MarkerArray,
            '/particle_markers',
            10
        )

        self.marker_ns = 'particles'
        self.frame_id = 'map'  # make sure this matches AMCL's global_frame_id
        self.get_logger().info("Particle Visualizer Node started")

    def particle_callback(self, msg):
        print("Received particle cloud with {} poses".format(len(msg.poses)))
        marker_array = MarkerArray()

        for i, pose in enumerate(msg.poses):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = self.marker_ns
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = pose
            marker.scale.x = 0.2  # Arrow shaft length
            marker.scale.y = 0.05  # Arrow shaft diameter
            marker.scale.z = 0.05  # Arrow head diameter
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.8
            marker.color.b = 1.0
            marker_array.markers.append(marker)

        # Delete old markers if number decreases
        delete_markers = MarkerArray()
        for j in range(len(msg.poses), 5000):  # assuming 5000 as max marker count
            delete_marker = Marker()
            delete_marker.action = Marker.DELETE
            delete_marker.id = j
            delete_marker.ns = self.marker_ns
            delete_marker.header.frame_id = self.frame_id
            delete_markers.markers.append(delete_marker)
        marker_array.markers.extend(delete_markers.markers)

        self.publisher.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = ParticleVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

