#!/usr/bin/env python3


import rclpy
from rclpy.node import Node







class RRT(Node):
    def __init__(self):
        super().__init__('RRT_node')

        # ROS Topics
        odom_topic = '/ego_racecar/odom'
        drive_topic = '/drive'
        waypoint_topic = '/waypoint'
        self.pose_subscription = self.create_subscription(Odometry, odom_topic, self.pose_callback, 10)
        self.acker_publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.marker_pub = self.create_publisher(Marker, '/vertex_marker', 10)
        


        self.V = []
        self.E = []






    def nearest(self):





    def expand(self):






    def edge_collision(self):



    def pose_callback(self,msg):

        for k in range(150):

            x_rand = [rand(map_resolution), rand(map_resolution)]
            if map(x_rand) = 0:
                



def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    RRT_node = RRT()
    rclpy.spin(RRT_node)
    RRT_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


