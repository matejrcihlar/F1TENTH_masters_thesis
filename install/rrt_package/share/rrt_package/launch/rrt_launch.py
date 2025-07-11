from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # Start the RRT node
        Node(
            package='rrt_package',
            executable='RRT_node',
            name='rrt_node',
            output='screen'
        ),

        # Start the Pure Pursuit node
        Node(
            package='rrt_package',
            executable='rrt_pure_pursuit',
            name='pure_pursuit_node',
            output='screen'
        )
    ])

