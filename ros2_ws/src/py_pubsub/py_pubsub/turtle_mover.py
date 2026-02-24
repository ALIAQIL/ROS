"""
Turtle Mover Node.
Publishes Twist messages to /turtle1/cmd_vel to move the turtlesim turtle in circles.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class TurtleMover(Node):
    """Moves the turtlesim turtle by publishing velocity commands."""

    def __init__(self):
        super().__init__('turtle_mover')
        self.publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.move)

    def move(self):
        msg = Twist()
        msg.linear.x = 2.0
        msg.angular.z = 1.0
        self.publisher_.publish(msg)
        self.get_logger().info('Mouvement envoyé')


def main(args=None):
    rclpy.init(args=args)
    node = TurtleMover()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
