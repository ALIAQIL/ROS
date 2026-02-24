"""
Gesture Control ROS 2 Node.
Captures webcam frames, classifies hand gestures using the trained CNN,
and publishes Twist commands to /turtle1/cmd_vel to control the turtlesim turtle.

Gesture → Command mapping:
    up    → move forward  (linear.x = 2.0)
    down  → move backward (linear.x = -2.0)
    left  → turn left     (angular.z = 1.5)
    right → turn right    (angular.z = -1.5)

Usage:
    ros2 run py_pubsub gesture_controller --ros-args -p model_path:=./gesture_model.pth
"""

import cv2
import torch
import numpy as np
from collections import deque
from torchvision import transforms
from PIL import Image

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from gesture_model import GestureCNN, CLASS_NAMES, IMG_SIZE


class GestureControlNode(Node):
    """ROS 2 node that controls turtlesim using hand gestures detected via webcam."""

    def __init__(self):
        super().__init__('gesture_controller')

        # Declare parameters
        self.declare_parameter('model_path', './gesture_model.pth')
        self.declare_parameter('publish_rate', 5.0)  # Hz
        self.declare_parameter('window_size', 10)
        self.declare_parameter('confidence_threshold', 0.6)
        self.declare_parameter('linear_speed', 2.0)
        self.declare_parameter('angular_speed', 1.5)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        window_size = self.get_parameter('window_size').get_parameter_value().integer_value
        self.confidence_threshold = self.get_parameter(
            'confidence_threshold').get_parameter_value().double_value
        self.linear_speed = self.get_parameter(
            'linear_speed').get_parameter_value().double_value
        self.angular_speed = self.get_parameter(
            'angular_speed').get_parameter_value().double_value

        # Publisher
        self.publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)

        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GestureCNN()
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        self.get_logger().info(f'Model loaded from {model_path}')

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Sliding window
        self.prediction_window = deque(maxlen=window_size)

        # Webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('Cannot open webcam!')
            return

        # Timer — publishes commands at the specified rate
        # A rate of ~5 Hz provides smooth control without overwhelming
        # the turtle with too many commands.
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.control_callback)

        self.get_logger().info(
            f'Gesture controller started | Rate: {publish_rate} Hz | '
            f'Window: {window_size} | Threshold: {self.confidence_threshold}')

    def predict_gesture(self):
        """Capture a frame and predict the gesture."""
        ret, frame = self.cap.read()
        if not ret:
            return None, None, 0.0

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # ROI
        roi_size = 300
        x1 = w // 2 - roi_size // 2
        y1 = h // 2 - roi_size // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        roi = frame[y1:y2, x1:x2]

        # Preprocess
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(roi_rgb)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return frame, predicted.item(), confidence.item()

    def control_callback(self):
        """Timer callback: predict gesture and publish Twist command."""
        frame, predicted_class, confidence = self.predict_gesture()
        if frame is None:
            return

        # Smoothing
        if confidence >= self.confidence_threshold:
            self.prediction_window.append(predicted_class)

        # Majority vote
        cmd = Twist()
        gesture = "none"

        if len(self.prediction_window) > 0:
            vote_counts = np.bincount(
                list(self.prediction_window), minlength=len(CLASS_NAMES))
            smoothed_class = np.argmax(vote_counts)
            gesture = CLASS_NAMES[smoothed_class]

            if gesture == 'up':
                cmd.linear.x = self.linear_speed
            elif gesture == 'down':
                cmd.linear.x = -self.linear_speed
            elif gesture == 'left':
                cmd.angular.z = self.angular_speed
            elif gesture == 'right':
                cmd.angular.z = -self.angular_speed

        self.publisher_.publish(cmd)
        self.get_logger().info(
            f'Gesture: {gesture.upper()} | Confidence: {confidence:.0%} | '
            f'Cmd: lin={cmd.linear.x:.1f}, ang={cmd.angular.z:.1f}')

        # Display (optional visualization)
        h, w = frame.shape[:2]
        roi_size = 300
        x1 = w // 2 - roi_size // 2
        y1 = h // 2 - roi_size // 2
        cv2.rectangle(frame, (x1, y1), (x1 + roi_size, y1 + roi_size), (0, 255, 0), 2)
        cv2.putText(frame, f'Gesture: {gesture.upper()}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f'Confidence: {confidence:.0%}',
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Gesture Control', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        """Clean up webcam and OpenCV windows."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GestureControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
