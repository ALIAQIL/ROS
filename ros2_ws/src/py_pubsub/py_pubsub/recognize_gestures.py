"""
Real-Time Gesture Recognition (standalone, no ROS 2).
Uses the trained CNN model to recognize hand gestures from a webcam feed.
Applies a sliding window vote to smooth predictions.

Usage:
    python3 recognize_gestures.py --model ./gesture_model.pth
"""

import cv2
import torch
import argparse
import numpy as np
from collections import deque
from torchvision import transforms
from PIL import Image

# Import model architecture
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from gesture_model import GestureCNN, CLASS_NAMES, IMG_SIZE


def main():
    parser = argparse.ArgumentParser(description='Real-time gesture recognition')
    parser.add_argument('--model', type=str, default='./gesture_model.pth',
                        help='Path to trained model weights')
    parser.add_argument('--window_size', type=int, default=10,
                        help='Sliding window size for vote smoothing')
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                        help='Minimum confidence to accept a prediction')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = GestureCNN()
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Preprocessing transform (same as evaluation)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Sliding window for vote smoothing
    # This helps prevent erratic frame-by-frame predictions.
    # The window keeps the last N predictions, and the final decision
    # is the majority vote. This acts like a temporal filter.
    prediction_window = deque(maxlen=args.window_size)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("=== Real-Time Gesture Recognition ===")
    print(f"Model: {args.model}")
    print(f"Smoothing window: {args.window_size}")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Extract ROI (center)
        roi_size = 300
        x1 = w // 2 - roi_size // 2
        y1 = h // 2 - roi_size // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        roi = frame[y1:y2, x1:x2]

        # Preprocess
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(roi_rgb)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = predicted.item()
            confidence_val = confidence.item()

        # Apply to sliding window
        if confidence_val >= args.confidence_threshold:
            prediction_window.append(predicted_class)

        # Majority vote
        if len(prediction_window) > 0:
            vote_counts = np.bincount(list(prediction_window), minlength=len(CLASS_NAMES))
            smoothed_class = np.argmax(vote_counts)
            gesture_label = CLASS_NAMES[smoothed_class]
        else:
            gesture_label = "---"

        # Draw UI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label and confidence
        raw_label = CLASS_NAMES[predicted_class]
        cv2.putText(frame, f'Raw: {raw_label} ({confidence_val:.0%})',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Smoothed: {gesture_label.upper()}',
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        # Direction arrow
        arrow_colors = {'up': (0, 255, 0), 'down': (0, 0, 255),
                        'left': (255, 0, 0), 'right': (255, 165, 0)}
        if gesture_label in arrow_colors:
            center = (w - 60, 60)
            arrow_pts = {
                'up': (center[0], center[1] - 30),
                'down': (center[0], center[1] + 30),
                'left': (center[0] - 30, center[1]),
                'right': (center[0] + 30, center[1]),
            }
            cv2.arrowedLine(frame, center, arrow_pts[gesture_label],
                            arrow_colors[gesture_label], 3, tipLength=0.4)

        cv2.imshow('Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
