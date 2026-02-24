"""
Gesture Data Collection Script.
Uses the webcam to capture hand gesture images for training.
Images are saved into class directories: up, down, left, right.

Usage:
    python3 collect_gestures.py [--output_dir ./gesture_data] [--num_images 200]

Controls:
    u  - Switch to capturing 'up' gesture
    d  - Switch to capturing 'down' gesture
    l  - Switch to capturing 'left' gesture
    r  - Switch to capturing 'right' gesture
    s  - Start / stop capturing
    q  - Quit
"""

import os
import cv2
import argparse
import time


def main():
    parser = argparse.ArgumentParser(description='Collect hand gesture images')
    parser.add_argument('--output_dir', type=str, default='./gesture_data',
                        help='Root directory to save gesture images')
    parser.add_argument('--num_images', type=int, default=200,
                        help='Number of images to capture per class')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Size to resize captured images')
    args = parser.parse_args()

    classes = ['up', 'down', 'left', 'right']
    for cls in classes:
        os.makedirs(os.path.join(args.output_dir, cls), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    current_class = 'up'
    capturing = False
    count = 0

    print("=== Gesture Data Collection ===")
    print(f"Output directory: {args.output_dir}")
    print(f"Target images per class: {args.num_images}")
    print()
    print("Controls:")
    print("  u/d/l/r - Switch to up/down/left/right class")
    print("  s       - Start/stop capturing")
    print("  q       - Quit")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        frame = cv2.flip(frame, 1)  # Mirror for intuitive interaction

        # Draw ROI rectangle (center region for hand)
        h, w = frame.shape[:2]
        roi_size = 300
        x1 = w // 2 - roi_size // 2
        y1 = h // 2 - roi_size // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Status text
        status = "CAPTURING" if capturing else "PAUSED"
        color = (0, 0, 255) if capturing else (0, 165, 255)
        existing = len(os.listdir(os.path.join(args.output_dir, current_class)))
        cv2.putText(frame, f'Class: {current_class.upper()} | {status} | Count: {existing}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, 'Place hand inside green box',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if capturing:
            # Extract and save ROI
            roi = frame[y1:y2, x1:x2]
            roi_resized = cv2.resize(roi, (args.img_size, args.img_size))
            filename = os.path.join(args.output_dir, current_class,
                                    f'{current_class}_{int(time.time() * 1000)}.jpg')
            cv2.imwrite(filename, roi_resized)
            count += 1

            if count >= args.num_images:
                capturing = False
                count = 0
                print(f"Done capturing {args.num_images} images for '{current_class}'")

        cv2.imshow('Gesture Collection', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('u'):
            current_class = 'up'
            count = 0
            print(f"Switched to class: UP")
        elif key == ord('d'):
            current_class = 'down'
            count = 0
            print(f"Switched to class: DOWN")
        elif key == ord('l'):
            current_class = 'left'
            count = 0
            print(f"Switched to class: LEFT")
        elif key == ord('r'):
            current_class = 'right'
            count = 0
            print(f"Switched to class: RIGHT")
        elif key == ord('s'):
            capturing = not capturing
            count = 0
            print(f"Capturing: {'ON' if capturing else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection finished.")


if __name__ == '__main__':
    main()
