#!/usr/bin/env python3
"""
Simple People Detection Example for FLIR Lepton 3.5

This is a simplified version of the people detection system that demonstrates
the basic concepts without complex tracking. Great for learning and customization.

This example shows:
- Importing the flir_lepton module
- Streaming thermal video
- Detecting warm blobs (potential people)
- Drawing bounding boxes
- Counting detections

Requirements:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
from flir_lepton import FLIRLepton35, LeptonError


def detect_warm_objects(thermal_frame, min_temp=30, min_area=100, max_area=4000):
    """
    Detect warm objects in a thermal frame.

    Args:
        thermal_frame: Normalized thermal image (0-255, uint8)
        min_temp: Minimum temperature threshold for detection
        min_area: Minimum object area in pixels
        max_area: Maximum object area in pixels

    Returns:
        List of bounding boxes [(x, y, w, h), ...]
    """
    detections = []

    # Threshold to find hot regions
    _, binary = cv2.threshold(thermal_frame, min_temp, 255, cv2.THRESH_BINARY)

    # Clean up noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours (blobs)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by size
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            detections.append((x, y, w, h))

    return detections


def draw_detections(frame, detections, scale=4):
    """
    Draw bounding boxes on the frame.

    Args:
        frame: Thermal frame (grayscale)
        detections: List of (x, y, w, h) bounding boxes
        scale: Display scale factor

    Returns:
        Annotated frame with bounding boxes
    """
    # Apply colormap for better visualization
    colored = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

    # Scale up for display
    height, width = colored.shape[:2]
    display = cv2.resize(colored, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)

    # Draw bounding boxes
    for i, (x, y, w, h) in enumerate(detections, 1):
        # Scale coordinates
        x *= scale
        y *= scale
        w *= scale
        h *= scale

        # Draw rectangle
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw label
        label = f"Person {i}"
        cv2.putText(display, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw count
    count_text = f"Count: {len(detections)}"
    cv2.putText(display, count_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.putText(display, count_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return display


def main():
    """Main function"""
    print("=" * 60)
    print("Simple People Detection for FLIR Lepton 3.5")
    print("=" * 60)
    print()
    print("This example detects warm objects (people) and draws boxes.")
    print()
    print("Controls:")
    print("  Q - Quit")
    print("  + - Increase temperature threshold")
    print("  - - Decrease temperature threshold")
    print("  S - Save screenshot")
    print()

    # Detection parameters
    min_temp_threshold = 30  # Adjust based on your environment
    min_area = 100
    max_area = 4000
    display_scale = 4

    try:
        # Initialize camera (imports our library!)
        with FLIRLepton35() as camera:
            print("✓ Camera initialized successfully")
            print(f"  Temperature threshold: {min_temp_threshold}")
            print()

            window_name = "Simple People Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            screenshot_count = 0

            # Stream and process frames
            for frame in camera.get_frame_stream(normalize=True):
                # Detect people
                detections = detect_warm_objects(
                    frame,
                    min_temp=min_temp_threshold,
                    min_area=min_area,
                    max_area=max_area
                )

                # Visualize
                display = draw_detections(frame, detections, scale=display_scale)

                # Show threshold value on screen
                threshold_text = f"Temp Threshold: {min_temp_threshold}"
                cv2.putText(display, threshold_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4)
                cv2.putText(display, threshold_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Display
                cv2.imshow(window_name, display)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting...")
                    break
                elif key == ord('+') or key == ord('='):
                    min_temp_threshold = min(255, min_temp_threshold + 5)
                    print(f"Temperature threshold: {min_temp_threshold}")
                elif key == ord('-') or key == ord('_'):
                    min_temp_threshold = max(0, min_temp_threshold - 5)
                    print(f"Temperature threshold: {min_temp_threshold}")
                elif key == ord('s') or key == ord('S'):
                    filename = f"detection_{screenshot_count:04d}.png"
                    cv2.imwrite(filename, display)
                    print(f"Saved: {filename}")
                    screenshot_count += 1

            cv2.destroyAllWindows()

    except LeptonError as e:
        print(f"✗ Camera error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n✓ Interrupted by user")
        cv2.destroyAllWindows()
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n✓ Done!")
    return 0


if __name__ == "__main__":
    exit(main())
