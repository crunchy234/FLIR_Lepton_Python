#!/usr/bin/env python3
"""
Video Stream Example: Display live thermal video from FLIR Lepton 3.5

This example demonstrates:
- Continuous frame capture using the frame stream
- Real-time display using OpenCV
- Applying color maps for better visualization
- Frame rate calculation

Requirements:
    pip install opencv-python numpy
"""

import time
import numpy as np
from flir_lepton import FLIRLepton35, LeptonError

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available. Install with: pip install opencv-python")


class FrameRateCounter:
    """Simple frame rate counter"""

    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size
        self.timestamps = []

    def tick(self):
        """Record a frame timestamp"""
        self.timestamps.append(time.time())
        if len(self.timestamps) > self.buffer_size:
            self.timestamps.pop(0)

    def get_fps(self):
        """Calculate current FPS"""
        if len(self.timestamps) < 2:
            return 0.0

        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff > 0:
            return (len(self.timestamps) - 1) / time_diff
        return 0.0


def main():
    print("FLIR Lepton 3.5 - Video Stream Example")
    print("=" * 50)

    if not CV2_AVAILABLE:
        print("✗ OpenCV is required for this example")
        print("  Install with: pip install opencv-python")
        return 1

    # Available colormaps (try different ones!)
    colormaps = [
        ('JET', cv2.COLORMAP_JET),
        ('HOT', cv2.COLORMAP_HOT),
        ('RAINBOW', cv2.COLORMAP_RAINBOW),
        ('BONE', cv2.COLORMAP_BONE),
        ('INFERNO', cv2.COLORMAP_INFERNO),
        ('VIRIDIS', cv2.COLORMAP_VIRIDIS),
    ]
    current_colormap_idx = 0

    print("Controls:")
    print("  Q - Quit")
    print("  C - Change colormap")
    print("  S - Save current frame")
    print()

    try:
        # Initialize camera
        with FLIRLepton35() as camera:
            print("✓ Camera initialized")
            print("  Starting video stream...")
            print()

            # Create window
            window_name = 'FLIR Lepton 3.5 - Thermal Video'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            # Frame rate counter
            fps_counter = FrameRateCounter()
            frame_count = 0

            # Get frame stream
            for frame in camera.get_frame_stream(normalize=True):
                frame_count += 1
                fps_counter.tick()

                # Apply colormap
                colormap_name, colormap = colormaps[current_colormap_idx]
                colored_frame = cv2.applyColorMap(frame, colormap)

                # Resize for display (scale up from 160x120)
                display_frame = cv2.resize(colored_frame, (640, 480),
                                          interpolation=cv2.INTER_NEAREST)

                # Add information overlay
                fps = fps_counter.get_fps()
                info_text = [
                    f"FPS: {fps:.1f}",
                    f"Colormap: {colormap_name}",
                    f"Frame: {frame_count}",
                    f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
                ]

                # Draw text overlay
                y_offset = 30
                for i, text in enumerate(info_text):
                    cv2.putText(display_frame, text, (10, y_offset + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_frame, text, (10, y_offset + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                # Display frame
                cv2.imshow(window_name, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting...")
                    break
                elif key == ord('c') or key == ord('C'):
                    # Change colormap
                    current_colormap_idx = (current_colormap_idx + 1) % len(colormaps)
                    print(f"Colormap changed to: {colormaps[current_colormap_idx][0]}")
                elif key == ord('s') or key == ord('S'):
                    # Save frame
                    filename = f'thermal_frame_{frame_count}.png'
                    cv2.imwrite(filename, colored_frame)
                    print(f"Frame saved: {filename}")

            # Cleanup
            cv2.destroyAllWindows()
            print(f"\nTotal frames captured: {frame_count}")
            print(f"Average FPS: {fps_counter.get_fps():.2f}")

    except LeptonError as e:
        print(f"✗ Lepton Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n✓ Interrupted by user")
        cv2.destroyAllWindows()
        return 0
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
