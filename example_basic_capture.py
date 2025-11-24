#!/usr/bin/env python3
"""
Basic Example: Capture and save a thermal image from FLIR Lepton 3.5

This example demonstrates:
- Initializing the camera
- Capturing a single frame
- Saving the frame as an image file
"""

import numpy as np
from flir_lepton import FLIRLepton35, LeptonError

try:
    # Optional: Use PIL for saving images
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available. Install with: pip install Pillow")


def save_frame_numpy(frame, filename='thermal_image.npy'):
    """Save frame as numpy array"""
    np.save(filename, frame)
    print(f"Frame saved as numpy array: {filename}")


def save_frame_image(frame, filename='thermal_image.png'):
    """Save frame as image file using PIL"""
    if not PIL_AVAILABLE:
        print("Cannot save as image: PIL not installed")
        return

    img = Image.fromarray(frame)
    img.save(filename)
    print(f"Frame saved as image: {filename}")


def main():
    print("FLIR Lepton 3.5 - Basic Capture Example")
    print("=" * 50)

    try:
        # Initialize camera with context manager for automatic cleanup
        with FLIRLepton35() as camera:
            print("Camera initialized successfully!")
            print()

            # Capture a normalized frame (0-255 range)
            print("Capturing thermal frame...")
            frame = camera.capture_frame(normalize=True)

            if frame is not None:
                print(f"✓ Frame captured successfully!")
                print(f"  Shape: {frame.shape}")
                print(f"  Data type: {frame.dtype}")
                print(f"  Min value: {frame.min()}")
                print(f"  Max value: {frame.max()}")
                print(f"  Mean value: {frame.mean():.2f}")
                print()

                # Save the frame
                save_frame_numpy(frame, 'thermal_frame.npy')
                save_frame_image(frame, 'thermal_frame.png')
                print()

                # Also capture raw thermal data
                print("Capturing raw thermal data...")
                raw_frame = camera.capture_frame(normalize=False)
                if raw_frame is not None:
                    print(f"✓ Raw frame captured!")
                    print(f"  Raw values range: {raw_frame.min()} - {raw_frame.max()}")
                    save_frame_numpy(raw_frame, 'thermal_frame_raw.npy')

            else:
                print("✗ Failed to capture frame")

    except LeptonError as e:
        print(f"✗ Lepton Error: {e}")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
