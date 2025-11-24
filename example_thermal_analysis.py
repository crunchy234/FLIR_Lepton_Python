#!/usr/bin/env python3
"""
Thermal Analysis Example: Analyze thermal data from FLIR Lepton 3.5

This example demonstrates:
- Analyzing raw thermal data
- Finding hot and cold spots
- Calculating thermal statistics
- Visualizing temperature distribution

Note: This uses relative temperature values. For absolute temperature,
calibration data and additional processing would be required.
"""

import numpy as np
from flir_lepton import FLIRLepton35, LeptonError

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def find_hot_cold_spots(frame, num_spots=5):
    """
    Find the hottest and coldest spots in the frame.

    Args:
        frame: Thermal frame (2D numpy array)
        num_spots: Number of hot/cold spots to find

    Returns:
        tuple: (hot_spots, cold_spots) - lists of (row, col, value) tuples
    """
    # Flatten frame to 1D for sorting
    flat_frame = frame.flatten()
    flat_indices = np.argsort(flat_frame)

    # Get indices of coldest and hottest pixels
    cold_indices = flat_indices[:num_spots]
    hot_indices = flat_indices[-num_spots:][::-1]  # Reverse for descending order

    # Convert flat indices to 2D coordinates
    rows, cols = frame.shape
    hot_spots = []
    for idx in hot_indices:
        row = idx // cols
        col = idx % cols
        hot_spots.append((row, col, frame[row, col]))

    cold_spots = []
    for idx in cold_indices:
        row = idx // cols
        col = idx % cols
        cold_spots.append((row, col, frame[row, col]))

    return hot_spots, cold_spots


def calculate_thermal_stats(frame):
    """
    Calculate thermal statistics for the frame.

    Args:
        frame: Thermal frame (2D numpy array)

    Returns:
        dict: Dictionary of thermal statistics
    """
    return {
        'min': float(frame.min()),
        'max': float(frame.max()),
        'mean': float(frame.mean()),
        'median': float(np.median(frame)),
        'std': float(frame.std()),
        'range': float(frame.max() - frame.min()),
    }


def create_histogram(frame, bins=50):
    """
    Create histogram of thermal values.

    Args:
        frame: Thermal frame (2D numpy array)
        bins: Number of histogram bins

    Returns:
        tuple: (hist, bin_edges)
    """
    hist, bin_edges = np.histogram(frame.flatten(), bins=bins)
    return hist, bin_edges


def visualize_thermal_analysis(frame_raw, frame_normalized):
    """
    Create visualization of thermal analysis with OpenCV.

    Args:
        frame_raw: Raw thermal frame
        frame_normalized: Normalized frame (0-255)
    """
    if not CV2_AVAILABLE:
        return

    # Create display frame with colormap
    colored = cv2.applyColorMap(frame_normalized, cv2.COLORMAP_JET)
    display = cv2.resize(colored, (640, 480), interpolation=cv2.INTER_NEAREST)

    # Find hot and cold spots
    hot_spots, cold_spots = find_hot_cold_spots(frame_raw, num_spots=3)

    # Draw markers for hot spots (red circles)
    for row, col, value in hot_spots:
        # Scale coordinates to display size
        x = int(col * 640 / frame_raw.shape[1])
        y = int(row * 480 / frame_raw.shape[0])
        cv2.circle(display, (x, y), 10, (0, 0, 255), 2)
        cv2.putText(display, "HOT", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 2)

    # Draw markers for cold spots (blue circles)
    for row, col, value in cold_spots:
        x = int(col * 640 / frame_raw.shape[1])
        y = int(row * 480 / frame_raw.shape[0])
        cv2.circle(display, (x, y), 10, (255, 0, 0), 2)
        cv2.putText(display, "COLD", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 2)

    # Calculate and display statistics
    stats = calculate_thermal_stats(frame_raw)
    info_text = [
        f"Min: {stats['min']:.0f}",
        f"Max: {stats['max']:.0f}",
        f"Mean: {stats['mean']:.0f}",
        f"Range: {stats['range']:.0f}",
    ]

    # Draw statistics overlay
    y_offset = 30
    for i, text in enumerate(info_text):
        cv2.putText(display, text, (10, y_offset + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, text, (10, y_offset + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow('Thermal Analysis', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_thermal_report(frame_raw, frame_normalized):
    """
    Print detailed thermal analysis report.

    Args:
        frame_raw: Raw thermal frame
        frame_normalized: Normalized frame
    """
    print("\n" + "=" * 60)
    print("THERMAL ANALYSIS REPORT")
    print("=" * 60)

    # Basic statistics
    stats = calculate_thermal_stats(frame_raw)
    print("\nThermal Statistics (Raw Values):")
    print(f"  Minimum:     {stats['min']:.2f}")
    print(f"  Maximum:     {stats['max']:.2f}")
    print(f"  Mean:        {stats['mean']:.2f}")
    print(f"  Median:      {stats['median']:.2f}")
    print(f"  Std Dev:     {stats['std']:.2f}")
    print(f"  Range:       {stats['range']:.2f}")

    # Hot and cold spots
    hot_spots, cold_spots = find_hot_cold_spots(frame_raw, num_spots=5)

    print("\nHottest Spots:")
    for i, (row, col, value) in enumerate(hot_spots, 1):
        print(f"  {i}. Position ({row}, {col}): {value:.2f}")

    print("\nColdest Spots:")
    for i, (row, col, value) in enumerate(cold_spots, 1):
        print(f"  {i}. Position ({row}, {col}): {value:.2f}")

    # Histogram analysis
    hist, bin_edges = create_histogram(frame_raw, bins=10)
    print("\nTemperature Distribution (10 bins):")
    for i in range(len(hist)):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        bar = '#' * int(hist[i] / hist.max() * 40)
        print(f"  {bin_start:7.1f} - {bin_end:7.1f}: {bar} ({hist[i]})")

    print("\n" + "=" * 60)


def main():
    print("FLIR Lepton 3.5 - Thermal Analysis Example")
    print("=" * 60)

    try:
        # Initialize camera
        with FLIRLepton35() as camera:
            print("✓ Camera initialized")
            print("  Capturing thermal frame...")

            # Capture both raw and normalized frames
            frame_raw = camera.capture_frame(normalize=False)
            frame_normalized = camera.capture_frame(normalize=True)

            if frame_raw is None or frame_normalized is None:
                print("✗ Failed to capture frame")
                return 1

            print("✓ Frame captured successfully")

            # Print detailed report
            print_thermal_report(frame_raw, frame_normalized)

            # Visual analysis (if OpenCV available)
            if CV2_AVAILABLE:
                print("\nPress any key to close visualization...")
                visualize_thermal_analysis(frame_raw, frame_normalized)
            else:
                print("\nNote: Install OpenCV for visual analysis:")
                print("  pip install opencv-python")

    except LeptonError as e:
        print(f"✗ Lepton Error: {e}")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
