#!/usr/bin/env python3
"""
FLIR Lepton 3.5 People Detection and Counting System

This script demonstrates:
- Importing and using the flir_lepton library
- Real-time thermal video streaming
- Detecting people based on thermal signatures
- Drawing bounding boxes around detected people
- Tracking and counting people over time

The system uses thermal blob detection, shape analysis, and tracking
algorithms optimized for thermal imaging.

Requirements:
    pip install opencv-python numpy scipy
"""

import cv2
import numpy as np
import time
from collections import deque
from typing import List, Tuple, Optional
import argparse

# Import our FLIR Lepton library
from flir_lepton import FLIRLepton35, LeptonError


class Person:
    """Represents a detected person with tracking information"""

    def __init__(self, person_id: int, bbox: Tuple[int, int, int, int], centroid: Tuple[int, int]):
        self.id = person_id
        self.bbox = bbox  # (x, y, w, h)
        self.centroid = centroid  # (x, y)
        self.history = deque(maxlen=30)  # Position history for tracking
        self.history.append(centroid)
        self.frames_missing = 0
        self.confirmed = False  # Becomes True after N consistent detections

    def update(self, bbox: Tuple[int, int, int, int], centroid: Tuple[int, int]):
        """Update person's position"""
        self.bbox = bbox
        self.centroid = centroid
        self.history.append(centroid)
        self.frames_missing = 0
        if len(self.history) >= 5:
            self.confirmed = True

    def mark_missing(self):
        """Mark that this person was not detected in current frame"""
        self.frames_missing += 1


class ThermalPeopleDetector:
    """
    Detects people in thermal images using blob detection and shape analysis.
    """

    def __init__(self,
                 min_area: int = 100,
                 max_area: int = 4000,
                 min_temp_threshold: int = 20,
                 min_aspect_ratio: float = 0.3,
                 max_aspect_ratio: float = 4.0):
        """
        Initialize the people detector.

        Args:
            min_area: Minimum blob area in pixels
            max_area: Maximum blob area in pixels
            min_temp_threshold: Minimum temperature threshold for detection
            min_aspect_ratio: Minimum aspect ratio for human-like shapes
            max_aspect_ratio: Maximum aspect ratio for human-like shapes
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_temp_threshold = min_temp_threshold
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )

    def detect_people(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int]]]:
        """
        Detect people in a thermal frame.

        Args:
            frame: Normalized thermal frame (0-255, uint8)

        Returns:
            List of (bbox, centroid) tuples where:
                bbox: (x, y, width, height)
                centroid: (cx, cy)
        """
        detections = []

        # Step 1: Threshold to find hot regions (potential people)
        # People are typically warmer than background
        _, thresh = cv2.threshold(frame, self.min_temp_threshold, 255, cv2.THRESH_BINARY)

        # Step 2: Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Step 3: Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 4: Filter contours based on size and shape
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by aspect ratio (people are usually taller than wide in thermal)
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2

            detections.append(((x, y, w, h), (cx, cy)))

        return detections


class PeopleTracker:
    """
    Tracks detected people across frames and maintains count.
    """

    def __init__(self, max_missing_frames: int = 10, max_distance: float = 50.0):
        """
        Initialize the tracker.

        Args:
            max_missing_frames: Max frames before considering a person gone
            max_distance: Maximum distance for associating detections
        """
        self.people = []  # List of tracked Person objects
        self.next_id = 1
        self.max_missing_frames = max_missing_frames
        self.max_distance = max_distance
        self.total_count = 0  # Total unique people seen
        self.current_count = 0  # Currently visible people

    def update(self, detections: List[Tuple[Tuple[int, int, int, int], Tuple[int, int]]]):
        """
        Update tracker with new detections.

        Args:
            detections: List of (bbox, centroid) tuples
        """
        # Mark all existing people as missing initially
        for person in self.people:
            person.mark_missing()

        # Match detections to existing people
        if len(self.people) > 0 and len(detections) > 0:
            # Create distance matrix
            detection_centroids = np.array([d[1] for d in detections])
            person_centroids = np.array([p.centroid for p in self.people])

            # Calculate distances
            distances = np.linalg.norm(
                detection_centroids[:, np.newaxis] - person_centroids,
                axis=2
            )

            # Associate detections to people (greedy matching)
            used_detections = set()
            used_people = set()

            # Sort by distance and match
            for _ in range(min(len(detections), len(self.people))):
                min_idx = np.argmin(distances)
                det_idx = min_idx // len(self.people)
                person_idx = min_idx % len(self.people)

                if distances[det_idx, person_idx] < self.max_distance:
                    if det_idx not in used_detections and person_idx not in used_people:
                        # Match found
                        bbox, centroid = detections[det_idx]
                        self.people[person_idx].update(bbox, centroid)
                        used_detections.add(det_idx)
                        used_people.add(person_idx)
                        distances[det_idx, :] = np.inf
                        distances[:, person_idx] = np.inf
                else:
                    break

            # Create new people for unmatched detections
            for det_idx in range(len(detections)):
                if det_idx not in used_detections:
                    bbox, centroid = detections[det_idx]
                    new_person = Person(self.next_id, bbox, centroid)
                    self.people.append(new_person)
                    self.next_id += 1
                    self.total_count += 1

        else:
            # No existing people, all detections are new
            for bbox, centroid in detections:
                new_person = Person(self.next_id, bbox, centroid)
                self.people.append(new_person)
                self.next_id += 1
                self.total_count += 1

        # Remove people who have been missing too long
        self.people = [p for p in self.people if p.frames_missing < self.max_missing_frames]

        # Update current count (only confirmed people)
        self.current_count = sum(1 for p in self.people if p.confirmed)

    def get_tracked_people(self) -> List[Person]:
        """Get list of currently tracked people"""
        return [p for p in self.people if p.confirmed]


class ThermalPeopleCountingSystem:
    """
    Complete system for thermal people detection, tracking, and counting.
    """

    def __init__(self,
                 detector: ThermalPeopleDetector,
                 tracker: PeopleTracker,
                 display_scale: int = 4):
        """
        Initialize the system.

        Args:
            detector: People detector instance
            tracker: People tracker instance
            display_scale: Scale factor for display window
        """
        self.detector = detector
        self.tracker = tracker
        self.display_scale = display_scale
        self.fps_counter = deque(maxlen=30)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Process a thermal frame.

        Args:
            frame: Normalized thermal frame (0-255, uint8)

        Returns:
            Tuple of (annotated_frame, current_count, total_count)
        """
        # Detect people
        detections = self.detector.detect_people(frame)

        # Update tracker
        self.tracker.update(detections)

        # Create visualization
        annotated = self.visualize(frame)

        return annotated, self.tracker.current_count, self.tracker.total_count

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """
        Create annotated visualization of the frame.

        Args:
            frame: Original thermal frame

        Returns:
            Annotated frame with bounding boxes and info
        """
        # Apply colormap
        colored = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        # Scale up for display
        height, width = colored.shape[:2]
        display = cv2.resize(
            colored,
            (width * self.display_scale, height * self.display_scale),
            interpolation=cv2.INTER_NEAREST
        )

        # Draw bounding boxes and IDs for tracked people
        for person in self.tracker.get_tracked_people():
            x, y, w, h = person.bbox

            # Scale coordinates
            x *= self.display_scale
            y *= self.display_scale
            w *= self.display_scale
            h *= self.display_scale

            # Draw bounding box
            color = (0, 255, 0)  # Green for tracked people
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

            # Draw ID label
            label = f"Person {person.id}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(display, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(display, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Draw tracking trail
            if len(person.history) > 1:
                points = [(int(px * self.display_scale), int(py * self.display_scale))
                         for px, py in person.history]
                for i in range(1, len(points)):
                    cv2.line(display, points[i - 1], points[i], color, 1)

        # Draw info overlay
        info_y = 30
        info_text = [
            f"Current Count: {self.tracker.current_count}",
            f"Total Seen: {self.tracker.total_count}",
            f"Tracking: {len(self.tracker.get_tracked_people())} people",
        ]

        # Calculate FPS
        if len(self.fps_counter) > 1:
            fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
            info_text.append(f"FPS: {fps:.1f}")

        for i, text in enumerate(info_text):
            y_pos = info_y + i * 30
            # White text with black outline
            cv2.putText(display, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4)
            cv2.putText(display, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return display

    def run(self, camera: FLIRLepton35):
        """
        Run the people counting system.

        Args:
            camera: Initialized FLIRLepton35 camera instance
        """
        print("Starting Thermal People Counting System...")
        print("Controls:")
        print("  Q - Quit")
        print("  R - Reset counter")
        print("  S - Save screenshot")
        print()

        window_name = "Thermal People Counter"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        screenshot_count = 0

        try:
            for frame in camera.get_frame_stream(normalize=True):
                # Record timestamp for FPS
                self.fps_counter.append(time.time())

                # Process frame
                annotated, current, total = self.process_frame(frame)

                # Display
                cv2.imshow(window_name, annotated)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q'):
                    print("\nShutting down...")
                    break
                elif key == ord('r') or key == ord('R'):
                    print("\nResetting counter...")
                    self.tracker = PeopleTracker(
                        max_missing_frames=self.tracker.max_missing_frames,
                        max_distance=self.tracker.max_distance
                    )
                elif key == ord('s') or key == ord('S'):
                    filename = f"thermal_people_count_{screenshot_count:04d}.png"
                    cv2.imwrite(filename, annotated)
                    print(f"Screenshot saved: {filename}")
                    screenshot_count += 1

        except KeyboardInterrupt:
            print("\n✓ Interrupted by user")
        finally:
            cv2.destroyAllWindows()
            print(f"\nSession Summary:")
            print(f"  Total people detected: {self.tracker.total_count}")
            print(f"  Final count: {self.tracker.current_count}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="FLIR Lepton 3.5 Thermal People Detection and Counting System"
    )
    parser.add_argument("--min-area", type=int, default=100,
                       help="Minimum blob area for detection (default: 100)")
    parser.add_argument("--max-area", type=int, default=4000,
                       help="Maximum blob area for detection (default: 4000)")
    parser.add_argument("--min-temp", type=int, default=20,
                       help="Minimum temperature threshold (default: 20)")
    parser.add_argument("--max-missing", type=int, default=10,
                       help="Max frames before person considered gone (default: 10)")
    parser.add_argument("--scale", type=int, default=4,
                       help="Display scale factor (default: 4)")
    parser.add_argument("--spi-bus", type=int, default=0,
                       help="SPI bus number (default: 0)")
    parser.add_argument("--spi-device", type=int, default=0,
                       help="SPI device number (default: 0)")

    args = parser.parse_args()

    print("=" * 60)
    print("FLIR Lepton 3.5 - Thermal People Counting System")
    print("=" * 60)
    print()

    # Check OpenCV availability
    if cv2.__version__:
        print(f"✓ OpenCV version: {cv2.__version__}")
    else:
        print("✗ OpenCV not available")
        return 1

    # Initialize detector
    detector = ThermalPeopleDetector(
        min_area=args.min_area,
        max_area=args.max_area,
        min_temp_threshold=args.min_temp,
    )
    print("✓ People detector initialized")

    # Initialize tracker
    tracker = PeopleTracker(max_missing_frames=args.max_missing)
    print("✓ People tracker initialized")

    # Initialize system
    system = ThermalPeopleCountingSystem(detector, tracker, display_scale=args.scale)
    print("✓ Counting system initialized")
    print()

    # Initialize camera
    try:
        with FLIRLepton35(spi_bus=args.spi_bus, spi_device=args.spi_device) as camera:
            print("✓ Camera initialized")
            print()

            # Run the system
            system.run(camera)

    except LeptonError as e:
        print(f"✗ Camera Error: {e}")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
