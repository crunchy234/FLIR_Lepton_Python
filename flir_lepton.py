"""
FLIR Lepton 3.5 Thermal Camera Interface Library

This library provides a Python interface for the FLIR Lepton 3.5 thermal camera
on Raspberry Pi 5 and NVIDIA Jetson Orin Nano.

The Lepton 3.5 uses:
- SPI for video data transfer
- I2C for Command and Control Interface (CCI)

Hardware Specifications:
- Resolution: 160x120 pixels
- Bit Depth: 14-bit thermal data
- Frame Rate: Up to 8.7 Hz
- Temperature Range: -10°C to 140°C (High Gain), -10°C to 400°C (Low Gain)
"""

import spidev
import time
import numpy as np
from typing import Optional, Tuple
import platform
import os


class LeptonError(Exception):
    """Base exception for Lepton camera errors"""
    pass


class LeptonCommunicationError(LeptonError):
    """Raised when communication with the camera fails"""
    pass


class LeptonConfigurationError(LeptonError):
    """Raised when camera configuration fails"""
    pass


class FLIRLepton35:
    """
    Interface class for FLIR Lepton 3.5 thermal camera.

    This class handles communication with the Lepton 3.5 via SPI for image data
    and provides methods for camera control and configuration.

    Attributes:
        IMAGE_WIDTH (int): Image width in pixels (160)
        IMAGE_HEIGHT (int): Image height in pixels (120)
        PACKET_SIZE (int): Size of each SPI packet in bytes
        PACKETS_PER_SEGMENT (int): Number of packets per segment
        SEGMENTS_PER_FRAME (int): Number of segments per frame
        FRAME_SIZE (int): Total frame size in bytes
    """

    # Lepton 3.5 specifications
    IMAGE_WIDTH = 160
    IMAGE_HEIGHT = 120
    PACKET_SIZE = 164  # 4 bytes header + 160 bytes data
    PACKETS_PER_SEGMENT = 60  # Lepton 3.x uses 4 segments per frame
    SEGMENTS_PER_FRAME = 4
    FRAME_SIZE = PACKET_SIZE * PACKETS_PER_SEGMENT * SEGMENTS_PER_FRAME

    # SPI packet format
    PACKET_HEADER_SIZE = 4
    PACKET_DATA_SIZE = 160

    # Discard packet ID (used for resync)
    DISCARD_PACKET_ID = 0x0F

    def __init__(self,
                 spi_bus: int = 0,
                 spi_device: int = 0,
                 spi_speed_hz: int = 20000000,
                 auto_detect_platform: bool = True):
        """
        Initialize the FLIR Lepton 3.5 interface.

        Args:
            spi_bus: SPI bus number (default: 0)
            spi_device: SPI device number (default: 0)
            spi_speed_hz: SPI communication speed in Hz (default: 20 MHz)
            auto_detect_platform: Automatically detect and configure for platform

        Raises:
            LeptonError: If initialization fails
        """
        self.spi_bus = spi_bus
        self.spi_device = spi_device
        self.spi_speed_hz = spi_speed_hz
        self.spi = None
        self.platform_name = None

        if auto_detect_platform:
            self._detect_platform()

        self._initialize_spi()

    def _detect_platform(self) -> None:
        """
        Detect the platform (Raspberry Pi 5 or Jetson Orin Nano).
        Sets platform-specific configurations.
        """
        system = platform.system()
        machine = platform.machine()

        # Try to detect Raspberry Pi
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().lower()
                if 'raspberry pi 5' in model:
                    self.platform_name = 'Raspberry Pi 5'
                elif 'raspberry pi' in model:
                    self.platform_name = f'Raspberry Pi (Model: {model.strip()})'
        except FileNotFoundError:
            pass

        # Try to detect Jetson
        if self.platform_name is None:
            try:
                with open('/etc/nv_tegra_release', 'r') as f:
                    tegra_info = f.read()
                    if 'orin' in tegra_info.lower():
                        self.platform_name = 'NVIDIA Jetson Orin Nano'
                    else:
                        self.platform_name = f'NVIDIA Jetson (Unknown model)'
            except FileNotFoundError:
                pass

        # Generic Linux fallback
        if self.platform_name is None:
            self.platform_name = f'{system} on {machine}'

        print(f"Detected platform: {self.platform_name}")

    def _initialize_spi(self) -> None:
        """
        Initialize SPI communication with the Lepton camera.

        Raises:
            LeptonCommunicationError: If SPI initialization fails
        """
        try:
            self.spi = spidev.SpiDev()
            self.spi.open(self.spi_bus, self.spi_device)
            self.spi.max_speed_hz = self.spi_speed_hz
            self.spi.mode = 3  # SPI Mode 3 (CPOL=1, CPHA=1)
            self.spi.bits_per_word = 8
            print(f"SPI initialized: Bus {self.spi_bus}, Device {self.spi_device}, Speed {self.spi_speed_hz} Hz")
        except Exception as e:
            raise LeptonCommunicationError(f"Failed to initialize SPI: {str(e)}")

    def _read_packet(self) -> Tuple[int, int, int, bytes]:
        """
        Read a single packet from the Lepton via SPI.

        Returns:
            Tuple of (packet_id, segment_id, packet_number, packet_data)

        Raises:
            LeptonCommunicationError: If packet read fails
        """
        try:
            packet = bytes(self.spi.readbytes(self.PACKET_SIZE))

            # Extract packet number from header
            # Byte 0 contains ID field, Byte 1 contains CRC
            packet_id = packet[0]
            packet_number = packet[1]
            segment_id = (packet_id >> 4) & 0x07

            # Extract data portion (skip 4-byte header)
            packet_data = packet[self.PACKET_HEADER_SIZE:]

            return packet_id, segment_id, packet_number, packet_data
        except Exception as e:
            raise LeptonCommunicationError(f"Failed to read SPI packet: {str(e)}")

    def _capture_frame_raw(self, max_retries: int = 10) -> Optional[np.ndarray]:
        """
        Capture a raw frame from the Lepton camera.

        Args:
            max_retries: Maximum number of frame capture attempts

        Returns:
            numpy array of shape (120, 160) containing 16-bit thermal data,
            or None if capture fails
        """
        for retry in range(max_retries):
            try:
                # Initialize segment buffers
                segments = {
                    1: [None] * self.PACKETS_PER_SEGMENT,
                    2: [None] * self.PACKETS_PER_SEGMENT,
                    3: [None] * self.PACKETS_PER_SEGMENT,
                    4: [None] * self.PACKETS_PER_SEGMENT,
                }
                segments_seen = set()
                discard_count = 0
                packets_read = 0
                max_packets = self.PACKETS_PER_SEGMENT * self.SEGMENTS_PER_FRAME * 5

                # Read packets until we get a complete frame
                while packets_read < max_packets:
                    packet_id, segment_id, packet_number, packet_data = self._read_packet()
                    packets_read += 1

                    # Check for discard packet (resync needed)
                    if (packet_id & 0x0F) == self.DISCARD_PACKET_ID:
                        discard_count += 1
                        if discard_count > 750:  # Timeout after many discards
                            break
                        time.sleep(0.001)  # Short delay for resync
                        continue

                    if segment_id not in segments:
                        continue

                    if packet_number == 0:
                        segments[segment_id] = [None] * self.PACKETS_PER_SEGMENT
                        segments_seen.add(segment_id)

                    if packet_number < self.PACKETS_PER_SEGMENT:
                        segments[segment_id][packet_number] = packet_data

                    if len(segments_seen) == self.SEGMENTS_PER_FRAME:
                        if all(all(p is not None for p in segments[sid]) for sid in segments_seen):
                            break

                # Check if we got a complete frame across all segments
                if len(segments_seen) == self.SEGMENTS_PER_FRAME and all(
                    all(p is not None for p in segments[sid]) for sid in segments_seen
                ):
                    segment_arrays = {}
                    for sid in range(1, self.SEGMENTS_PER_FRAME + 1):
                        segment_bytes = b''.join(segments[sid])
                        segment_array = np.frombuffer(segment_bytes, dtype='>u2')
                        segment_arrays[sid] = segment_array.reshape((self.PACKETS_PER_SEGMENT, 80))

                    top = np.hstack((segment_arrays[1], segment_arrays[2]))
                    bottom = np.hstack((segment_arrays[3], segment_arrays[4]))
                    image_data = np.vstack((top, bottom))

                    return image_data

            except LeptonCommunicationError:
                if retry == max_retries - 1:
                    raise
                time.sleep(0.1)  # Wait before retry

        return None

    def capture_frame(self, normalize: bool = True) -> Optional[np.ndarray]:
        """
        Capture a thermal frame from the camera.

        Args:
            normalize: If True, normalize the output to 0-255 range (uint8)
                      If False, return raw 16-bit thermal data

        Returns:
            numpy array containing thermal image data
            - If normalize=True: shape (120, 160) dtype uint8
            - If normalize=False: shape (120, 160) dtype uint16
            Returns None if capture fails
        """
        frame = self._capture_frame_raw()

        if frame is None:
            return None

        if normalize:
            # Normalize to 0-255 range for display
            frame_min = frame.min()
            frame_max = frame.max()

            if frame_max > frame_min:
                frame_normalized = ((frame - frame_min) * 255.0 / (frame_max - frame_min)).astype(np.uint8)
                return frame_normalized
            else:
                return np.zeros((self.IMAGE_HEIGHT, self.IMAGE_WIDTH), dtype=np.uint8)
        else:
            return frame

    def get_frame_stream(self, normalize: bool = True):
        """
        Generator that yields thermal frames continuously.

        Args:
            normalize: If True, normalize frames to 0-255 range

        Yields:
            numpy arrays containing thermal image data
        """
        while True:
            frame = self.capture_frame(normalize=normalize)
            if frame is not None:
                yield frame

    def get_temperature_data(self) -> Optional[np.ndarray]:
        """
        Get temperature data from the camera.

        Note: Converting raw sensor values to absolute temperature requires
        calibration data and is complex. This returns raw thermal values.
        For absolute temperature, additional calibration is needed.

        Returns:
            numpy array of raw thermal values (uint16)
        """
        return self.capture_frame(normalize=False)

    def close(self) -> None:
        """
        Close the SPI connection and cleanup resources.
        """
        if self.spi is not None:
            self.spi.close()
            self.spi = None
            print("SPI connection closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.close()

    @staticmethod
    def get_platform_info() -> dict:
        """
        Get information about the current platform.

        Returns:
            Dictionary containing platform information
        """
        info = {
            'system': platform.system(),
            'machine': platform.machine(),
            'platform': platform.platform(),
            'processor': platform.processor(),
        }

        # Check for Raspberry Pi
        try:
            with open('/proc/device-tree/model', 'r') as f:
                info['device_model'] = f.read().strip()
        except:
            pass

        # Check for Jetson
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                info['tegra_release'] = f.read().strip()
        except:
            pass

        return info


# Convenience functions for quick access

def create_camera(spi_bus: int = 0,
                  spi_device: int = 0,
                  spi_speed_hz: int = 20000000) -> FLIRLepton35:
    """
    Convenience function to create and initialize a Lepton camera instance.

    Args:
        spi_bus: SPI bus number (default: 0)
        spi_device: SPI device number (default: 0)
        spi_speed_hz: SPI speed in Hz (default: 20 MHz)

    Returns:
        Initialized FLIRLepton35 instance
    """
    return FLIRLepton35(spi_bus=spi_bus,
                        spi_device=spi_device,
                        spi_speed_hz=spi_speed_hz)


def test_camera_connection() -> bool:
    """
    Test if the camera can be initialized and accessed.

    Returns:
        True if camera is accessible, False otherwise
    """
    try:
        with FLIRLepton35() as camera:
            frame = camera.capture_frame()
            return frame is not None
    except Exception as e:
        print(f"Camera test failed: {str(e)}")
        return False


if __name__ == "__main__":
    """Example usage and testing"""
    print("FLIR Lepton 3.5 Thermal Camera Library")
    print("=" * 50)
    print()

    # Display platform information
    print("Platform Information:")
    platform_info = FLIRLepton35.get_platform_info()
    for key, value in platform_info.items():
        print(f"  {key}: {value}")
    print()

    # Test camera connection
    print("Testing camera connection...")
    try:
        with FLIRLepton35() as camera:
            print("Camera initialized successfully!")
            print()

            # Capture a single frame
            print("Capturing test frame...")
            frame = camera.capture_frame()

            if frame is not None:
                print(f"Frame captured successfully!")
                print(f"  Shape: {frame.shape}")
                print(f"  Data type: {frame.dtype}")
                print(f"  Min value: {frame.min()}")
                print(f"  Max value: {frame.max()}")
                print(f"  Mean value: {frame.mean():.2f}")
            else:
                print("Failed to capture frame")

    except LeptonError as e:
        print(f"Lepton Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
