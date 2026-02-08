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
    PACKET_SIZE_UINT16 = 82  # PACKET_SIZE / 2 = 164 / 2

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

    def _resync_vospi(self):
        """
        Force a VoSPI resync by deasserting CS for at least 185ms.
        The Lepton requires this to reset its VoSPI state machine.
        """
        self.spi.close()
        time.sleep(0.2)
        self.spi.open(self.spi_bus, self.spi_device)
        self.spi.max_speed_hz = self.spi_speed_hz
        self.spi.mode = 3
        self.spi.bits_per_word = 8

    def _read_packet_raw(self) -> bytearray:
        """Read a single 164-byte VoSPI packet."""
        return bytearray(self.spi.readbytes(self.PACKET_SIZE))

    def _capture_frame_raw(self, max_retries: int = 50, debug: bool = False) -> Optional[np.ndarray]:
        """
        Capture a raw frame from the Lepton camera.

        Strategy:
        1. Resync VoSPI on startup and after repeated failures
        2. Read one packet at a time to find packet 0 (sync)
        3. Once synced, read remaining 59 packets of the segment
        4. Collect all 4 segments to assemble a full frame
        """
        for retry in range(max_retries):
            try:
                # Resync periodically to recover from persistent desync
                if retry % 10 == 0:
                    if debug and retry > 0:
                        print(f"Resync VoSPI (retry {retry})")
                    self._resync_vospi()
                    time.sleep(0.05)

                segments = {}
                segments_seen = set()
                discard_count = 0
                max_discards = 600  # ~2.5 full frames worth of packets

                while len(segments_seen) < 4 and discard_count < max_discards:
                    # Step 1: Read packets one at a time until we find packet 0
                    pkt = self._read_packet_raw()
                    discard_count += 1

                    # Skip discard packets
                    if (pkt[0] & 0x0F) == 0x0F:
                        continue

                    # Not packet 0 — skip and keep looking
                    if pkt[1] != 0:
                        continue

                    # Found packet 0! Now read the remaining 59 packets
                    segment_packets = [pkt]
                    valid = True
                    segment_id = -1

                    for i in range(1, self.PACKETS_PER_SEGMENT):
                        next_pkt = self._read_packet_raw()
                        discard_count += 1

                        if next_pkt[1] != i:
                            if debug:
                                print(f"Sync lost at packet {i}: expected {i}, got {next_pkt[1]}")
                            valid = False
                            break

                        segment_packets.append(next_pkt)

                        # Extract segment ID from packet 20
                        if i == 20:
                            segment_id = (next_pkt[0] >> 4) & 0x07

                    if not valid:
                        continue

                    # Validate segment ID (must be 1-4 for Lepton 3.x)
                    if segment_id < 1 or segment_id > 4:
                        continue

                    if segment_id not in segments_seen:
                        segments[segment_id] = segment_packets
                        segments_seen.add(segment_id)
                        if debug:
                            print(f"Captured segment {segment_id}")

                if len(segments_seen) == 4:
                    # Assemble the 160x120 frame from the 4 segments
                    image_data = np.zeros((self.IMAGE_HEIGHT, self.IMAGE_WIDTH), dtype=np.uint16)

                    for sid in range(1, 5):
                        for p_idx, pkt in enumerate(segments[sid]):
                            row = (p_idx // 2) + (sid - 1) * 30
                            col_start = 80 * (p_idx % 2)

                            packet_data = pkt[self.PACKET_HEADER_SIZE:]
                            row_pixels = np.frombuffer(bytes(packet_data), dtype='>u2')
                            image_data[row, col_start:col_start + 80] = row_pixels

                    return image_data
                elif debug:
                    print(f"Failed to capture all segments. Seen: {segments_seen}")

            except Exception as e:
                if debug:
                    print(f"Error during capture attempt: {e}")
                time.sleep(0.005)
                continue

        return None

    def capture_frame(self, normalize: bool = True, debug: bool = False) -> Optional[np.ndarray]:
        """
        Capture a thermal frame from the camera.

        Args:
            normalize: If True, normalize the output to 0-255 range (uint8)
                      If False, return raw 16-bit thermal data
            debug: If True, enable debug logging during capture

        Returns:
            numpy array containing thermal image data
            - If normalize=True: shape (120, 160) dtype uint8
            - If normalize=False: shape (120, 160) dtype uint16
            Returns None if capture fails
        """
        frame = self._capture_frame_raw(debug=debug)

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
            # Use higher max_retries and debug mode for the CLI test
            frame = camera.capture_frame(debug=True)

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
