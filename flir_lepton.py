
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

import platform
import time
from typing import Optional

import numpy as np
import spidev


class LeptonError(Exception):
    """Base exception for Lepton camera errors"""
    pass


class LeptonCommunicationError(LeptonError):
    """Raised when communication with the camera fails"""
    pass


class LeptonConfigurationError(LeptonError):
    """Raised when camera configuration fails"""
    pass


class LeptonI2C:
    """
    I2C/CCI interface for FLIR Lepton camera control commands.

    The Lepton uses I2C for its Command and Control Interface (CCI).
    This allows sending commands like reboot, FFC, and status queries.

    CCI Protocol:
        - Register 0x0002: Status register (16-bit)
        - Register 0x0004: Command ID register (16-bit)
        - Register 0x0006: Data length register (16-bit)
        - Registers 0x0008+: Data registers
    """

    # Default I2C address for Lepton
    LEPTON_I2C_ADDRESS = 0x2A

    # CCI register addresses
    REG_STATUS = 0x0002
    REG_COMMAND_ID = 0x0004
    REG_DATA_LENGTH = 0x0006
    REG_DATA_0 = 0x0008

    # CCI command IDs
    CMD_OEM_REBOOT = 0x4842       # OEM Run Reboot
    CMD_SYS_RUN_FFC = 0x0242      # SYS Run FFC Normalization
    CMD_SYS_STATUS = 0x0204       # SYS Get Status
    CMD_OEM_GPIO_MODE_SET = 0x4854  # OEM Set GPIO Mode
    CMD_OEM_GPIO_MODE_GET = 0x4855  # OEM Get GPIO Mode
    CMD_OEM_GPIO_VSYNC_PHASE_DELAY_SET = 0x4858  # OEM Set GPIO VSync Phase Delay

    # Status bits
    STATUS_BUSY_BIT = 0x01
    STATUS_BOOT_MODE_BIT = 0x02
    STATUS_BOOT_STATUS_BIT = 0x04

    def __init__(self, i2c_bus: int = 1):
        self.i2c_bus = i2c_bus
        self.bus = None
        self._init_i2c()

    def _init_i2c(self):
        """Initialize the I2C bus using smbus2."""
        try:
            import smbus2
            self.bus = smbus2.SMBus(self.i2c_bus)
            print(f"I2C initialized on bus {self.i2c_bus}")
        except ImportError:
            print("Warning: smbus2 not installed. I2C commands (reboot/FFC) unavailable.")
            print("Install with: pip install smbus2")
            self.bus = None
        except Exception as e:
            print(f"Warning: Could not open I2C bus {self.i2c_bus}: {e}")
            self.bus = None

    def _write_register(self, reg: int, value: int):
        """Write a 16-bit value to a CCI register."""
        msb = (value >> 8) & 0xFF
        lsb = value & 0xFF
        self.bus.write_i2c_block_data(self.LEPTON_I2C_ADDRESS, (reg >> 8) & 0xFF,
                                      [reg & 0xFF, msb, lsb])

    def _write_command(self, command_id: int, data_length: int = 0):
        """Write a CCI command using word-addressable register writes."""
        # Write data length register
        self.bus.write_word_data(self.LEPTON_I2C_ADDRESS, self.REG_DATA_LENGTH,
                                 self._swap16(data_length))
        # Write command ID register (triggers the command)
        self.bus.write_word_data(self.LEPTON_I2C_ADDRESS, self.REG_COMMAND_ID,
                                 self._swap16(command_id))

    def _read_status(self) -> int:
        """Read the CCI status register."""
        raw = self.bus.read_word_data(self.LEPTON_I2C_ADDRESS, self.REG_STATUS)
        return self._swap16(raw)

    @staticmethod
    def _swap16(val: int) -> int:
        """Swap bytes in a 16-bit word (SMBus byte order quirk)."""
        return ((val & 0xFF) << 8) | ((val >> 8) & 0xFF)

    def _wait_busy(self, timeout: float = 5.0) -> bool:
        """Wait for camera to finish processing a command."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                status = self._read_status()
                if not (status & self.STATUS_BUSY_BIT):
                    return True
            except Exception:
                pass  # Camera may be unresponsive during boot
            time.sleep(0.01)
        return False

    @property
    def available(self) -> bool:
        return self.bus is not None

    def reboot(self) -> bool:
        """
        Send OEM Reboot command to the Lepton via I2C/CCI.
        This resets the Lepton's internal state including VoSPI.
        The camera takes ~5 seconds to boot after reboot.

        Returns:
            True if command was sent, False if I2C unavailable
        """
        if not self.available:
            return False
        try:
            self._wait_busy()
            self._write_command(self.CMD_OEM_REBOOT)
            print("Lepton reboot command sent via I2C")
            return True
        except Exception as e:
            print(f"I2C reboot command failed: {e}")
            return False

    def run_ffc(self) -> bool:
        """
        Trigger Flat Field Correction (FFC / shutter calibration).

        Returns:
            True if command was sent, False if I2C unavailable
        """
        if not self.available:
            return False
        try:
            self._wait_busy()
            self._write_command(self.CMD_SYS_RUN_FFC)
            print("FFC command sent via I2C")
            return True
        except Exception as e:
            print(f"I2C FFC command failed: {e}")
            return False

    def configure_gpio_vsync(self) -> bool:
        """
        Configure GPIO3 as VSYNC output.

        Note: The Lepton SDK configures GPIO3 for VSYNC by default.
        This command sets the GPIO mode to VSYNC (mode 5).

        Returns:
            True if configuration succeeded, False otherwise
        """
        if not self.available:
            return False
        try:
            self._wait_busy()

            # GPIO Mode values (from Lepton SDK):
            # 0 = LEP_OEM_GPIO_MODE_GPIO
            # 1 = LEP_OEM_GPIO_MODE_I2C_MASTER
            # 2 = LEP_OEM_GPIO_MODE_SPI_MASTER_VLB_DATA
            # 3 = LEP_OEM_GPIO_MODE_SPIO_MASTER_REG_DATA
            # 4 = LEP_OEM_GPIO_MODE_SPI_SLAVE_VLB_DATA
            # 5 = LEP_OEM_GPIO_MODE_VSYNC (what we want)
            gpio_mode = 5  # VSYNC mode

            # Data format: 32-bit enum value (stored as 2x 16-bit words)
            # SDK sends this as a 32-bit little-endian value

            # Write to data registers as two 16-bit words
            # Word 0 (lower 16 bits): mode value
            # Word 1 (upper 16 bits): 0 (padding for 32-bit enum)
            self.bus.write_word_data(self.LEPTON_I2C_ADDRESS, self.REG_DATA_0,
                                    self._swap16(gpio_mode & 0xFFFF))
            self.bus.write_word_data(self.LEPTON_I2C_ADDRESS, self.REG_DATA_0 + 2,
                                    self._swap16(0))

            # Write data length (2 words = 4 bytes)
            self.bus.write_word_data(self.LEPTON_I2C_ADDRESS, self.REG_DATA_LENGTH,
                                    self._swap16(2))

            # Write command ID to trigger the operation
            self.bus.write_word_data(self.LEPTON_I2C_ADDRESS, self.REG_COMMAND_ID,
                                    self._swap16(self.CMD_OEM_GPIO_MODE_SET))

            self._wait_busy()
            print("GPIO3 configured as VSYNC output")
            return True
        except Exception as e:
            print(f"Failed to configure GPIO as VSYNC: {e}")
            return False

    def close(self):
        if self.bus is not None:
            self.bus.close()
            self.bus = None


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

    # Timing constants (seconds)
    SEGMENT_PERIOD = 0.0265  # ~26.5ms per segment
    FRAME_PERIOD = 0.106  # ~106ms per frame (4 segments)
    RESYNC_DELAY = 0.185  # Minimum CS deassert time for VoSPI reset

    def __init__(self,
                 spi_bus: int = 0,
                 spi_device: int = 0,
                 spi_speed_hz: int = 20000000,
                 i2c_bus: int = 1,
                 vsync_gpio: int = None,
                 reset_gpio: int = None,
                 auto_detect_platform: bool = True):
        """
        Initialize the FLIR Lepton 3.5 interface.

        Args:
            spi_bus: SPI bus number (default: 0)
            spi_device: SPI device number (default: 0)
            spi_speed_hz: SPI communication speed in Hz (default: 20 MHz)
            i2c_bus: I2C bus number for CCI commands (default: 1)
            vsync_gpio: GPIO pin number for VSYNC signal (optional, improves sync)
            reset_gpio: GPIO pin connected to Lepton RESET_L (optional, enables hard reset)
            auto_detect_platform: Automatically detect and configure for platform

        Raises:
            LeptonError: If initialization fails
        """
        # Initialize ALL attributes first so close()/del never hits AttributeError
        self.spi_bus = spi_bus
        self.spi_device = spi_device
        self.spi_speed_hz = spi_speed_hz
        self.spi = None
        self.i2c = None
        self.platform_name = None
        self.vsync_gpio = vsync_gpio
        self.reset_gpio = reset_gpio
        self._gpio = None
        self._vsync_line = None
        self._reset_line = None

        # Sync health tracking
        self._consecutive_failures = 0
        self._total_resyncs = 0
        self._last_frame_time = 0
        self._frames_captured = 0

        if auto_detect_platform:
            self._detect_platform()

        # Initialize I2C for CCI commands (reboot, FFC)
        self.i2c = LeptonI2C(i2c_bus)

        # Configure VSYNC on Lepton's GPIO3 if VSYNC is enabled
        if vsync_gpio is not None and self.i2c.available:
            self.i2c.configure_gpio_vsync()

        self._initialize_spi()
        self._initialize_gpio()

        # Run SPI health check after initialization
        self._check_spi_health()

    def _detect_platform(self) -> None:
        """Detect the platform (Raspberry Pi 5 or Jetson Orin Nano)."""
        system = platform.system()
        machine = platform.machine()

        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().lower()
                if 'raspberry pi 5' in model:
                    self.platform_name = 'Raspberry Pi 5'
                elif 'raspberry pi' in model:
                    self.platform_name = f'Raspberry Pi (Model: {model.strip()})'
        except FileNotFoundError:
            pass

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

        if self.platform_name is None:
            self.platform_name = f'{system} on {machine}'

        print(f"Detected platform: {self.platform_name}")

    def _initialize_spi(self) -> None:
        """Initialize SPI communication with the Lepton camera."""
        try:
            self.spi = spidev.SpiDev()
            self.spi.open(self.spi_bus, self.spi_device)
            self.spi.max_speed_hz = self.spi_speed_hz
            self.spi.mode = 3  # SPI Mode 3 (CPOL=1, CPHA=1)
            self.spi.bits_per_word = 8
            print(f"SPI initialized: Bus {self.spi_bus}, Device {self.spi_device}, Speed {self.spi_speed_hz} Hz")
        except Exception as e:
            raise LeptonCommunicationError(f"Failed to initialize SPI: {str(e)}")

    # noinspection PyTypeChecker
    def _initialize_gpio(self) -> None:
        """Initialize GPIO for VSYNC and/or RESET_L if configured (Pi 5 compatible via gpiod)."""
        if self.vsync_gpio is None and self.reset_gpio is None:
            return
        try:
            import gpiod
            from gpiod.line import Direction, Edge, Value

            chip = gpiod.Chip('/dev/gpiochip4')

            line_configs = {}

            if self.vsync_gpio is not None:
                line_configs[self.vsync_gpio] = gpiod.LineSettings(
                    direction=Direction.INPUT,
                    edge_detection=Edge.RISING,
                )

            if self.reset_gpio is not None:
                line_configs[self.reset_gpio] = gpiod.LineSettings(
                    direction=Direction.OUTPUT,
                    output_value=Value.ACTIVE,  # HIGH — device operates
                )

            request = chip.request_lines(
                consumer="flir_lepton",
                config=line_configs,
            )

            self._gpio = request  # gpiod.LineRequest object

            if self.vsync_gpio is not None:
                self._vsync_line = self.vsync_gpio
                print(f"VSYNC configured on GPIO {self.vsync_gpio}")

            if self.reset_gpio is not None:
                self._reset_line = self.reset_gpio
                print(f"RESET_L configured and verified on GPIO {self.reset_gpio}")

        except ImportError:
            print("ERROR: gpiod not installed. GPIO features (VSYNC/RESET) unavailable.")
            print("  Install with: pip install gpiod")
        except Exception as e:
            print(f"ERROR: GPIO init failed: {e}")
            print("  Verify /dev/gpiochip4 exists: ls /dev/gpiochip*")

    def _open_spi(self):
        """Open and configure SPI (helper to avoid repetition)."""
        self.spi.open(self.spi_bus, self.spi_device)
        self.spi.max_speed_hz = self.spi_speed_hz
        self.spi.mode = 3
        self.spi.bits_per_word = 8

    def _resync_vospi(self):
        """
        Force a VoSPI resync by deasserting CS for at least 185ms.
        The Lepton requires this to reset its VoSPI state machine.

        CRITICAL: Do NOT read packets after reopening - let the capture
        function handle finding the sync point. The Lepton streams continuously.
        """
        self.spi.close()
        time.sleep(0.25)  # >185ms required by spec, use 250ms for safety
        self._open_spi()
        # Give camera a moment to stabilize after CS reassert
        time.sleep(0.01)
        self._total_resyncs += 1

    def _hard_reset(self) -> bool:
        """Perform a hardware reset by pulling RESET_L low."""
        if self._reset_line is not None and self._gpio is not None:
            print("Performing hardware reset via RESET_L pin...")
            self._gpio.output(self._reset_line, self._gpio.LOW)
            time.sleep(0.015)
            self._gpio.output(self._reset_line, self._gpio.HIGH)
            time.sleep(5.0)
            self._resync_vospi()
            return True
        return False

    def _soft_reboot(self) -> bool:
        """
        Perform a soft reboot via I2C CCI command.
        Falls back to hardware reset if I2C unavailable.
        
        CRITICAL: After reboot, we must do a full VoSPI resync (CS deassert
        for >185ms) before starting reads. Without this, the SPI reads will
        be misaligned with the Lepton's VoSPI stream.
        """
        if self.i2c is not None and self.i2c.available:
            # Close SPI before reboot to avoid contention
            self.spi.close()
            success = self.i2c.reboot()
            if success:
                print("Waiting for Lepton to reboot (~5s)...")
                time.sleep(5.0)
                # Reopen SPI then immediately do a VoSPI resync
                self._open_spi()
                # The critical missing piece: resync VoSPI AFTER reboot
                self._resync_vospi()
                time.sleep(0.1)  # Extra settling time
                self._consecutive_failures = 0
                return True
            # Reopen SPI even if reboot failed
            self._open_spi()

        return self._hard_reset()

    def _wait_for_vsync(self, timeout_ms: int = 200) -> bool:
        """Wait for VSYNC rising edge if VSYNC GPIO is configured."""
        if self._vsync_line is None or self._gpio is None:
            return False
        try:
            # Wait for the rising edge with timeout
            channel = self._gpio.wait_for_edge(self._vsync_line, self._gpio.RISING, timeout=timeout_ms)
            return channel is not None
        except Exception:
            pass
        return False

    def _read_packet_raw(self) -> bytearray:
        """Read a single 164-byte VoSPI packet."""
        data = bytearray(self.spi.readbytes(self.PACKET_SIZE))
        return data

    def _check_spi_health(self) -> dict:
        """
        Check if SPI is receiving valid data from the camera.
        Returns a dict with diagnostic information.
        """
        print("Checking SPI health...")
        all_zeros = 0
        all_ones = 0
        discard_packets = 0
        valid_looking = 0

        for i in range(100):
            pkt = self._read_packet_raw()
            data_sum = sum(pkt)

            if data_sum == 0:
                all_zeros += 1
            elif data_sum == (164 * 255):
                all_ones += 1
            elif (pkt[0] & 0x0F) == 0x0F:
                discard_packets += 1
            elif pkt[1] < 60:  # Valid packet number
                valid_looking += 1

        result = {
            'all_zeros': all_zeros,
            'all_ones': all_ones,
            'discard_packets': discard_packets,
            'valid_looking': valid_looking,
            'total': 100
        }

        print(f"SPI Health Check Results:")
        print(f"  All zeros: {all_zeros}/100")
        print(f"  All ones: {all_ones}/100")
        print(f"  Discard packets: {discard_packets}/100")
        print(f"  Valid-looking: {valid_looking}/100")

        if all_zeros > 90:
            print("  ⚠️  WARNING: SPI receiving mostly zeros - check hardware connections!")
            print("     - Verify MISO pin is connected")
            print("     - Check camera power supply")
            print("     - Verify SPI bus/device numbers")

        return result

    def _dump_raw_packets(self, count: int = 10):
        """Debug helper: dump raw packet headers to diagnose SPI alignment."""
        print(f"--- Raw packet dump ({count} packets) ---")
        for i in range(count):
            pkt = self._read_packet_raw()
            id_nibble = pkt[0] & 0x0F
            pkt_num = pkt[1]
            is_discard = (id_nibble == 0x0F)
            seg_id = (pkt[0] >> 4) & 0x07 if pkt_num == 20 else -1
            # Check if packet data is all zeros
            data_sum = sum(pkt[4:])
            status = 'DISCARD' if is_discard else 'VALID' if pkt_num < 60 else 'INVALID'
            print(f"  pkt[{i:3d}]: hdr=[{pkt[0]:02x} {pkt[1]:02x} {pkt[2]:02x} {pkt[3]:02x}] "
                  f"id_nibble={id_nibble:x} pkt_num={pkt_num:3d} "
                  f"{status} data_sum={data_sum}")
            if pkt_num == 20 and seg_id > 0:
                print(f"           ^ segment {seg_id}")
        print("--- End dump ---")

    def _capture_frame_raw(self, max_retries: int = 40, debug: bool = False) -> Optional[np.ndarray]:
        """
        Capture a raw frame from the Lepton camera.

        Strategy (improved for long-running stability):
        1. Always resync VoSPI at the start
        2. On sync loss, break out immediately and resync
        3. Escalate: resync → I2C reboot → hardware reset
        4. After reboot, dump raw packets to verify SPI health
        """
        for retry in range(max_retries):
            try:
                # Escalating recovery strategy
                if retry == 0:
                    self._resync_vospi()
                elif retry >= 25:
                    if debug:
                        print(f"Retry {retry}: escalating to soft reboot")
                    self._soft_reboot()
                    time.sleep(0.5)
                    if debug:
                        self._dump_raw_packets(20)
                        # Resync again after the dump reads misaligned things
                        self._resync_vospi()
                elif retry % 5 == 0:
                    if debug:
                        print(f"Retry {retry}: VoSPI resync")
                    self._resync_vospi()

                # If VSYNC is available, wait for frame boundary
                self._wait_for_vsync(timeout_ms=150)

                segments = {}
                segments_seen = set()
                discard_count = 0
                max_discards = 2000  # Increased significantly - need to scan through multiple frames
                sync_loss_count = 0

                # Track what we're seeing
                seen_valid_packets = False

                while len(segments_seen) < 4 and discard_count < max_discards:
                    pkt = self._read_packet_raw()
                    discard_count += 1

                    # Skip discard packets (ID nibble = 0x0F)
                    if (pkt[0] & 0x0F) == 0x0F:
                        if debug and discard_count % 100 == 0:
                            print(f"Skipping discard packets... ({discard_count} packets read)")
                        continue

                    # Check if packet is all zeros (invalid/gap data)
                    if pkt[0] == 0 and pkt[1] == 0 and pkt[2] == 0 and pkt[3] == 0:
                        if debug and discard_count % 100 == 0:
                            print(f"Skipping zero packets... ({discard_count} packets read)")
                        continue

                    # Not packet 0 — skip
                    if pkt[1] != 0:
                        if not seen_valid_packets:
                            seen_valid_packets = True
                            if debug:
                                print(f"Seeing valid packets in stream - continuing search for packet 0...")
                        if debug and discard_count <= 20:
                            print(f"Looking for packet 0, got packet {pkt[1]}: hdr=[{pkt[0]:02x} {pkt[1]:02x} {pkt[2]:02x} {pkt[3]:02x}]")
                        continue

                    # Found packet 0!
                    if debug:
                        print(f"Found packet 0: hdr=[{pkt[0]:02x} {pkt[1]:02x} {pkt[2]:02x} {pkt[3]:02x}]")

                    # Read remaining 59 packets
                    segment_packets = [pkt]
                    valid = True
                    segment_id = -1

                    for i in range(1, self.PACKETS_PER_SEGMENT):
                        next_pkt = self._read_packet_raw()
                        discard_count += 1

                        # Check if this is a discard packet - if so, we've lost sync
                        if (next_pkt[0] & 0x0F) == 0x0F:
                            if debug:
                                print(f"Discard packet at position {i}, resyncing")
                            valid = False
                            sync_loss_count += 1
                            break

                        # Lepton 3.x: Check packet number
                        # Note: packet numbers go 0-59, each represents one packet in sequence
                        expected_pkt_num = i
                        if next_pkt[1] != expected_pkt_num:
                            if debug:
                                id_nibble = next_pkt[0] & 0x0F
                                print(f"Sync lost at packet {i}: expected {expected_pkt_num}, got {next_pkt[1]} (ID nibble: 0x{id_nibble:x}, hdr=[{next_pkt[0]:02x} {next_pkt[1]:02x}])")
                                # Dump next few packets to understand the pattern
                                if i == 1:
                                    print(f"  Dumping next 10 packets to diagnose:")
                                    for j in range(10):
                                        tmp_pkt = self._read_packet_raw()
                                        print(f"    [{j}]: hdr=[{tmp_pkt[0]:02x} {tmp_pkt[1]:02x} {tmp_pkt[2]:02x} {tmp_pkt[3]:02x}]")
                            valid = False
                            sync_loss_count += 1
                            break

                        segment_packets.append(next_pkt)

                        if i == 20:
                            segment_id = (next_pkt[0] >> 4) & 0x07

                    if not valid:
                        # Break immediately on repeated sync loss — need resync
                        if sync_loss_count >= 5:
                            if debug:
                                print(f"Persistent sync loss ({sync_loss_count}x), breaking for resync")
                            break
                        continue

                    if segment_id < 1 or segment_id > 4:
                        continue

                    if segment_id not in segments_seen:
                        segments[segment_id] = segment_packets
                        segments_seen.add(segment_id)
                        if debug:
                            print(f"Captured segment {segment_id}")
                        sync_loss_count = 0

                if len(segments_seen) == 4:
                    # Assemble the 160x120 frame
                    image_data = np.zeros((self.IMAGE_HEIGHT, self.IMAGE_WIDTH), dtype=np.uint16)

                    for sid in range(1, 5):
                        for p_idx, pkt in enumerate(segments[sid]):
                            row = (p_idx // 2) + (sid - 1) * 30
                            col_start = 80 * (p_idx % 2)

                            packet_data = pkt[self.PACKET_HEADER_SIZE:]
                            row_pixels = np.frombuffer(bytes(packet_data), dtype='>u2')
                            image_data[row, col_start:col_start + 80] = row_pixels

                    self._consecutive_failures = 0
                    self._frames_captured += 1
                    self._last_frame_time = time.monotonic()
                    return image_data
                else:
                    if debug:
                        print(f"Failed to capture all segments. Seen: {segments_seen}")
                    # Don't resync here - let the retry logic handle it

            except Exception as e:
                if debug:
                    print(f"Error during capture attempt: {e}")
                time.sleep(0.005)
                continue

        # All retries exhausted
        self._consecutive_failures += 1

        if self._consecutive_failures >= 3:
            print(f"Persistent sync failure ({self._consecutive_failures} consecutive). "
                  f"Triggering Lepton reboot...")
            self._soft_reboot()
            self._consecutive_failures = 0

        return None

    def capture_frame(self, normalize: bool = True, debug: bool = False) -> Optional[np.ndarray]:
        """
        Capture a thermal frame from the camera.

        Args:
            normalize: If True, normalize the output to 0-255 range (uint8)
                      If False, return raw 16-bit thermal data
            debug: If True, enable debug logging during capture

        Returns:
            numpy array containing thermal image data, or None if capture fails
        """
        frame = self._capture_frame_raw(debug=debug)

        if frame is None:
            return None

        if normalize:
            frame_min = frame.min()
            frame_max = frame.max()

            if frame_max > frame_min:
                return ((frame - frame_min) * 255.0 / (frame_max - frame_min)).astype(np.uint8)
            else:
                return np.zeros((self.IMAGE_HEIGHT, self.IMAGE_WIDTH), dtype=np.uint8)
        else:
            return frame

    def get_frame_stream(self, normalize: bool = True):
        """Generator that yields thermal frames continuously."""
        while True:
            frame = self.capture_frame(normalize=normalize)
            if frame is not None:
                yield frame

    def get_temperature_data(self) -> Optional[np.ndarray]:
        """Get raw temperature data (uint16) from the camera."""
        return self.capture_frame(normalize=False)

    def run_ffc(self) -> bool:
        """Trigger Flat Field Correction (shutter calibration) via I2C."""
        if self.i2c is not None:
            return self.i2c.run_ffc()
        return False

    def reboot(self) -> bool:
        """Reboot the Lepton camera. Takes ~5 seconds to come back online."""
        return self._soft_reboot()

    @property
    def sync_stats(self) -> dict:
        """Get synchronization health statistics."""
        return {
            'frames_captured': self._frames_captured,
            'total_resyncs': self._total_resyncs,
            'consecutive_failures': self._consecutive_failures,
            'last_frame_time': self._last_frame_time,
        }

    def close(self) -> None:
        """Close the SPI/I2C connections and cleanup resources."""
        if getattr(self, 'spi', None) is not None:
            self.spi.close()
            self.spi = None
        if getattr(self, 'i2c', None) is not None:
            self.i2c.close()
            self.i2c = None
        if getattr(self, '_gpio', None) is not None:
            try:
                self._gpio.cleanup()
            except Exception:
                pass
            self._gpio = None
            self._reset_line = None
            self._vsync_line = None
        print("Connections closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    @staticmethod
    def get_platform_info() -> dict:
        """Get information about the current platform."""
        info = {
            'system': platform.system(),
            'machine': platform.machine(),
            'platform': platform.platform(),
            'processor': platform.processor(),
        }
        try:
            with open('/proc/device-tree/model', 'r') as f:
                info['device_model'] = f.read().strip()
        except:
            pass
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                info['tegra_release'] = f.read().strip()
        except:
            pass
        return info


# Convenience functions

def create_camera(spi_bus: int = 0,
                  spi_device: int = 0,
                  spi_speed_hz: int = 20000000) -> FLIRLepton35:
    """Create and initialize a Lepton camera instance."""
    return FLIRLepton35(spi_bus=spi_bus,
                        spi_device=spi_device,
                        spi_speed_hz=spi_speed_hz,
                        vsync_gpio=17,
                        reset_gpio=27)


def test_camera_connection() -> bool:
    """Test if the camera can be initialized and accessed."""
    try:
        with FLIRLepton35() as camera:
            frame = camera.capture_frame()
            return frame is not None
    except Exception as e:
        print(f"Camera test failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("FLIR Lepton 3.5 Thermal Camera Library")
    print("=" * 50)
    print()

    print("Platform Information:")
    platform_info = FLIRLepton35.get_platform_info()
    for key, value in platform_info.items():
        print(f"  {key}: {value}")
    print()

    print("Testing camera connection...")
    try:
        with FLIRLepton35(vsync_gpio=17, reset_gpio=27) as camera:
            print("Camera initialized successfully!")
            print()

            print("Capturing test frame...")
            frame = camera.capture_frame(debug=True)

            if frame is not None:
                print(f"Frame captured successfully!")
                print(f"  Shape: {frame.shape}")
                print(f"  Data type: {frame.dtype}")
                print(f"  Min value: {frame.min()}")
                print(f"  Max value: {frame.max()}")
                print(f"  Mean value: {frame.mean():.2f}")
                print(f"  Sync stats: {camera.sync_stats}")
            else:
                print("Failed to capture frame")
                print(f"  Sync stats: {camera.sync_stats}")

    except LeptonError as e:
        print(f"Lepton Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")