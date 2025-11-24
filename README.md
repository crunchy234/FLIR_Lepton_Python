# FLIR Lepton 3.5 Python Library

A Python library for interfacing with the FLIR Lepton 3.5 thermal camera on Raspberry Pi 5 and NVIDIA Jetson Orin Nano.

## Features

- **Easy-to-use Python API** for thermal image capture
- **SPI communication** for high-speed image data transfer
- **Platform detection** - automatically configures for Raspberry Pi 5 or Jetson Orin Nano
- **Raw thermal data access** - get 16-bit thermal values
- **Normalized image output** - automatic scaling to 0-255 for easy display
- **Stream support** - generator-based continuous frame capture
- **Context manager support** - proper resource cleanup
- **Error handling** - custom exceptions for better debugging

## Hardware Specifications

The FLIR Lepton 3.5 thermal camera module:
- **Resolution:** 160x120 pixels
- **Bit Depth:** 14-bit thermal data
- **Frame Rate:** Up to 8.7 Hz
- **Temperature Range:**
  - High Gain: -10°C to 140°C
  - Low Gain: -10°C to 400°C
- **Interface:** SPI (video data) + I2C (control)
- **Field of View:** 57° (H) × 44° (V)
- **Thermal Sensitivity:** <50 mK

## Installation

### Prerequisites

1. **Enable SPI on your device:**

   **Raspberry Pi 5:**
   ```bash
   sudo raspi-config
   # Navigate to: Interface Options -> SPI -> Enable
   ```

   **Jetson Orin Nano:**
   ```bash
   # SPI is typically enabled by default
   # Verify with: ls /dev/spidev*
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Hardware Connection

Connect the FLIR Lepton 3.5 to your device:

| Lepton Pin | Function | RPi 5 Pin | Jetson Pin |
|------------|----------|-----------|------------|
| 1          | GND      | GND       | GND        |
| 2          | CS       | GPIO 8    | SPI0_CS0   |
| 3          | MISO     | GPIO 9    | SPI0_MISO  |
| 4          | CLK      | GPIO 11   | SPI0_CLK   |
| 5          | VIN      | 3.3V      | 3.3V       |
| 6          | SDA      | GPIO 2    | I2C0_SDA   |
| 7          | SCL      | GPIO 3    | I2C0_SCL   |

## Quick Start

### Basic Usage

```python
from flir_lepton import FLIRLepton35

# Initialize camera
with FLIRLepton35() as camera:
    # Capture a single frame
    frame = camera.capture_frame()

    if frame is not None:
        print(f"Captured frame: {frame.shape}")
        print(f"Temperature range: {frame.min()} - {frame.max()}")
```

### Continuous Frame Capture

```python
from flir_lepton import FLIRLepton35

with FLIRLepton35() as camera:
    # Get frame stream
    for frame in camera.get_frame_stream():
        # Process each frame
        print(f"Frame shape: {frame.shape}")

        # Your processing code here
        # Break after 10 frames for this example
        if some_condition:
            break
```

### Display Thermal Video (with OpenCV)

```python
import cv2
from flir_lepton import FLIRLepton35

with FLIRLepton35() as camera:
    for frame in camera.get_frame_stream(normalize=True):
        # Apply colormap for better visualization
        colored_frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        # Resize for better viewing
        display_frame = cv2.resize(colored_frame, (640, 480))

        # Display
        cv2.imshow('FLIR Lepton 3.5', display_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
```

### Get Raw Temperature Data

```python
from flir_lepton import FLIRLepton35

with FLIRLepton35() as camera:
    # Get raw 16-bit thermal values
    thermal_data = camera.get_temperature_data()

    if thermal_data is not None:
        print(f"Raw thermal values: {thermal_data.dtype}")
        print(f"Min: {thermal_data.min()}, Max: {thermal_data.max()}")
```

## API Reference

### FLIRLepton35 Class

#### Constructor

```python
FLIRLepton35(spi_bus=0, spi_device=0, spi_speed_hz=20000000, auto_detect_platform=True)
```

**Parameters:**
- `spi_bus` (int): SPI bus number (default: 0)
- `spi_device` (int): SPI device number (default: 0)
- `spi_speed_hz` (int): SPI speed in Hz (default: 20 MHz)
- `auto_detect_platform` (bool): Automatically detect platform (default: True)

#### Methods

##### capture_frame(normalize=True)

Capture a single thermal frame.

**Parameters:**
- `normalize` (bool): If True, normalize to 0-255 (uint8). If False, return raw 16-bit data.

**Returns:**
- `numpy.ndarray`: Thermal image of shape (120, 160)

##### get_frame_stream(normalize=True)

Generator that yields frames continuously.

**Parameters:**
- `normalize` (bool): If True, normalize frames to 0-255

**Yields:**
- `numpy.ndarray`: Thermal images

##### get_temperature_data()

Get raw temperature data from the camera.

**Returns:**
- `numpy.ndarray`: Raw 16-bit thermal values

##### close()

Close SPI connection and cleanup resources.

##### get_platform_info() (static)

Get information about the current platform.

**Returns:**
- `dict`: Platform information

### Convenience Functions

#### create_camera(spi_bus=0, spi_device=0, spi_speed_hz=20000000)

Create and initialize a camera instance.

**Returns:**
- `FLIRLepton35`: Initialized camera object

#### test_camera_connection()

Test if the camera is accessible.

**Returns:**
- `bool`: True if camera works, False otherwise

## Error Handling

The library provides custom exceptions:

```python
from flir_lepton import LeptonError, LeptonCommunicationError, LeptonConfigurationError

try:
    with FLIRLepton35() as camera:
        frame = camera.capture_frame()
except LeptonCommunicationError as e:
    print(f"Communication error: {e}")
except LeptonConfigurationError as e:
    print(f"Configuration error: {e}")
except LeptonError as e:
    print(f"General Lepton error: {e}")
```

## Troubleshooting

### "Failed to initialize SPI"

1. Ensure SPI is enabled on your device
2. Check that you have permissions: `sudo usermod -a -G spi,gpio $USER`
3. Verify SPI device exists: `ls -l /dev/spidev*`

### "Failed to read SPI packet"

1. Check hardware connections
2. Verify the camera has stable power (3.3V)
3. Try reducing SPI speed: `FLIRLepton35(spi_speed_hz=10000000)`

### No frames captured / All zeros

1. Allow camera boot time (1-2 seconds after power on)
2. Camera may be performing calibration (wait a few seconds)
3. Check that the camera is properly seated on the connector

### Permission denied errors

Run with sudo or add your user to the required groups:
```bash
sudo usermod -a -G spi,gpio,i2c $USER
# Log out and back in for changes to take effect
```

## Platform-Specific Notes

### Raspberry Pi 5

- SPI is available at `/dev/spidev0.0`
- Default SPI speed of 20 MHz works well
- Enable SPI via `raspi-config`

### Jetson Orin Nano

- SPI is available at `/dev/spidev0.0` and `/dev/spidev0.1`
- May need to configure device tree for SPI
- Check `/proc/device-tree/` for SPI configuration

## Testing the Library

Run the library directly to test:

```bash
python flir_lepton.py
```

This will:
1. Display platform information
2. Initialize the camera
3. Capture a test frame
4. Display frame statistics

## Future Enhancements

- I2C/CCI interface for camera control (gain, calibration, etc.)
- Absolute temperature calculation with calibration
- AGC (Automatic Gain Control) configuration
- FFC (Flat Field Correction) control
- Radiometric mode support
- Video recording capabilities

## License

MIT License - feel free to use in your projects!

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## References

- [FLIR Lepton Datasheet](https://www.flir.com/products/lepton/)
- [Lepton Engineering Datasheet](https://lepton.flir.com/)
- [SPI Protocol Documentation](https://www.kernel.org/doc/Documentation/spi/)

## Support

For issues and questions:
- Check the Troubleshooting section above
- Review the example scripts
- Open an issue on the repository

---

**Note:** This library provides the basic interface for image capture. Converting raw sensor values to absolute temperature requires additional calibration data specific to your camera module.
