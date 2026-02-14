#!/usr/bin/env python3
"""
RTSP Stream Example: Stream thermal video from FLIR Lepton 3.5 over RTSP

This example demonstrates how to:
1. Capture frames from the Lepton 3.5
2. Process frames (colormap, resize)
3. Serve the video stream over RTSP using GStreamer

System Requirements (Raspberry Pi):
    sudo apt-get update
    sudo apt-get install python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-rtsp-server-1.0

Python Requirements:
    pip install opencv-python numpy
"""

import sys
import time
import threading
import numpy as np
import cv2
from flir_lepton import FLIRLepton35, LeptonError

# Try to import GStreamer components
try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstRtspServer', '1.0')
    from gi.repository import Gst, GstRtspServer, GObject, GLib
except ImportError:
    print("Error: GStreamer dependencies missing.")
    print("Please install: sudo apt-get install python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-rtsp-server-1.0")
    sys.exit(1)

# Initialize GStreamer
Gst.init(None)

class FrameProducer:
    """Captures frames in a background thread so GStreamer callbacks never block."""

    def __init__(self, camera):
        self.camera = camera
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            frame = self.camera.capture_frame(normalize=True)
            if frame is not None:
                colored = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                resized = cv2.resize(colored, (640, 480), interpolation=cv2.INTER_NEAREST)
                with self.lock:
                    self.latest_frame = resized
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.latest_frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)


class LeptonRtspFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, frame_producer, **properties):
        super(LeptonRtspFactory, self).__init__(**properties)
        self.frame_producer = frame_producer
        self.fps = 9
        self.duration = int(Gst.SECOND / self.fps)
        self.launch_string = (
            '( appsrc name=source is-live=true block=true format=GST_FORMAT_TIME '
            'caps=video/x-raw,format=BGR,width=640,height=480,framerate={}/1 ! '
            'videoconvert ! video/x-raw,format=I420 ! '
            'openh264enc complexity=low bitrate=500000 ! '
            'video/x-h264,profile=baseline ! '
            'h264parse ! '
            'rtph264pay config-interval=1 name=pay0 )'
        ).format(self.fps)

    def on_need_data(self, src, length, context):
        try:
            display_frame = self.frame_producer.get_frame()

            if display_frame is None:
                display_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            data = display_frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)

            buf.duration = self.duration
            timestamp = context['frame_number'] * self.duration
            buf.pts = buf.dts = int(timestamp)
            buf.offset = context['frame_number']
            context['frame_number'] += 1

            retval = src.emit('push-buffer', buf)
            if retval != Gst.FlowReturn.OK and retval != Gst.FlowReturn.FLUSHING:
                print(f"Error pushing buffer: {retval}")

        except Exception as e:
            print(f"Error in on_need_data: {e}")

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        context = {'frame_number': 0}
        appsrc = rtsp_media.get_element().get_by_name('source')
        appsrc.connect('need-data', self.on_need_data, context)


class GstServer:
    def __init__(self, frame_producer, port="8554", path="/thermal"):
        self.server = GstRtspServer.RTSPServer.new()
        self.server.set_service(port)
        self.server.set_address("0.0.0.0")

        self.factory = LeptonRtspFactory(frame_producer)
        self.factory.set_shared(True)

        self.server.get_mount_points().add_factory(path, self.factory)
        self.server.attach(None)

        print(f"RTSP Stream available at: rtsp://localhost:{port}{path}")


def main():
    print("FLIR Lepton 3.5 - RTSP Stream Server")
    print("=" * 50)

    try:
        camera = FLIRLepton35(vsync_gpio=17, reset_gpio=27)
        print("✓ Camera initialized")

        # Start background frame capture
        frame_producer = FrameProducer(camera)
        print("✓ Frame capture thread started")

        # Start RTSP Server
        server = GstServer(frame_producer)

        # Start GLib MainLoop
        loop = GLib.MainLoop()
        print("Press Ctrl+C to stop the server")

        try:
            loop.run()
        except KeyboardInterrupt:
            print("\nStopping server...")
            loop.quit()
        finally:
            frame_producer.stop()
            camera.close()

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
