import time
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst


class Camera:
    def __init__(
        self,
        camera_id=0,
        width=640,
        height=480,
        fps=30,
        rtsp_url="rtsp://127.0.0.1:8554/cam",
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.rtsp_url = rtsp_url

        self.pipeline = None
        self.appsink = None
        self.opened = False

        Gst.init(None)

    def open(self):
        device = self._camera_device()

        pipeline_desc = f"""
v4l2src device={device}
! video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1
! videoconvert
! tee name=t

t. ! queue leaky=downstream max-size-buffers=1
   ! videoconvert
   ! video/x-raw,format=BGR,width={self.width},height={self.height}
   ! appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true

t. ! queue
   ! videoconvert
   ! mpph264enc
   ! h264parse config-interval=1
   ! rtspclientsink location={self.rtsp_url}
"""

        self.pipeline = Gst.parse_launch(pipeline_desc)
        self.appsink = self.pipeline.get_by_name("appsink")

        if self.appsink is None:
            raise RuntimeError("Cannot get appsink from GStreamer pipeline")

        ret = self.pipeline.set_state(Gst.State.PLAYING)

        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start GStreamer camera pipeline")

        self.opened = True
        time.sleep(0.5)

    def read(self):
        if not self.opened or self.appsink is None:
            raise RuntimeError("Camera is not opened")

        sample = self.appsink.emit("try-pull-sample", Gst.SECOND)

        if sample is None:
            return None

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)

        width = structure.get_value("width")
        height = structure.get_value("height")

        success, map_info = buffer.map(Gst.MapFlags.READ)

        if not success:
            return None

        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame.reshape((height, width, 3))
            return frame.copy()
        finally:
            buffer.unmap(map_info)

    def release(self):
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
            self.appsink = None
            self.opened = False

    def _camera_device(self):
        if isinstance(self.camera_id, int):
            return f"/dev/video{self.camera_id}"

        return str(self.camera_id)