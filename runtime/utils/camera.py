from __future__ import annotations

import time

import cv2

from lane.geometry import RAW_H, RAW_W


class LiveCamera:
    """주행 frame을 BGR 1296x972로 읽어오는 얇은 wrapper."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.backend = str(self.cfg.get("backend", "picamera2"))
        self.width = int(self.cfg.get("width", RAW_W))
        self.height = int(self.cfg.get("height", RAW_H))
        self.force_resize = bool(self.cfg.get("force_resize_to_raw", True))
        self.color_mode = str(self.cfg.get("color_mode", "bgr")).lower()
        self.rotate_180 = bool(self.cfg.get("rotate_180", False))
        self.flip_horizontal = bool(self.cfg.get("flip_horizontal", False))
        self.flip_vertical = bool(self.cfg.get("flip_vertical", False))
        self.picam2 = None
        self.cap = None

        if self.backend == "picamera2":
            from picamera2 import Picamera2

            self.picam2 = Picamera2()
            cam_cfg = self.picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam2.configure(cam_cfg)
            self.picam2.start()
            time.sleep(float(self.cfg.get("startup_sleep_sec", 0.5)))
            print(
                f"[camera] picamera2 {self.width}x{self.height} "
                f"color={self.color_mode} rot180={self.rotate_180}"
            )
        elif self.backend == "opencv":
            index = int(self.cfg.get("opencv_index", 0))
            self.cap = cv2.VideoCapture(index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open OpenCV camera index {index}")
            print(f"[camera] opencv index={index} {self.width}x{self.height}")
        else:
            raise ValueError(f"Unsupported camera backend: {self.backend}")

    def _frame_to_bgr(self, frame):
        if self.color_mode in {"bgr", "bgr888", "as_is", "as-is", "none"}:
            return frame.copy()
        if self.color_mode in {"rgb", "rgb888", "rgb_to_bgr"}:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        raise ValueError("Unsupported camera color_mode. Use 'bgr' or 'rgb_to_bgr'.")

    def _apply_geometry(self, bgr):
        if self.rotate_180:
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        if self.flip_horizontal:
            bgr = cv2.flip(bgr, 1)
        if self.flip_vertical:
            bgr = cv2.flip(bgr, 0)
        return bgr

    def read_bgr(self):
        if self.picam2 is not None:
            frame = self.picam2.capture_array()
            bgr = self._frame_to_bgr(frame)
        else:
            ok, bgr = self.cap.read()
            if not ok or bgr is None:
                raise RuntimeError("OpenCV camera read failed")

        bgr = self._apply_geometry(bgr)
        if self.force_resize and (bgr.shape[0] != RAW_H or bgr.shape[1] != RAW_W):
            bgr = cv2.resize(bgr, (RAW_W, RAW_H), interpolation=cv2.INTER_LINEAR)
        return bgr

    def close(self):
        if self.picam2 is not None:
            self.picam2.stop()
        if self.cap is not None:
            self.cap.release()
