#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import struct
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from rich import print
import zmq


class HeadZMQClient:
    """ZMQ subscriber for the stitched head camera stream."""

    def __init__(
        self,
        server_address: str = "127.0.0.1",
        port: int = 5555,
        image_shape: Optional[Tuple[int, int, int]] = None,
        image_show: bool = False,
    ) -> None:
        self.server_address = server_address
        self.port = port
        self.image_shape = image_shape
        self.image_show = image_show
        self.running = True

        self._latest_image: Optional[np.ndarray] = None
        self._latest_timestamp_ms: Optional[int] = None
        self._lock = threading.Lock()

    def _decode_message(self, message: bytes) -> Optional[np.ndarray]:
        if len(message) < 12:
            return None

        width, height, channels = struct.unpack("iii", message[:12])
        payload = message[12:]
        expected_size = width * height * channels
        if len(payload) != expected_size:
            return None

        try:
            raw_image = np.frombuffer(payload, dtype=np.uint8).reshape((height, width, channels))
        except ValueError:
            return None

        if channels == 4:
            image = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2BGR)
        elif channels == 3:
            image = raw_image
        else:
            return None

        if self.image_shape is not None:
            target_h, target_w = self.image_shape[:2]
            if image.shape[:2] != (target_h, target_w):
                image = cv2.resize(image, (target_w, target_h))

        return image

    def _set_latest_image(self, image: np.ndarray) -> None:
        with self._lock:
            self._latest_image = image.copy()
            self._latest_timestamp_ms = int(time.time() * 1000)

    def get_latest_image(self, copy: bool = True) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_image is None:
                return None
            return self._latest_image.copy() if copy else self._latest_image

    def get_latest_timestamp_ms(self) -> Optional[int]:
        with self._lock:
            return self._latest_timestamp_ms

    def stop(self) -> None:
        self.running = False

    def _show_image(self, image: np.ndarray) -> None:
        display_img = cv2.resize(image, (1280, 360))
        cv2.imshow("HeadZMQClient - RGB", display_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.running = False

    def _close(self, socket: zmq.Socket, context: zmq.Context) -> None:
        socket.close()
        context.term()
        if self.image_show:
            cv2.destroyAllWindows()
        print("[HeadZMQClient] Closed.")

    def receive_process(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(f"tcp://{self.server_address}:{self.port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.setsockopt(zmq.CONFLATE, 1)
        print(f"[HeadZMQClient] Subscribed to tcp://{self.server_address}:{self.port}. Waiting for data...")

        try:
            while self.running:
                try:
                    message = socket.recv(zmq.NOBLOCK)
                    image = self._decode_message(message)
                    if image is None:
                        continue

                    self._set_latest_image(image)
                    if self.image_show:
                        self._show_image(image)
                except zmq.Again:
                    time.sleep(0.001)
        except KeyboardInterrupt:
            print("[HeadZMQClient] Interrupted by user.")
        except Exception as exc:
            print(f"[HeadZMQClient] Error: {exc}")
        finally:
            self._close(socket, context)


class WristZMQClient:
    """ZMQ subscriber for the stitched wrist camera JPEG stream."""

    def __init__(
        self,
        server_address: str = "127.0.0.1",
        port: int = 5554,
        image_shape: Optional[Tuple[int, int, int]] = None,
        image_show: bool = False,
    ) -> None:
        self.server_address = server_address
        self.port = port
        self.connect_addr = f"tcp://{server_address}:{port}"
        self.image_shape = image_shape
        self.image_show = image_show
        self.running = True

        self._latest_image: Optional[np.ndarray] = None
        self._latest_timestamp_ms: Optional[int] = None
        self._lock = threading.Lock()

    def _decode_message(self, message: bytes) -> Optional[np.ndarray]:
        if len(message) < 12:
            return None

        width, height, jpeg_len = struct.unpack("iii", message[:12])
        jpeg_data = message[12:]
        if len(jpeg_data) != jpeg_len:
            return None

        image = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None

        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))

        if self.image_shape is not None:
            target_h, target_w = self.image_shape[:2]
            if image.shape[:2] != (target_h, target_w):
                image = cv2.resize(image, (target_w, target_h))

        return image

    def _set_latest_image(self, image: np.ndarray) -> None:
        with self._lock:
            self._latest_image = image.copy()
            self._latest_timestamp_ms = int(time.time() * 1000)

    def get_latest_image(self, copy: bool = True) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_image is None:
                return None
            return self._latest_image.copy() if copy else self._latest_image

    def get_latest_timestamp_ms(self) -> Optional[int]:
        with self._lock:
            return self._latest_timestamp_ms

    def stop(self) -> None:
        self.running = False

    def _show_image(self, image: np.ndarray) -> None:
        display_img = cv2.resize(image, (1280, 360))
        cv2.imshow("WristZMQClient", display_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.running = False

    def _close(self, socket: zmq.Socket, context: zmq.Context) -> None:
        socket.close()
        context.term()
        if self.image_show:
            cv2.destroyAllWindows()
        print("[WristZMQClient] Closed.")

    def receive_process(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(self.connect_addr)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.setsockopt(zmq.CONFLATE, 1)
        print(f"[WristZMQClient] Subscribed to {self.connect_addr}")

        try:
            while self.running:
                try:
                    message = socket.recv(zmq.NOBLOCK)
                    image = self._decode_message(message)
                    if image is None:
                        continue

                    self._set_latest_image(image)
                    if self.image_show:
                        self._show_image(image)
                except zmq.Again:
                    time.sleep(0.001)
        except KeyboardInterrupt:
            print("[WristZMQClient] Interrupted by user.")
        except Exception as exc:
            print(f"[WristZMQClient] Error: {exc}")
        finally:
            self._close(socket, context)


if __name__ == "__main__":
    head_client = HeadZMQClient(
        server_address="192.168.123.164",
        port=5555,
        image_shape=(360, 1280, 3),
        image_show=True,
    )
    wrist_client = WristZMQClient(
        server_address="192.168.123.164",
        port=5554,
        image_shape=(360, 1280, 3),
        image_show=True,
    )
    head_client.receive_process()
    wrist_client.receive_process()
    head_client.stop()
    wrist_client.stop()