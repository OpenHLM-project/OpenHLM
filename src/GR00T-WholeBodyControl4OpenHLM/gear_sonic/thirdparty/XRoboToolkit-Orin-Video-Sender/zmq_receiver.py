#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import zmq
import cv2
import numpy as np
import struct
import time

class ZMQRawImageReceiver:
    def __init__(self, server_ip="192.168.200.112", port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{server_ip}:{port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        
        self.running = True
        print(f"[ZMQRawReceiver] Connected to tcp://{server_ip}:{port}")
    
    def receive_and_display(self):
        """接收原始图像并显示"""
        print("[ZMQRawReceiver] Starting to receive raw images...")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        last_fps_time = time.time()
        fps_counter = 0
        
        try:
            while self.running:
                try:
                    # 接收消息
                    message = self.socket.recv(zmq.NOBLOCK)
                    
                    if len(message) < 12:  # 至少需要12字节的头部
                        continue
                    
                    # 解析头部信息
                    width = struct.unpack('i', message[0:4])[0]
                    height = struct.unpack('i', message[4:8])[0]
                    channels = struct.unpack('i', message[8:12])[0]
                    
                    # 计算期望的图像数据大小
                    expected_data_size = width * height * channels
                    actual_data_size = len(message) - 12
                    
                    if actual_data_size != expected_data_size:
                        print(f"[Warning] Size mismatch: expected {expected_data_size}, got {actual_data_size}")
                        continue
                    
                    # 提取图像数据
                    image_data = message[12:]
                    
                    # 转换为numpy数组
                    if channels == 4:  # BGRA
                        image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))
                        # 转换为BGR用于显示
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                    elif channels == 3:  # BGR
                        image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 3))
                    else:
                        print(f"[Warning] Unsupported channel count: {channels}")
                        continue
                    
                    # 调整大小以便显示 (原图太大)
                    display_image = cv2.resize(image, (1280, 360))
                    
                    # 添加信息文字
                    fps_counter += 1
                    current_time = time.time()
                    if current_time - last_fps_time >= 1.0:
                        fps = fps_counter / (current_time - last_fps_time)
                        fps_counter = 0
                        last_fps_time = current_time
                    else:
                        fps = 0
                    
                    # 显示信息
                    info_text = f"Frame {frame_count} | {width}x{height}x{channels} | FPS: {fps:.1f}"
                    cv2.putText(display_image, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 显示图像
                    cv2.imshow('ZED Raw Stream', display_image)
                    
                    # 处理按键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # 保存当前帧
                        filename = f"zed_frame_{frame_count:06d}.jpg"
                        cv2.imwrite(filename, image)
                        print(f"[Saved] {filename}")
                    
                    frame_count += 1
                    
                except zmq.Again:
                    # 没有消息，继续
                    time.sleep(0.001)
                    continue
                    
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            self.close()
            print(f"\n[ZMQRawReceiver] Received {frame_count} frames")
    
    def process_image(self, image):
        """
        在这里添加你的图像处理代码
        image: numpy数组，BGR格式
        返回: 处理后的图像
        """
        # 示例：边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 可以在这里添加你的AI模型
        # result = your_model.predict(image)
        # processed = draw_results(image, result)
        
        return edges_colored
    
    def receive_and_process(self):
        """接收图像并进行处理"""
        print("[ZMQRawReceiver] Starting to receive and process raw images...")
        print("Press 'q' to quit, 'p' to toggle processing")
        
        frame_count = 0
        processing_enabled = True
        last_fps_time = time.time()
        fps_counter = 0
        
        try:
            while self.running:
                try:
                    message = self.socket.recv(zmq.NOBLOCK)
                    
                    if len(message) < 12:
                        continue
                    
                    # 解析图像
                    width = struct.unpack('i', message[0:4])[0]
                    height = struct.unpack('i', message[4:8])[0]
                    channels = struct.unpack('i', message[8:12])[0]
                    
                    expected_data_size = width * height * channels
                    if len(message) - 12 != expected_data_size:
                        continue
                    
                    image_data = message[12:]
                    
                    if channels == 4:
                        image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                    elif channels == 3:
                        image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 3))
                    else:
                        continue
                    
                    # 处理图像
                    if processing_enabled:
                        processed = self.process_image(image)
                        display_image = cv2.resize(processed, (1280, 360))
                        window_title = 'ZED Processed Stream'
                    else:
                        display_image = cv2.resize(image, (1280, 360))
                        window_title = 'ZED Raw Stream'
                    
                    # FPS计算
                    fps_counter += 1
                    current_time = time.time()
                    if current_time - last_fps_time >= 1.0:
                        fps = fps_counter / (current_time - last_fps_time)
                        fps_counter = 0
                        last_fps_time = current_time
                    else:
                        fps = 0
                    
                    # 添加信息
                    status = "Processing ON" if processing_enabled else "Processing OFF"
                    info_text = f"Frame {frame_count} | FPS: {fps:.1f} | {status}"
                    cv2.putText(display_image, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow(window_title, display_image)
                    
                    # 处理按键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        processing_enabled = not processing_enabled
                        print(f"[Toggle] Processing {'enabled' if processing_enabled else 'disabled'}")
                    
                    frame_count += 1
                    
                except zmq.Again:
                    time.sleep(0.001)
                    continue
                    
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            self.close()
    
    def close(self):
        """关闭连接"""
        self.socket.close()
        self.context.term()

def main():
    SERVER_IP = "192.168.123.164"  # 改成你的ZED设备IP
    PORT = 5555
    
    receiver = ZMQRawImageReceiver(SERVER_IP, PORT)
    
    print("ZMQ Raw Image Receiver")
    print("Choose mode:")
    print("1. Simple display")
    print("2. Display with image processing")
    
    try:
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == "2":
            receiver.receive_and_process()
        else:
            receiver.receive_and_display()
            
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()
        print("Disconnected")

if __name__ == "__main__":
    main() 