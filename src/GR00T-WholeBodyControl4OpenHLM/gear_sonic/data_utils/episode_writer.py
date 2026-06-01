import os
import cv2
import json
import datetime
import numpy as np
import time
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from rich import print

class EpisodeWriter():
    VIDEO_CODEC = "mp4v"
    VIDEO_EXT = ".mp4"
    CAMERA_KEYS = ("rgb_left", "rgb_right", "wrist_rgb_left", "wrist_rgb_right")

    def __init__(self, task_dir, frequency=30,
                 image_shape=(480, 640, 3),
                 wrist_shape=None,
                 data_keys = None):
        """
        image_shape: [height, width, channel]
        state_shape: [29]
        action_shape: [29]
        """
        print("==> EpisodeWriter initializing...\n")
        self.task_dir = task_dir
        self.frequency = frequency
        self.image_shape = image_shape
        self.wrist_shape = wrist_shape
        self.data_keys = list(data_keys) if data_keys is not None else list(self.CAMERA_KEYS)
        self.video_writers = {}
        self.video_paths = {}
        self.video_frame_counts = {}
        self.video_sizes = {}
        self._resize_warnings = set()
        
        self.data = {}
        self.episode_data = []
        self.item_id = -1
        self.episode_id = -1
        if os.path.exists(self.task_dir):
            self.episode_id = self._get_latest_episode_id()
            print(f"==> task_dir directory already exist, now self.episode_id is:{self.episode_id}\n")
        else:
            os.makedirs(self.task_dir)
            print(f"==> episode directory does not exist, now create one.\n")
        self.data_info()
        self.text_desc()

        self.is_available = True  # Indicates whether the class is available for new operations
        # Initialize the queue and worker thread
        self.item_data_queue = Queue(maxsize=100)
        self.stop_worker = False
        self.need_save = False  # Flag to indicate when save_episode is triggered
        self.worker_thread = Thread(target=self.process_queue)
        self.worker_thread.start()

        print("==> EpisodeWriter initialized successfully.\n")

    def _get_existing_episode_ids(self):
        episode_ids = []
        if not os.path.exists(self.task_dir):
            return episode_ids

        for episode_dir in os.listdir(self.task_dir):
            if not episode_dir.startswith("episode_"):
                continue
            try:
                episode_ids.append(int(episode_dir.split("_")[-1]))
            except ValueError:
                continue
        return sorted(episode_ids)

    def _get_latest_episode_id(self):
        episode_ids = self._get_existing_episode_ids()
        return 0 if len(episode_ids) == 0 else episode_ids[-1]

    def _get_next_episode_id(self):
        episode_ids = self._get_existing_episode_ids()
        return 0 if len(episode_ids) == 0 else episode_ids[-1] + 1

    def data_info(self, version='1.0.0', date=None):
        video_cameras = {}
        for key in self.data_keys:
            shape = self._expected_camera_shape(key)
            if shape is None:
                continue
            video_cameras[key] = {
                "height": int(shape[0]),
                "width": int(shape[1]),
                "channels": int(shape[2]),
                "fps": self.frequency,
                "path": f"videos/{key}{self.VIDEO_EXT}",
            }

        self.info = {
                "version": "1.0.0" if version is None else version, 
                "date": datetime.date.today().strftime('%Y-%m-%d') if date is None else date,
                "author": "Humanoid Loco-Manipulation Co-training Team",
                "image": {"height":self.image_shape[0], "width":self.image_shape[1], "fps":self.frequency},
                "video": {
                    "format": self.VIDEO_EXT.lstrip("."),
                    "codec": self.VIDEO_CODEC,
                    "fps": self.frequency,
                    "cameras": video_cameras,
                },
            }
        if self.wrist_shape is not None:
            self.info["wrist_image"] = {
                "height": self.wrist_shape[0],
                "width": self.wrist_shape[1],
                "fps": self.frequency,
            }
        
    def text_desc(self, goal="pick up the red cup on the table.", 
                  desc="Pick up the cup from the table and place it in another position. The operation should be smooth and the water in the cup should not spill out",
                  steps="step1: searching for cups. step2: go to the target location. step3: pick up the cup"):
        self.text = {
            "goal": goal,
            "desc": desc,
            "steps":steps,
        }
 
    def create_episode(self):
        """
        Create a new episode.
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        Note:
            Once successfully created, this function will only be available again after save_episode complete its save task.
        """
        if not self.is_available:
            print("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False  # Return False if the class is unavailable

        # Reset episode-related data and create necessary directories
        self.item_id = -1
        self.episode_data = []
        self.data = {}
        self.video_writers = {}
        self.video_paths = {}
        self.video_frame_counts = {}
        self.video_sizes = {}
        self._resize_warnings = set()
        self.episode_id = self._get_next_episode_id()
        

        self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        os.makedirs(self.episode_dir, exist_ok=True)
        self.video_dir = os.path.join(self.episode_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)
        for key in self.data_keys:
            if key not in self.CAMERA_KEYS:
                continue
            video_path = os.path.join(self.video_dir, f"{key}{self.VIDEO_EXT}")
            self.video_paths[key] = video_path
            self.video_frame_counts[key] = 0
            print(f"==> {key}_video: {video_path}")

        self.json_path = os.path.join(self.episode_dir, 'data.json')

        self.is_available = False  # After the episode is created, the class is marked as unavailable until the episode is successfully saved
        print(f"==> New episode created: {self.episode_dir}")
        return True  # Return True if the episode is successfully created
        
    def add_item(self, data_dict):
        # Increment the item ID
        self.item_id += 1
        for key in self.CAMERA_KEYS:
            if isinstance(data_dict.get(key), np.ndarray):
                data_dict[key] = data_dict[key].copy()
        # Enqueue the item data
        self.item_data_queue.put(data_dict)

    def process_queue(self):
        while not self.stop_worker or not self.item_data_queue.empty():
            # Process items in the queue
            try:
                item_data = self.item_data_queue.get(timeout=1)
                try:
                    self._process_item_data(item_data)
                except Exception as e:
                    print(f"Error processing item_data (idx={item_data['idx']}): {e}")
                self.item_data_queue.task_done()
            except Empty:
                pass
        
            # Check if save_episode was triggered
            if self.need_save and self.item_data_queue.empty():
                self._save_episode()

    def _process_item_data(self, item_data):
        idx = item_data['idx']

        # vision
        rgb_left = item_data.get('rgb_left', None)
        rgb_right = item_data.get('rgb_right', None)
        wrist_rgb_left = item_data.get('wrist_rgb_left', None)
        wrist_rgb_right = item_data.get('wrist_rgb_right', None)

        # body and hand state
        state_body = item_data.get('state_body', None)
        state_hand_left = item_data.get('state_hand_left', None)
        state_hand_right = item_data.get('state_hand_right', None)

        # body and hand action
        action_body = item_data.get('action_body', None)
        action_hand_left = item_data.get('action_hand_left', None)
        action_hand_right = item_data.get('action_hand_right', None)

        # human data
        human_data = item_data.get('human_data', None)

        # retarget data
        retarget_data = item_data.get('retarget_data', None)

        # low level action
        action_low_level = item_data.get('action_low_level', None)

        # Save camera frames into per-episode videos. Each frame keeps a video
        # reference and the frame index instead of a path to a single image file.
        for key, image in (
            ("rgb_left", rgb_left),
            ("rgb_right", rgb_right),
            ("wrist_rgb_left", wrist_rgb_left),
            ("wrist_rgb_right", wrist_rgb_right),
        ):
            if image is not None and key in self.video_paths:
                item_data[key] = self._write_video_frame(key, image)
            
                
        # state and action are directly saved to the episode_data
        if state_body is not None:
            item_data['state_body'] = state_body
        if state_hand_left is not None:
            item_data['state_hand_left'] = state_hand_left
        if state_hand_right is not None:
            item_data['state_hand_right'] = state_hand_right

        if action_body is not None:
            item_data['action_body'] = action_body
        if action_hand_left is not None:
            item_data['action_hand_left'] = action_hand_left
        if action_hand_right is not None:
            item_data['action_hand_right'] = action_hand_right

        if human_data is not None:
            item_data['human_data'] = human_data
        if retarget_data is not None:
            item_data['retarget_data'] = retarget_data
        if action_low_level is not None:
            item_data['action_low_level'] = action_low_level

        # Save item_data to episode_data
        # Update episode data
        self.episode_data.append(item_data)

        curent_record_time = time.time()
        print(f"==> episode_id:{self.episode_id}  item_id:{self.item_id}  current_time:{curent_record_time}")

    def _expected_camera_shape(self, key):
        if key in ("rgb_left", "rgb_right"):
            if self.image_shape is None:
                return None
            height, width, channels = self.image_shape
            return (height, width // 2, channels) if width % 2 == 0 else self.image_shape
        if key in ("wrist_rgb_left", "wrist_rgb_right"):
            if self.wrist_shape is None:
                return None
            height, width, channels = self.wrist_shape
            return (height, width // 2, channels) if width % 2 == 0 else self.wrist_shape
        return None

    def _open_video_writer(self, key, frame):
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.VIDEO_CODEC)
        writer = cv2.VideoWriter(self.video_paths[key], fourcc, float(self.frequency), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {key}: {self.video_paths[key]}")
        self.video_writers[key] = writer
        self.video_sizes[key] = (width, height)
        return writer

    def _write_video_frame(self, key, frame):
        if key not in self.video_paths:
            return None

        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"{key} frame must have shape (H, W, 3), got {frame.shape}")
        frame = np.ascontiguousarray(frame)

        writer = self.video_writers.get(key)
        if writer is None:
            writer = self._open_video_writer(key, frame)

        width, height = self.video_sizes[key]
        if (frame.shape[1], frame.shape[0]) != (width, height):
            if key not in self._resize_warnings:
                print(f"Warning: resizing {key} frames from {frame.shape[1]}x{frame.shape[0]} to {width}x{height} for video consistency.")
                self._resize_warnings.add(key)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        frame_index = self.video_frame_counts.get(key, 0)
        writer.write(frame)
        self.video_frame_counts[key] = frame_index + 1

        rel_path = str(Path(self.video_paths[key]).relative_to(Path(self.json_path).parent))
        return {
            "video_path": rel_path,
            "frame_index": frame_index,
        }

    def save_episode(self):
        """
        Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.
        """
        self.need_save = True  # Set the save flag
        print(f"==> Episode saved start...")

    def _save_episode(self):
        """
        Save the episode data to a JSON file.
        """
        self._release_video_writers()
        self._update_video_metadata()
        self.data['info'] = self.info
        self.data['text'] = self.text
        self.data['data'] = self.episode_data
        class _NumpyEncoder(json.JSONEncoder):
            def default(self, o):
                import numpy as np
                if isinstance(o, np.integer):
                    return int(o)
                if isinstance(o, np.floating):
                    return float(o)
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return super().default(o)

        with open(self.json_path, 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(self.data, indent=4, ensure_ascii=False, cls=_NumpyEncoder))
        self.need_save = False     # Reset the save flag
        self.is_available = True   # Mark the class as available after saving
        print(f"==> Episode (length:{len(self.episode_data)}) saved successfully to {self.json_path}.")

    def _release_video_writers(self):
        for writer in self.video_writers.values():
            try:
                writer.release()
            except Exception:
                pass
        self.video_writers = {}

    def _update_video_metadata(self):
        video_info = self.info.setdefault("video", {})
        video_info["format"] = self.VIDEO_EXT.lstrip(".")
        video_info["codec"] = self.VIDEO_CODEC
        video_info["fps"] = self.frequency
        cameras = video_info.setdefault("cameras", {})
        for key, path in self.video_paths.items():
            width, height = self.video_sizes.get(key, (None, None))
            if width is None or height is None:
                expected = self._expected_camera_shape(key)
                if expected is not None:
                    height, width, channels = expected
                else:
                    height, width, channels = None, None, 3
            else:
                channels = 3
            cameras[key] = {
                "height": int(height) if height is not None else None,
                "width": int(width) if width is not None else None,
                "channels": int(channels),
                "fps": self.frequency,
                "frames": int(self.video_frame_counts.get(key, 0)),
                "path": str(Path(path).relative_to(Path(self.json_path).parent)),
            }

    def discard_episode(self):
        """
        Discard the current episode without saving.
        Removes the episode directory and resets the state.
        """
        import shutil
        
        # Wait for any pending items in the queue to be processed
        self.item_data_queue.join()
        self._release_video_writers()
        
        # Remove the episode directory if it exists
        if hasattr(self, 'episode_dir') and os.path.exists(self.episode_dir):
            try:
                shutil.rmtree(self.episode_dir)
                print(f"==> Episode directory discarded: {self.episode_dir}")
            except Exception as e:
                print(f"==> Error removing episode directory: {e}")
        
        # Reset state
        self.episode_data = []
        self.item_id = -1
        self.episode_id = self._get_latest_episode_id()
        self.is_available = True
        self.need_save = False
        print(f"==> Episode discarded. Next episode will be episode_{str(self._get_next_episode_id()).zfill(4)}")

    def close(self):
        """
        Stop the worker thread and ensure all tasks are completed.
        """
        self.item_data_queue.join()
        if not self.is_available:  # If self.is_available is False, it means there is still data not saved.
            self.save_episode()
        while not self.is_available:
            time.sleep(0.01)
        self.stop_worker = True
        self.worker_thread.join()
