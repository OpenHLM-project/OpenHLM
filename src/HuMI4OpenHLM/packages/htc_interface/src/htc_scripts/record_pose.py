"""Record transformed HTC tracker poses to per-episode JSON files (Windows).

This Windows-only script initializes OpenVR, reads tracker poses, transforms
them into the robot frame, and records episodes controlled via keyboard:

- Press `s` to start recording (live Rich table shows poses).
- Press `t` to stop and save an episode JSON to `output_dir`.
- Press `Backspace` to delete the last saved episode (with confirmation).

Each saved JSON contains a single `episode` key holding a list of frames. Each
frame includes a `timestamp` and `{role}_pose` entries, where every pose has
`position` and `quaternion_wxyz` arrays.

Console script: `record-pose`

Usage example:
    record-pose --frequency 120 --config-path tracker_config.json \
        --output-dir data/recordings_htc
"""

import asyncio
import json
import math
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable

import openvr  # type: ignore
import tyro
import win_precise_time as wpt  # type: ignore
import zmq
import zmq.asyncio
from htc.pose_common import (
    DEFAULT_ROLES,
    PoseData,
    PoseFrame,
    make_table,
    transform_poses,
)
from htc.rpc_protocol import (
    FRAME_EMPTY_DELIM,
    RpcError,
    RpcErrorCode,
    RpcRequest,
    RpcResponse,
    WireMessage,
    make_response_err,
    make_response_ok,
)
from rich.console import Console
from rich.live import Live
from rich.table import Table

_time_func: Callable[[], float] = wpt.time


def precise_wait_until(target_time: float, slack_time: float = 0.002):
    while True:
        now = wpt.time()
        remain = target_time - now
        if remain <= 0:
            break
        elif remain > slack_time:
            wpt.sleep(remain - slack_time)
        else:
            while wpt.time() < target_time:
                pass


def _build_idle_table(episode_count: int) -> Table:
    table = Table(title="HTC Raw Pose Recorder")
    table.add_column("Status", justify="left", style="cyan")
    table.add_column("Value", justify="left", style="magenta")
    table.add_row("Recorded episodes", str(episode_count))
    table.add_row("Controls", "s start  t stop  [Backspace] delete last")
    return table


def _nonblocking_getch() -> str | None:
    """Windows-only nonblocking keyboard read using msvcrt."""
    import msvcrt  # type: ignore

    if msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch == "\x7f":  # Normalize backspace
            ch = "\x08"
        return ch
    return None


@dataclass
class RecorderConfig:
    """Configuration for raw pose recording."""

    frequency: float = 120.0
    """Frequency for streaming tracker poses (Hz)."""
    roles_to_record: list[str] = field(default_factory=lambda: DEFAULT_ROLES)
    """Ordered roles required in each frame; missing roles
            cause the frame to be skipped to keep episode structure consistent."""
    config_path: Path = Path(__file__).parents[1] / Path("tracker_config.json")
    """Path to to `tracker_config.json` mapping role -> serial."""
    output_dir: Path = Path("data/recordings_htc")
    """Directory to save recorded episode JSON files."""


@dataclass
class RecorderRPCConfig:
    """Configuration for raw pose recorder RPC server."""

    serve: bool = False
    """If True, serve a ZMQ ROUTER RPC server."""
    bind: str = "tcp://0.0.0.0:4242"
    """RPC server bind address."""


class RawPoseRecorder:
    """Windows-only recorder that saves per-episode raw poses.

    Handles keyboard controls via `msvcrt`, displays a live Rich table while
    recording, and saves a compact episode JSON where every frame includes
    timestamp and per-role poses in robot coordinates.
    """

    def __init__(self, cfg: RecorderConfig, console: Console | None = None):
        self._cfg = cfg
        if self._cfg.roles_to_record is None:
            self._roles_to_record = DEFAULT_ROLES
        else:
            self._roles_to_record = self._cfg.roles_to_record

        # Enforce Windows platform
        if platform.system() != "Windows":
            raise RuntimeError(
                "This script is intended to run on Windows only."
            )

        # Initialize VR
        openvr.init(openvr.VRApplication_Other)
        self._vr = openvr.VRSystem()

        # Load tracker mapping
        with self._cfg.config_path.open("r") as f:
            tracker_config = json.load(f)
        # serial -> role
        self._serial_to_role: dict[str, str] = {
            v: k for k, v in tracker_config.items()
        }

        self._recording: bool = False
        self._episode_frames: list[dict] = []
        self._episode_start_ts: float | None = None
        self._episode_prompt: str | None = None
        self._console = Console() if console is None else console

        self._latest_pose_data: PoseData | None = None

    # Required API
    def ping(self) -> str:
        return "pong"

    def get_pose_data(self):
        """Get the latest pose data, if any."""
        return (
            asdict(self._latest_pose_data) if self._latest_pose_data else None
        )

    def peak_last(self) -> dict | None:
        """Get last episode info."""
        pattern = self._cfg.output_dir.glob("recording_*.json")
        last = sorted(pattern, key=lambda p: p.stat().st_mtime)
        if not last:
            return None
        target = last[-1]
        # Try to read frame count
        try:
            with target.open("r") as f:
                data = json.load(f)
            frames = len(data.get("episode", []))
        except Exception:
            frames = None
        return {
            "path": target.as_posix(),
            "frames": frames,
        }

    def get_num_recorded_episodes(self) -> int:
        """Get the number of recorded episodes."""
        if not self._cfg.output_dir.exists():
            return 0
        pattern = self._cfg.output_dir.glob("recording_*.json")
        return len(list(pattern))

    def get_current_recording_frame_count(self) -> int:
        """Get the number of frames in the current recording session."""
        return len(self._episode_frames)

    def start_recording(
        self, prompt: str | None = None
    ) -> dict[str, str] | None:
        if self._recording:
            return None
        self._episode_frames.clear()
        self._episode_start_ts = _time_func()
        self._episode_prompt = prompt
        self._recording = True
        return {"status": "ok"}

    def stop_recording(self) -> dict[str, str | int]:
        if not self._recording:
            return {"status": "not_recording"}
        self._recording = False
        if not self._episode_frames:
            self._console.print("No frames captured; nothing saved.")
            return {"status": "no_save"}

        # Validate temporal and spatial continuity before saving.
        _error_str = self._validate_episode(
            time_jump_threshold=0.1, position_jump_threshold=0.1
        )
        if _error_str is None:
            error_str = "All correct!"
        else:
            error_str = _error_str

        start_ts = (
            self._episode_start_ts or self._episode_frames[0]["timestamp"]
        )
        ts_str = datetime.fromtimestamp(start_ts).strftime(
            "%Y.%m.%d_%H.%M.%S.%f"
        )
        out_dir = self._cfg.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"recording_{ts_str}.json"
        payload: dict[str, Any] = {"episode": self._episode_frames}
        if self._episode_prompt:
            payload["prompt"] = self._episode_prompt
        with out_path.open("w") as f:
            json.dump(payload, f, indent=4)
        frame_count = len(self._episode_frames)
        self._console.print(
            f"Saved recording with {frame_count} frames to {out_path}"
        )
        self._episode_frames.clear()
        self._episode_start_ts = None
        self._episode_prompt = None
        return {
            "path": out_path.as_posix(),
            "frames": frame_count,
            "error": error_str,
        }

    def delete_last_episode(self, confirm: bool = True):
        pattern = self._cfg.output_dir.glob("recording_*.json")
        last = sorted(pattern, key=lambda p: p.stat().st_mtime)
        if not last:
            self._console.print("No recorded files to delete.")
            return
        target = last[-1]
        if confirm:
            self._console.print(
                f"Delete last episode: {target}? [y/N] ", end=""
            )
            # Temporarily leave Live rendering by reading from stdin
            ans = input().strip().lower()
            if ans == "y":
                try:
                    target.unlink()
                    self._console.print("Deleted.")
                except Exception as e:
                    self._console.print(f"Failed to delete: {e}")
            else:
                self._console.print("Cancelled.")
        else:
            try:
                target.unlink()
                self._console.print(f"Deleted last episode: {target}")
            except Exception as e:
                self._console.print(f"Failed to delete: {e}")

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    def get_rpc_handler_map(self) -> dict[str, Callable[..., Any]]:
        """Get a mapping of RPC method names to handler functions."""
        return {
            "ping": self.ping,
            "get_pose_data": self.get_pose_data,
            "peak_last": self.peak_last,
            "get_num_recorded_episodes": self.get_num_recorded_episodes,
            "get_current_recording_frame_count": self.get_current_recording_frame_count,
            "start_recording": self.start_recording,
            "stop_recording": self.stop_recording,
            "delete_last_episode": partial(
                self.delete_last_episode, confirm=False
            ),
            "is_recording": self.is_recording,
        }

    # Main loop
    def run(self):
        self._console.print("VR system initialized. Waiting for trackers...")
        wpt.sleep(2.0)

        dt = 1.0 / self._cfg.frequency
        try:
            with Live(refresh_per_second=30, console=self._console) as live:
                while True:
                    poses = PoseFrame.read_from_vr_system(
                        self._vr, time_func=_time_func
                    )

                    # Ensure all trackers in config are recognized
                    unknown = [
                        p.serial_number
                        for p in poses
                        if p.serial_number not in self._serial_to_role
                    ]
                    if unknown:
                        live.update(_build_idle_table(self._episode_count()))
                        self._console.print(
                            f"Warning: Unknown tracker serials: {unknown}"
                        )
                    else:
                        poses = transform_poses(poses, self._serial_to_role)
                        try:
                            self._latest_pose_data = PoseData.from_pose_frames(
                                pose_frames=poses,
                                roles_to_send=self._roles_to_record,
                                serial_to_role=self._serial_to_role,
                            )
                        except ValueError as e:
                            self._console.print(f"Error: {e}")

                    now = _time_func()
                    next_time = now + dt

                    # Handle recording
                    if self._recording:
                        frame = self._build_frame(now, poses)
                        if frame is not None:
                            self._episode_frames.append(frame)
                        table = make_table(poses, self._serial_to_role)
                        table.title = (
                            f"Recording... frames={len(self._episode_frames)}"
                        )
                        live.update(table)
                    else:
                        live.update(
                            _build_idle_table(
                                self._episode_count(),
                            )
                        )

                    # Input handling
                    ch = _nonblocking_getch()
                    if ch is not None:
                        if ch.lower() == "s":
                            self.start_recording()
                        elif ch.lower() == "t":
                            self.stop_recording()
                        elif ch == "\x08":  # Backspace
                            # Pause Live to allow clean input prompt
                            live.stop()
                            try:
                                self.delete_last_episode()
                            finally:
                                live.start()

                    precise_wait_until(next_time)

        except KeyboardInterrupt:
            pass
        finally:
            try:
                openvr.shutdown()
            except Exception:
                pass

    def _build_frame(
        self, timestamp: float, poses: list[PoseFrame]
    ) -> dict | None:
        # Map roles present in this frame
        serial_to_pose = {p.serial_number: p for p in poses}
        role_to_pose: dict[str, PoseFrame] = {}
        for serial, role in self._serial_to_role.items():
            if serial in serial_to_pose:
                role_to_pose[role] = serial_to_pose[serial]

        # Ensure all required roles are present
        for role in self._roles_to_record or []:
            if role not in role_to_pose:
                # Skip frame to ensure each element has all requested roles
                return None

        frame: dict = {"timestamp": timestamp}
        for role in self._roles_to_record or []:
            pose = role_to_pose[role]
            frame[f"{role}_pose"] = {
                "position": pose.position.tolist(),
                "quaternion_wxyz": pose.quaternion_wxyz.tolist(),
            }
        return frame

    def _episode_count(self) -> int:
        try:
            return len(list(self._cfg.output_dir.glob("recording_*.json")))
        except Exception:
            return 0

    def _validate_episode(
        self, time_jump_threshold: float, position_jump_threshold: float
    ) -> str | None:
        frames = self._episode_frames
        error_str = ""
        if len(frames) < 2:
            return
        roles = self._roles_to_record or []
        for i in range(1, len(frames)):
            prev = frames[i - 1]
            cur = frames[i]
            dt = cur["timestamp"] - prev["timestamp"]
            if dt > time_jump_threshold:
                self._console.print(
                    f"Warning: time jump {dt:.3f}s between frames {i - 1}->{i}"
                )
                error_str += (
                    f"Time jump {dt:.3f}s between frames {i - 1}->{i}\n"
                )
            for role in roles:
                key = f"{role}_pose"
                if key in prev and key in cur:
                    p0 = prev[key]["position"]
                    p1 = cur[key]["position"]
                    try:
                        d = math.dist(p0, p1)  # type: ignore[arg-type]
                    except TypeError:
                        # Fallback for environments without math.dist or bad types
                        dx = p1[0] - p0[0]
                        dy = p1[1] - p0[1]
                        dz = p1[2] - p0[2]
                        d = (dx * dx + dy * dy + dz * dz) ** 0.5
                    if d > position_jump_threshold:
                        self._console.print(
                            f"Warning: {role} position jump {d:.3f}m between frames {i - 1}->{i}"
                        )
                        error_str += f"{role} position jump {d:.3f}m between frames {i - 1}->{i}\n"
        return error_str if error_str else None


class RpcServer:
    """ZMQ-based RPC server for RawPoseRecorder."""

    def __init__(
        self,
        cfg: RecorderRPCConfig,
        recorder: RawPoseRecorder,
        console: Console | None = None,
    ):
        self._cfg = cfg
        self._recorder = recorder
        self._console = console or Console()

        # Initialize ZMQ asyncio context and socket
        self._ctx = zmq.asyncio.Context.instance()
        self._sock = self._ctx.socket(zmq.ROUTER)
        # Graceful shutdown
        self._sock.setsockopt(zmq.LINGER, 0)

        # Build handler map
        self._handler_map = recorder.get_rpc_handler_map()

        # Events
        self._stop_event = asyncio.Event()

    async def _send_response(
        self, client_id: bytes, response: RpcResponse | RpcError
    ) -> None:
        """Send a response to the client."""
        wire_msg = WireMessage.from_payload_obj(response)
        payload_bytes = wire_msg.payload.encode("utf-8")
        await self._sock.send_multipart(
            [client_id, FRAME_EMPTY_DELIM, payload_bytes]
        )

    async def run(self):
        """Run the RPC server loop."""
        self._sock.bind(self._cfg.bind)
        self._console.print(
            f"[Server] RPC server listening at {self._cfg.bind}..."
        )

        while not self._stop_event.is_set():
            try:
                # Receive multipart message: client id, empty, payload
                frames = await self._sock.recv_multipart()
                if len(frames) < 3 or frames[-2] != FRAME_EMPTY_DELIM:
                    self._console.print("Received malformed RPC message.")
                    continue
                client_id, _empty, payload = frames[0], frames[1], frames[2]
                # Parse request
                try:
                    wire_msg = WireMessage(payload=payload.decode("utf-8"))
                    req: RpcRequest = wire_msg.to_req()
                except Exception as e:
                    err: RpcError = make_response_err(
                        code=RpcErrorCode.INVALID_PARAMS,
                        message=f"Invalid request payload: {e}",
                    )
                    await self._send_response(client_id, err)
                    continue
                # Handle request
                resp = await self.handle_request(req)
                await self._send_response(client_id, resp)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._console.print(f"[Server] Error in the main loop: {e}")
        self._console.print("[Server] RPC server shutting down.")

    async def handle_request(self, req: RpcRequest) -> RpcResponse | RpcError:
        """Handle a single RPC request."""
        req_id = req.get("id")
        method_name = req.get("method")
        if not isinstance(method_name, str):
            return make_response_err(
                code=RpcErrorCode.INVALID_PARAMS,
                message=f"Invalid method name: {method_name}",
            )
        handler = self._handler_map.get(method_name)
        if handler is None:
            return make_response_err(
                code=RpcErrorCode.UNKNOWN_METHOD,
                message=f"Unknown method: {method_name}",
            )
        try:
            params = req.get("params", {})
            if params is None:
                params = {}
            if not isinstance(params, dict):
                return make_response_err(
                    code=RpcErrorCode.INVALID_PARAMS,
                    message=f"Invalid params for {method_name}: {params}",
                )
            result = handler(**params)
            return make_response_ok(id_=req_id, result=result)
        except Exception as e:
            return make_response_err(
                code=RpcErrorCode.INTERNAL_ERROR, message=f"Handler error: {e}"
            )

    async def stop(self) -> None:
        """Stop the RPC server gracefully."""
        self._stop_event.set()
        await asyncio.sleep(0.1)  # Allow loop to exit
        self._sock.close(linger=0)


def main(rec: RecorderConfig, rpc: RecorderRPCConfig):
    """Run the Windows-only raw pose recorder with CLI arguments."""
    console = Console()
    recorder = RawPoseRecorder(rec, console=console)

    if rpc.serve:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        async def rec_run():
            try:
                await asyncio.to_thread(recorder.run)
            except asyncio.CancelledError:
                pass

        async def serve():
            rpc_server = RpcServer(rpc, recorder, console=console)
            # Start both tasks
            rec_task = asyncio.create_task(rec_run())
            rpc_task = asyncio.create_task(rpc_server.run())
            tasks = [rec_task, rpc_task]
            try:
                # Wait until any task completes or is cancelled
                await asyncio.gather(*tasks)
            except KeyboardInterrupt:
                # Gracefully cancel tasks
                for t in tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                # Stop RPC server
                try:
                    await rpc_server.stop()
                except Exception:
                    pass

        # Start the event loop
        asyncio.run(serve())

    else:
        recorder.run()


if __name__ == "__main__":
    tyro.cli(main)


def cli():
    """Entry point for the `record-pose` console command.

    Wraps the Tyro-driven `main` so the script can be installed and invoked as
    a standard command without relying on `python -m` execution.
    """
    tyro.cli(main)
