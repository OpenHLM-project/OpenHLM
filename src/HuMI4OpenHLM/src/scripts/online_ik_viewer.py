import asyncio
import enum
import json
import logging
import signal
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, TypedDict, overload

import numpy as np
import tyro
import zmq
import zmq.asyncio
from htc.rpc_protocol import (
    FRAME_EMPTY_DELIM,
    RpcRequest,
    WireMessage,
    make_request,
)
from mink import SE3, SO3, NoSolutionFound

from ikumi.config import IKConfig, get_config
from ikumi.keypoints import (
    HandRootFootKeyPoints,
)
from ikumi.solver import IKSolver
from ikumi.viser_visualizer import ViserConfig


@dataclass(frozen=True)
class RecorderConfig:
    """All knobs for networking, visualization, and IK during recording."""

    rpc_address: str = "tcp://192.168.1.108:4242"  # ZMQ REQ address for remote recording control.
    rpc_timeout_ms: int = 5000  # ZMQ RPC call timeout (ms).
    ui_freq: float = 60.0  # UI refresh frequency (Hz).

    start_countdown: float = (
        3.0  # Delay between pressing start and beginning to record (s).
    )
    prompt: str = ""
    """Fallback prompt to show and save when no prompt file entry is available."""
    prompt_file: Path | None = None
    """Optional .txt or .json prompt list. Episode N uses line/item N."""

    viser: ViserConfig = ViserConfig()  # Viser visualization configuration.
    ik: IKConfig = IKConfig()  # IK solver configuration.

    def create(self) -> "Recorder":
        """Create a Recorder from this config."""
        return Recorder(config=self)


default_configs = {
    "proposal": (
        "Proposal task.",
        RecorderConfig(ik=get_config("proposal")),
    ),
    "walk": (
        "Walk task.",
        RecorderConfig(ik=get_config("walk")),
    ),
    "squat_pick_ground": (
        "Pick ground squat task.",
        RecorderConfig(ik=get_config("squat_pick_ground")),
    ),
    "pick-low": (
        "Pick low task.",
        RecorderConfig(ik=get_config("pick-low")),
    ),
    "toss": (
        "Toss task.",
        RecorderConfig(ik=get_config("toss")),
    ),
    "unsheathe": (
        "Unsheathe sword task.",
        RecorderConfig(ik=get_config("unsheathe")),
    ),
    "pick": (
        "Pick something and put it on the mouse pad.",
        RecorderConfig(ik=get_config("pick")),
    ),
    "shelf": (
        "Shelf task derived from pick without using the waist tracker.",
        RecorderConfig(ik=get_config("shelf")),
    ),
}


class RecorderState(enum.Enum):
    WAITING_FOR_CONNECTION = enum.auto()
    READY_TO_RECORD = enum.auto()
    RECORDING = enum.auto()
    COUNTDOWN = enum.auto()


class EpisodeInfo(TypedDict):
    """Metadata about a recorded episode."""

    path: str
    """File path of the recorded episode."""
    frames: int
    """Number of frames in the episode."""
    error: str
    """Error message if any occurred during recording."""


class ZMQRecorderClient:
    """Minimal ZMQ REQ client wrapper matching `RecorderRPCClient` methods."""

    def __init__(self, address: str, timeout_ms: int = 1000):
        self._timeout_ms = timeout_ms

        self._ctx = zmq.asyncio.Context.instance()
        self._sock = self._ctx.socket(zmq.DEALER)
        # No linger on close to avoid blocking shutdown.
        self._sock.setsockopt(zmq.LINGER, 0)
        # No high-water mark limit to avoid dropping messages.
        self._sock.setsockopt(zmq.SNDHWM, 0)
        # Heatbeat settings
        self._sock.setsockopt(
            zmq.HEARTBEAT_IVL, timeout_ms // 2
        )  # half of timeout
        self._sock.setsockopt(zmq.HEARTBEAT_TIMEOUT, int(timeout_ms * 0.9))
        self._sock.connect(address)

        self._client_id: bytes = b"client-" + uuid.uuid4().bytes
        self._sock.setsockopt(zmq.IDENTITY, self._client_id)

        # Record the event loop this client is bound to.
        # All RPC calls will be marshaled to this loop to avoid
        # cross-loop errors with asyncio primitives.
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # If no loop is running at construction time, defer to None;
            # calls from any loop will run directly without marshalling.
            self._loop = None

        # Ensure only one call at a time within the client loop.
        self._lock = asyncio.Lock()

    def close(self):
        try:
            self._sock.close(linger=0)
        except Exception:
            pass

    @overload
    async def _call(self, method: Literal["ping"]) -> Literal["pong"]: ...

    @overload
    async def _call(self, method: Literal["get_pose_data"]) -> dict | None: ...

    @overload
    async def _call(
        self, method: Literal["start_recording"]
    ) -> dict[str, str]: ...

    @overload
    async def _call(
        self, method: Literal["stop_recording"]
    ) -> EpisodeInfo: ...
    @overload
    async def _call(
        self,
        method: Literal["get_num_recorded_episodes"],
    ) -> int: ...
    @overload
    async def _call(
        self,
        method: Literal["get_current_recording_frame_count"],
    ) -> int: ...
    @overload
    async def _call(
        self,
        method: Literal["delete_last_episode"],
    ) -> None: ...
    @overload
    async def _call(
        self, method: Literal["peak_last"]
    ) -> EpisodeInfo | None: ...
    @overload
    async def _call(self, method: Literal["is_recording"]) -> bool: ...

    async def _call(self, method: str, params: dict | None = None):
        async with self._lock:
            req_id = str(uuid.uuid4())
            req: RpcRequest = make_request(
                id_=req_id, method=method, params=params
            )
            wire_msg = WireMessage.from_payload_obj(req)
            payload_bytes = wire_msg.payload.encode("utf-8")
            # Send request
            await self._sock.send_multipart([FRAME_EMPTY_DELIM, payload_bytes])
            # Receive reply
            try:
                frames = await asyncio.wait_for(
                    self._sock.recv_multipart(),
                    timeout=self._timeout_ms / 1000,
                )
            except asyncio.TimeoutError as e:
                raise asyncio.TimeoutError(
                    f"RPC call to method '{method}' timed out. Req: {req}"
                ) from e
            # Check reply frames
            if len(frames) != 2 or frames[0] != FRAME_EMPTY_DELIM:
                raise RuntimeError(f"Invalid RPC reply frames: {frames}")
            _empty, payload = frames[0], frames[1]
            # Parse reply
            wire_msg = WireMessage(payload=payload.decode("utf-8"))
            reply = wire_msg.to_resp_or_err()

            if "id" in reply and "result" in reply:
                # Successful response
                result = reply["result"]
                return result
            elif "code" in reply and "message" in reply:
                # Error response
                raise RuntimeError(
                    f"RPC error from server: {reply['code']}: {reply['message']}"
                )
            else:
                raise RuntimeError(f"Invalid RPC reply: {reply}")

    async def _run_in_client_loop(self, coro):
        """Ensure `coro` runs on the client's loop, even if called from another loop.

        If invoked from the same loop the client was created on (or if the client
        has no bound loop), simply await the coroutine. Otherwise, submit the
        coroutine to the client's loop and await the concurrent.futures.Future
        from the current loop.
        """
        # Determine current running loop (if any).
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        # If loops match or client loop is unknown, run inline.
        if self._loop is None or current_loop is self._loop:
            return await coro

        # Marshal to the client loop.
        cf_future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        # Wrap the concurrent.futures.Future so it can be awaited in the current loop.
        return await asyncio.wrap_future(cf_future)

    # RPC methods
    async def ping(self) -> Literal["pong"]:
        return await self._run_in_client_loop(self._call("ping"))

    async def get_pose_data(self) -> dict | None:
        return await self._run_in_client_loop(self._call("get_pose_data"))

    async def start_recording(
        self, prompt: str | None = None
    ) -> dict[str, str]:
        params = {"prompt": prompt} if prompt is not None else None
        return await self._run_in_client_loop(
            self._call("start_recording", params=params)
        )

    async def stop_recording(self) -> EpisodeInfo:
        return await self._run_in_client_loop(self._call("stop_recording"))

    async def get_num_recorded_episodes(self) -> int:
        return await self._run_in_client_loop(
            self._call("get_num_recorded_episodes")
        )

    async def get_current_recording_frame_count(self) -> int:
        return await self._run_in_client_loop(
            self._call("get_current_recording_frame_count")
        )

    async def delete_last_episode(self) -> None:
        return await self._run_in_client_loop(
            self._call("delete_last_episode")
        )

    async def peak_last(self) -> EpisodeInfo | None:
        return await self._run_in_client_loop(self._call("peak_last"))

    async def is_recording(self) -> bool:
        return await self._run_in_client_loop(self._call("is_recording"))


class Recorder:
    """Orchestrates streaming IK targets, solving, visualization, and recording."""

    def __init__(self, config: RecorderConfig):
        self._config = config

        self._state = RecorderState.WAITING_FOR_CONNECTION

        # Track the latest incoming target before the IK loop consumes it.
        self._latest_target: HandRootFootKeyPoints | None = None
        self._latest_timestamp: float | None = None

        # Remaining countdown seconds before recording starts.
        self._countdown_time_left: float = 0.0
        self._prompts = self._load_prompts(self._config.prompt_file)
        self._last_prompt_index_shown: int | None = None

        # Instantiate IK solver and visualizer helpers.
        self._ik_solver: IKSolver = self._config.ik.create()

        self._visualizer = self._config.viser.create()
        self._build_viser_gui()

    @staticmethod
    def _load_prompts(prompt_file: Path | None) -> list[str]:
        if prompt_file is None:
            return []
        prompt_file = prompt_file.expanduser()
        if not prompt_file.is_file():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        if prompt_file.suffix.lower() == ".json":
            with prompt_file.open("r") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                payload = payload.get("prompts", [])
            if not isinstance(payload, list):
                raise ValueError(
                    "Prompt JSON must be a list or a dict with a 'prompts' list."
                )
            return [str(item) for item in payload]
        with prompt_file.open("r") as f:
            return [line.strip() for line in f if line.strip()]

    def _prompt_for_episode_index(self, episode_index: int) -> str:
        if self._prompts:
            return self._prompts[episode_index % len(self._prompts)]
        return self._config.prompt

    def _refresh_prompt_gui(self, episode_index: int) -> None:
        if self._last_prompt_index_shown == episode_index:
            return
        self._prompt_text.value = self._prompt_for_episode_index(episode_index)
        self._last_prompt_index_shown = episode_index

    async def run(self):
        """Run the recorder application."""
        # Shared shutdown flag that lets loops exit cooperatively.
        self._shutdown_event = asyncio.Event()
        # Create locks and rpc client
        self._state_lock = asyncio.Lock()
        # Set up the ZMQ DEALER client.
        self._rpc_client = ZMQRecorderClient(
            address=self._config.rpc_address,
            timeout_ms=self._config.rpc_timeout_ms,
        )
        # Check connection
        await self._rpc_client.ping()
        # Create background tasks.
        loop = asyncio.get_running_loop()
        self._ui_update_task = loop.create_task(self._ui_update_loop())
        self._ik_task = loop.create_task(self._ik_loop())
        self._align_task = loop.create_task(self._align_server_state_loop())
        tasks = [
            self._ui_update_task,
            self._ik_task,
            self._align_task,
        ]

        # Register signal handlers to trigger graceful shutdown.
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown_event.set)
            except NotImplementedError:
                # Signal handling may not be supported (e.g., on Windows); rely on KeyboardInterrupt.
                pass

        try:
            # Wait until shutdown is requested (Ctrl+C or SIGTERM).
            await self._shutdown_event.wait()
        except KeyboardInterrupt:
            # Fallback in environments without signal handler support.
            self._shutdown_event.set()
        finally:
            # Cancel tasks and wait for them to finish.
            for t in tasks:
                if t is not None and not t.done():
                    t.cancel()
            # Wait for tasks to finish.
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                pass

            # Close RPC client
            try:
                self._rpc_client.close()
            except Exception:
                pass

    async def _ik_loop(self):
        """Asynchronous loop to solve IK for latest targets."""
        last_timestamp: float | None = None
        while not self._shutdown_event.is_set():
            try:
                raw: dict | None = await self._rpc_client.get_pose_data()
            except asyncio.TimeoutError:
                raw = None
                logging.warning("Timeout while waiting for pose data.")
            except asyncio.CancelledError:
                break
            if raw is not None:
                self._latest_target = self._decode_target_msg(raw)
                self._latest_timestamp = self._latest_target.timestamp
                async with self._state_lock:
                    if self._state == RecorderState.WAITING_FOR_CONNECTION:
                        self._state = RecorderState.READY_TO_RECORD
                        # Recompute transforms when first target is received.
                        self._ik_solver.recompute_reference_transform(
                            self._latest_target
                        )
                        logging.info("Received first target, ready to record.")
            else:
                await asyncio.sleep(0.01)
                continue
            target = self._latest_target
            timestamp = self._latest_timestamp
            if last_timestamp is not None:
                dt = timestamp - last_timestamp
            else:
                dt = self._config.ik.dt
            try:
                _ik_start = time.perf_counter()
                ik_solution = self._ik_solver.solve_one_step(target, dt=dt)
                self._visualizer.visualize_ik_solution(ik_solution)
                _ik_stop = time.perf_counter()
                _ik_elapsed = _ik_stop - _ik_start
                if _ik_elapsed > 0.1:
                    logging.warning(
                        f"IK solving took too long: {_ik_elapsed:.3f} seconds."
                    )
            except NoSolutionFound:
                self._visualizer.update_info_text(
                    "No IK solution found for the latest target."
                )
            last_timestamp = timestamp
            await asyncio.sleep(0.001)

    async def _ui_update_loop(self):
        """Asynchronous loop to update UI elements."""
        while not self._shutdown_event.is_set():
            # Update UI elements based on the current state.
            # Await RPC to avoid passing coroutine objects to viser.
            self._episode_counter.value = (
                await self._rpc_client.get_num_recorded_episodes()
            )
            self._refresh_prompt_gui(int(self._episode_counter.value))
            async with self._state_lock:
                current_state = self._state
            match current_state:
                case RecorderState.WAITING_FOR_CONNECTION:
                    self._status_text.value = f"Waiting for connection at {self._config.rpc_address}..."
                    # Enable/disable buttons for the waiting state.
                    self._start_stop_button.disabled = True
                    self._start_stop_button.label = "Start Recording"
                    self._remove_last_button.disabled = True
                    self._recompute_transforms_button.disabled = True
                case RecorderState.READY_TO_RECORD:
                    self._status_text.value = "Ready to record."
                    # Enable/disable buttons for the ready state.
                    self._start_stop_button.disabled = False
                    self._start_stop_button.label = "Start Recording"
                    self._remove_last_button.disabled = False
                    self._recompute_transforms_button.disabled = False
                case RecorderState.COUNTDOWN:
                    self._status_text.value = f"Recording starts in {self._countdown_time_left:.1f} seconds..."
                    # Enable/disable buttons while counting down.
                    self._start_stop_button.disabled = True
                    self._start_stop_button.label = f"Recording Starting... ({self._countdown_time_left:.1f}s)"
                    self._remove_last_button.disabled = True
                    self._recompute_transforms_button.disabled = True
                case RecorderState.RECORDING:
                    # Await RPC for current frame count.
                    _current_frame_count = await self._rpc_client.get_current_recording_frame_count()
                    self._status_text.value = (
                        f"Recording... {_current_frame_count} frames recorded."
                    )
                    # Enable/disable buttons while recording.
                    self._start_stop_button.disabled = False
                    self._start_stop_button.label = "Stop Recording"
                    self._remove_last_button.disabled = True
                    self._recompute_transforms_button.disabled = True
            # Sleep briefly to yield control.
            try:
                await asyncio.sleep(1.0 / self._config.ui_freq)
            except asyncio.CancelledError:
                break

    async def _align_server_state_loop(self):
        """Periodically align the visualizer state with the recorder state."""
        while not self._shutdown_event.is_set():
            async with self._state_lock:
                current_state = self._state

            if current_state == RecorderState.WAITING_FOR_CONNECTION:
                await asyncio.sleep(0.1)
                continue

            server_is_recording = await self._rpc_client.is_recording()
            if (
                current_state == RecorderState.RECORDING
                and not server_is_recording
            ):
                # Server stopped recording unexpectedly; update state.
                async with self._state_lock:
                    self._state = RecorderState.READY_TO_RECORD
                logging.warning("Server stopped recording unexpectedly.")
                self._visualizer.update_info_text(
                    "Recording stopped unexpectedly by the server."
                )
            elif (
                current_state != RecorderState.RECORDING
                and server_is_recording
            ):
                # Server started recording unexpectedly; update state.
                async with self._state_lock:
                    self._state = RecorderState.RECORDING
                logging.warning("Server started recording unexpectedly.")
                self._visualizer.update_info_text(
                    "Recording started unexpectedly by the server."
                )

            await asyncio.sleep(0.1)

    async def _check_server_connection_loop(self):
        """Periodically check connection to the RPC server."""
        while not self._shutdown_event.is_set():
            try:
                await self._rpc_client.ping()
            except asyncio.TimeoutError:
                logging.error("Lost connection to RPC server.")
                async with self._state_lock:
                    self._state = RecorderState.WAITING_FOR_CONNECTION
            await asyncio.sleep(0.1)

    def _build_viser_gui(self):
        """Build Viser GUI elements for recording control."""
        server = self._visualizer.server

        with server.gui.add_folder("Recorder", order=0):
            self._start_stop_button = server.gui.add_button(
                label="Start Recording",
                disabled=True,  # Enabled after the first target is received.
            )

            async def _start_stop_recording(_event=None):
                if self._shutdown_event.is_set():
                    return
                async with self._state_lock:
                    current_state = self._state
                match current_state:
                    case RecorderState.READY_TO_RECORD:
                        await self._start_recording()
                    case RecorderState.RECORDING:
                        await self._stop_recording()

            self._start_stop_button.on_click(_start_stop_recording)

            self._remove_last_button = server.gui.add_button(
                label="Remove Last",
                disabled=True,  # Enabled after the first target is received.
            )
            self._remove_last_button.on_click(self._remove_last)

            self._recompute_transforms_button = server.gui.add_button(
                label="Recompute Transforms",
                disabled=True,  # Enabled after the first target is received.
            )
            self._recompute_transforms_button.on_click(
                self._recompute_transforms
            )

            self._prompt_text = server.gui.add_text(
                "Next Prompt",
                initial_value=self._prompt_for_episode_index(0),
                disabled=False,
                multiline=True,
            )
            self._status_text = server.gui.add_text(
                "Status",
                initial_value="Waiting for connection...",
                disabled=True,
                multiline=True,
            )
            self._episode_counter = server.gui.add_number(
                label="Recorded Episodes",
                initial_value=0,
                disabled=True,
            )

    async def _start_recording(self):
        """Begin recording after honoring the configured countdown."""
        async with self._state_lock:
            if self._state != RecorderState.READY_TO_RECORD:
                return  # Cannot start recording from the current state.
            self._state = RecorderState.COUNTDOWN
        # Use asyncio sleep to avoid blocking the event loop.
        await asyncio.sleep(0.1)  # Brief sleep to let the UI refresh.
        countdown_start = time.time()
        while time.time() - countdown_start < self._config.start_countdown:
            self._countdown_time_left = self._config.start_countdown - (
                time.time() - countdown_start
            )
            await asyncio.sleep(0.1)
        self._countdown_time_left = 0.0
        # recompute transforms before starting
        self._ik_solver.reset()
        if self._latest_target is not None:
            self._ik_solver.recompute_reference_transform(self._latest_target)
        await asyncio.sleep(0.5)
        await self._rpc_client.start_recording(prompt=self._prompt_text.value)
        async with self._state_lock:
            self._state = RecorderState.RECORDING

    async def _stop_recording(self, suppress_gui: bool = False):
        """Stop recording and persist the current episode to disk."""
        async with self._state_lock:
            if self._state != RecorderState.RECORDING:
                return  # Cannot stop recording from the current state.
            self._state = RecorderState.READY_TO_RECORD
        episode_info = await self._rpc_client.stop_recording()
        num_frames = episode_info["frames"]
        last_ep_path = episode_info["path"]
        error = episode_info["error"]
        if num_frames == 0:
            if not suppress_gui:
                self._visualizer.update_info_text(
                    "No frames recorded, nothing to save."
                )
            return
        if not suppress_gui:
            self._visualizer.update_info_text(
                f"Recording saved to `{last_ep_path}` with {num_frames} frames.\n Error summary: {error}"
            )
        next_episode_index = await self._rpc_client.get_num_recorded_episodes()
        self._refresh_prompt_gui(next_episode_index)

    async def _remove_last(self, _event=None):
        """Remove the last recorded episode."""
        async with self._state_lock:
            if self._state != RecorderState.READY_TO_RECORD:
                return  # Cannot remove last episode from the current state.
        last_ep_info = await self._rpc_client.peak_last()
        gui = self._visualizer.server.gui
        with self._visualizer.server.gui.add_modal(
            "Removing Last Recording"
        ) as modal:
            if last_ep_info is None:
                gui.add_markdown("No recorded files found to remove.")
                close_button = gui.add_button("Dismiss")
                close_button.on_click(lambda _event: modal.close())
                return
            last_file: str = last_ep_info["path"]
            gui.add_markdown(
                f"Are you sure you want to remove the last recorded file:\n\n`{last_file}`?"
            )
            yes_button = gui.add_button("Yes")

            async def _do_delete(_event):
                await self._rpc_client.delete_last_episode()
                next_episode_index = (
                    await self._rpc_client.get_num_recorded_episodes()
                )
                self._refresh_prompt_gui(next_episode_index)
                modal.close()
                self._visualizer.update_info_text(
                    f"Removed last recorded file: `{last_file}`."
                )
                return

            yes_button.on_click(_do_delete)
            no_button = gui.add_button("No")
            no_button.on_click(lambda _event: modal.close())

    async def _recompute_transforms(self, _event=None):
        """Reset the IK solver to home and recompute reference transforms."""
        async with self._state_lock:
            if self._state != RecorderState.READY_TO_RECORD:
                return  # Cannot recompute transforms from the current state.
        target = self._latest_target
        if target is None:
            return  # No target to recompute from.
        self._ik_solver.reset()
        self._ik_solver.recompute_reference_transform(target)

    def _decode_target_msg(self, msg: dict) -> HandRootFootKeyPoints:
        """Decode a tracker message dict into the appropriate keypoint container."""

        def _decode_one_keypoint(
            pos: Sequence[float], quat_wxyz: Sequence[float]
        ) -> SE3:
            """Convert tracker translation/quaternion (wxyz) into an SE3 pose."""
            rotation = SO3(wxyz=np.array(quat_wxyz))
            translation = np.array(pos)
            return SE3.from_rotation_and_translation(rotation, translation)

        pos = msg["pos"]
        # Accept either 'quat_wxyz' or legacy 'quat_xyzw'; normalize to wxyz.
        if "quat_wxyz" in msg:
            quat_wxyz = msg["quat_wxyz"]
        elif "quat_xyzw" in msg:
            quat_xyzw = msg["quat_xyzw"]
            # Convert each xyzw quaternion to wxyz
            quat_wxyz = [[q[3], q[0], q[1], q[2]] for q in quat_xyzw]
        else:
            raise KeyError(
                "Incoming pose message missing 'quat_wxyz'/'quat_xyzw'"
            )

        if not (len(pos) == 5 and len(quat_wxyz) == 5):
            logging.error(f"Expected 5 keypoints, got {len(pos)}")
            self._visualizer.update_info_text(
                f"Error: Expected 5 keypoints, got {len(pos)}."
            )
        root_pose = _decode_one_keypoint(pos[0], quat_wxyz[0])
        left_hand_pose = _decode_one_keypoint(pos[1], quat_wxyz[1])
        right_hand_pose = _decode_one_keypoint(pos[2], quat_wxyz[2])
        left_foot_pose = _decode_one_keypoint(pos[3], quat_wxyz[3])
        right_foot_pose = _decode_one_keypoint(pos[4], quat_wxyz[4])
        return HandRootFootKeyPoints(
            timestamp=time.time(),
            root_pose=root_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            left_foot_pose=left_foot_pose,
            right_foot_pose=right_foot_pose,
        )


def main(
    recorder: RecorderConfig,
):
    """Create and run a Recorder from the given configuration."""
    recorder_instance = recorder.create()
    try:
        asyncio.run(recorder_instance.run())
    except KeyboardInterrupt:
        pass


def cli():
    rec = tyro.extras.overridable_config_cli(default_configs)
    main(rec)


if __name__ == "__main__":
    cli()
