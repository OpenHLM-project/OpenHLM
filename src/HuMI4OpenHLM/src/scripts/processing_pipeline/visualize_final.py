import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import plotly.graph_objects as go
import tyro
import viser
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation as R
from viser._gui_handles import (
    GuiButtonHandle,
    GuiDropdownHandle,
    GuiMarkdownHandle,
    GuiPlotlyHandle,
    GuiSliderHandle,
)
from viser.extras import ViserUrdf


def _quat_wxyz_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    q_xyzw = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def _read_frame_rgb(episode_dir: Path, ref: object) -> np.ndarray:
    if not isinstance(ref, dict):
        raise ValueError(f"Expected video frame reference, got {ref!r}")
    video_path = ref.get("video_path")
    frame_index = ref.get("frame_index")
    if video_path is None or frame_index is None:
        raise ValueError(f"Invalid video frame reference: {ref!r}")

    cap = cv2.VideoCapture((episode_dir / str(video_path)).as_posix())
    try:
        if not cap.isOpened():
            raise FileNotFoundError(
                f"Failed to open video: {episode_dir / str(video_path)}"
            )
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise RuntimeError(
                f"Failed to read frame {frame_index} from {video_path}"
            )
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


@dataclass(frozen=True)
class Args:
    input_dir: Annotated[Path, tyro.conf.arg(aliases=["-i"])] = Path("data/test2")
    final_dirname: str = "final_data"
    urdf_path: Path = Path("g1_description/g1_29dof_rev_1_0.urdf")
    port: int = 8082
    robot_height: float = 0.8
    fps: float = 30.0


@dataclass
class GuiHandles:
    episode_dropdown: GuiDropdownHandle
    frame_slider: GuiSliderHandle
    play_pause_button: GuiButtonHandle
    autoplay_speed: GuiSliderHandle
    prompt_text: GuiMarkdownHandle
    image_pair: GuiPlotlyHandle
    gripper_plot: GuiPlotlyHandle
    is_playing: bool = False


class FinalDataVisualizer:
    def __init__(self, args: Args):
        self.args = args
        self.final_dir = (args.input_dir / args.final_dirname).resolve()
        if not self.final_dir.is_dir():
            raise FileNotFoundError(f"final_data folder not found: {self.final_dir}")

        self.episode_dirs = sorted(
            p for p in self.final_dir.glob("episode_*") if p.is_dir()
        )
        if not self.episode_dirs:
            raise ValueError(f"No episode_* folders found under {self.final_dir}")

        self.server = viser.ViserServer(port=args.port)
        self.server.scene.set_up_direction("+z")
        self.server.scene.world_axes.visible = True
        self.server.scene.add_grid(
            "/grid",
            plane="xy",
            shadow_opacity=0.0,
            infinite_grid=True,
            cell_size=0.25,
        )

        self.robot_base = self.server.scene.add_frame(
            "/robot/g1",
            show_axes=True,
            axes_length=0.1,
            axes_radius=0.004,
        )
        self.robot = ViserUrdf(
            target=self.server,
            urdf_or_path=args.urdf_path,
            root_node_name="/robot/g1",
        )
        self.actuated_dof = len(self.robot.get_actuated_joint_names())

        self.cur_episode_idx = 0
        self.cur_episode_dir = self.episode_dirs[0]
        self.frames: list[dict] = []
        self.root_yaw: np.ndarray = np.zeros((0,), dtype=np.float64)
        self.frame_count = 0
        self.current_frame_idx = 0
        self.last_rendered_idx: int = -1
        self._pending_programmatic_slider_updates = 0
        self._play_start_ts = time.monotonic()
        self._play_start_frame_idx = 0

        self.gui = self._create_gui()
        self._bind_events()
        self._load_episode(0)

    def _create_gui(self) -> GuiHandles:
        episode_opts = [str(i) for i in range(len(self.episode_dirs))]
        ep = self.server.gui.add_dropdown(
            "Episode",
            options=episode_opts,
            initial_value=episode_opts[0],
            order=0,
        )
        frame = self.server.gui.add_slider(
            "Frame",
            min=0.0,
            max=1.0,
            step=1.0,
            initial_value=0.0,
            order=1,
            disabled=True,
        )
        play = self.server.gui.add_button("Play / Pause", order=2)
        speed = self.server.gui.add_slider(
            "Speed (fps)", 1, 120, 1, int(self.args.fps), order=3
        )
        prompt = self.server.gui.add_markdown("**Prompt:**", order=4)
        with self.server.gui.add_folder("Plots", order=5):
            image_pair = self.server.gui.add_plotly(figure=go.Figure(), order=0)
            gripper = self.server.gui.add_plotly(figure=go.Figure(), order=1)
        return GuiHandles(
            episode_dropdown=ep,
            frame_slider=frame,
            play_pause_button=play,
            autoplay_speed=speed,
            prompt_text=prompt,
            image_pair=image_pair,
            gripper_plot=gripper,
        )

    def _bind_events(self) -> None:
        @self.gui.play_pause_button.on_click
        def _(_) -> None:
            self.gui.is_playing = not self.gui.is_playing
            if self.gui.is_playing:
                self._reset_playback_anchor()

        @self.gui.episode_dropdown.on_update
        def _(_) -> None:
            was_playing = self.gui.is_playing
            self.cur_episode_idx = int(self.gui.episode_dropdown.value)
            self._load_episode(self.cur_episode_idx)
            self.gui.is_playing = was_playing
            if self.gui.is_playing:
                self._reset_playback_anchor()

        @self.gui.autoplay_speed.on_update
        def _(_) -> None:
            if self.gui.is_playing:
                self._reset_playback_anchor()

        self._bind_frame_slider_events()

    def _bind_frame_slider_events(self) -> None:
        @self.gui.frame_slider.on_update
        def _(_) -> None:
            if self._pending_programmatic_slider_updates > 0:
                self._pending_programmatic_slider_updates -= 1
                return
            if self.gui.is_playing:
                self._reset_playback_anchor()

    def _recreate_frame_slider(self, initial_idx: int = 0) -> None:
        max_idx = max(self.frame_count - 1, 0)
        initial_idx = max(0, min(initial_idx, max_idx))

        self.gui.frame_slider.remove()
        self.gui.frame_slider = self.server.gui.add_slider(
            "Frame",
            min=0.0,
            max=float(max(max_idx, 1)),
            step=1.0,
            initial_value=float(initial_idx),
            order=1,
            disabled=max_idx <= 0,
        )
        if max_idx <= 0:
            self.gui.frame_slider.max = 0.0
        self._pending_programmatic_slider_updates = 0
        self._bind_frame_slider_events()

    def _load_episode(self, episode_idx: int) -> None:
        self.cur_episode_dir = self.episode_dirs[episode_idx]
        data_path = self.cur_episode_dir / "data.json"
        with data_path.open("r") as f:
            payload = json.load(f)
        self.frames = payload["data"]
        self.gui.prompt_text.content = self._format_prompt(payload)
        self.frame_count = len(self.frames)
        self.current_frame_idx = 0
        self._recreate_frame_slider(initial_idx=0)
        self.root_yaw = self._integrate_root_yaw()
        self.last_rendered_idx = -1
        self._render_frame(0)
        self._reset_playback_anchor()

    @staticmethod
    def _format_prompt(payload: dict) -> str:
        text = payload.get("text", {})
        prompt = ""
        if isinstance(text, dict):
            prompt = str(text.get("goal") or text.get("desc") or "")
        return f"**Prompt:** {prompt}" if prompt else "**Prompt:**"

    def _maybe_frame_index(self, value: object) -> int | None:
        if self.frame_count <= 0:
            return 0
        try:
            idx = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(idx):
            return None
        return int(np.clip(round(idx), 0, self.frame_count - 1))

    def _integrate_root_yaw(self) -> np.ndarray:
        n = self.frame_count
        yaw = np.zeros((n,), dtype=np.float64)
        if n <= 1:
            return yaw
        dt = 1.0 / float(self.args.fps)
        yaw_vel = np.array([float(f["state_body"][2]) for f in self.frames])
        yaw[1:] = np.cumsum(yaw_vel[:-1]) * dt
        return yaw

    def _render_frame(self, idx: int) -> None:
        if self.frame_count == 0:
            return
        idx = max(0, min(idx, self.frame_count - 1))
        self.current_frame_idx = idx
        if idx == self.last_rendered_idx:
            return
        frame = self.frames[idx]

        body = frame.get("state_body", frame.get("action_body"))
        if body is None:
            raise KeyError("Frame has neither state_body nor action_body")
        if len(body) < 3 + self.actuated_dof:
            raise ValueError(
                f"state_body len={len(body)} < 3+{self.actuated_dof}"
            )

        roll = float(body[0])
        pitch = float(body[1])
        yaw = float(self.root_yaw[idx])
        joints = np.asarray(body[3 : 3 + self.actuated_dof], dtype=np.float64)

        with self.server.atomic():
            self.robot.update_cfg(joints)
            self.robot_base.position = np.array([0.0, 0.0, self.args.robot_height])
            self.robot_base.wxyz = _quat_wxyz_from_euler_xyz(roll, pitch, yaw)

        left_img = _read_frame_rgb(self.cur_episode_dir, frame["wrist_rgb_left"])
        right_img = _read_frame_rgb(self.cur_episode_dir, frame["wrist_rgb_right"])

        fig_images = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"wrist_rgb_left (Frame {idx})",
                f"wrist_rgb_right (Frame {idx})",
            ),
            horizontal_spacing=0.03,
        )
        fig_images.add_trace(go.Image(z=left_img), row=1, col=1)
        fig_images.add_trace(go.Image(z=right_img), row=1, col=2)
        fig_images.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig_images.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig_images.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            height=360,
        )
        self.gui.image_pair.figure = fig_images

        left_series = np.array(
            [float(x.get("action_hand_left", np.nan)) for x in self.frames]
        )
        right_series = np.array(
            [float(x.get("action_hand_right", np.nan)) for x in self.frames]
        )
        figg = go.Figure()
        figg.add_trace(go.Scatter(y=left_series, mode="lines", name="left"))
        figg.add_trace(go.Scatter(y=right_series, mode="lines", name="right"))
        figg.add_vline(
            x=idx,
            line=dict(color="red", width=2, dash="dash"),
            annotation_text=f"Frame {idx}",
        )
        figg.update_layout(
            title="Gripper Action",
            xaxis_title="Frame",
            yaxis_title="Action",
            legend_title="Hand",
        )
        if self.frame_count > 0:
            figg.update_xaxes(range=[0, self.frame_count - 1])
        figg.update_yaxes(range=[0.0, 1.0])
        self.gui.gripper_plot.figure = figg
        self.last_rendered_idx = idx

    def _set_frame_slider_value(self, idx: int) -> None:
        if self.frame_count > 0:
            idx = max(0, min(idx, self.frame_count - 1))
        else:
            idx = 0
        self._pending_programmatic_slider_updates += 1
        self.gui.frame_slider.value = float(idx)

    def _reset_playback_anchor(self) -> None:
        idx = self._maybe_frame_index(self.gui.frame_slider.value)
        if idx is not None:
            self.current_frame_idx = idx
        self._play_start_ts = time.monotonic()
        self._play_start_frame_idx = self.current_frame_idx

    def run(self) -> None:
        while True:
            if self.frame_count > 0:
                if self.gui.is_playing:
                    fps = max(float(self.gui.autoplay_speed.value), 1.0)
                    now = time.monotonic()
                    elapsed_s = max(now - self._play_start_ts, 0.0)
                    frames_advanced = int(elapsed_s * fps)
                    target_idx = (
                        self._play_start_frame_idx + frames_advanced
                    ) % self.frame_count
                    if target_idx != self.current_frame_idx:
                        self._set_frame_slider_value(target_idx)
                        self._render_frame(target_idx)
                    next_frame_s = (frames_advanced + 1) / fps
                    sleep_s = max(next_frame_s - elapsed_s, 0.001)
                else:
                    idx = self._maybe_frame_index(self.gui.frame_slider.value)
                    if idx is not None:
                        self._render_frame(idx)
                    sleep_s = 1.0 / 60.0
            else:
                sleep_s = 0.1
            time.sleep(sleep_s)


def main(args: Args) -> None:
    vis = FinalDataVisualizer(args)
    print(f"Server running at http://localhost:{args.port}")
    vis.run()


def cli() -> None:
    main(tyro.cli(Args))


if __name__ == "__main__":
    cli()
