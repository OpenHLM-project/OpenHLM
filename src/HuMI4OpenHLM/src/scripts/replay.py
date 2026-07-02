import asyncio
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

from ikumi.config import IKConfig
from ikumi.episodes import load_ik_episode, load_target_episode
from ikumi.solution import IKSolution
from ikumi.viser_visualizer import ViserConfig


@dataclass(frozen=True)
class ReplayConfig:
    """Configuration for the replay script."""

    viser: ViserConfig = ViserConfig()

    input_dir: Annotated[Path, tyro.conf.arg(aliases=["-i"])] = tyro.MISSING
    """Directory containing recorded episodes to replay."""

    def crate(self) -> "Replayer":
        """Create a Replayer instance from this configuration."""
        return Replayer(self)


class Replayer:
    """Replay episodes using the Viser server."""

    def __init__(
        self,
        config: ReplayConfig,
    ):
        self._config = config
        self._ik_config = IKConfig.from_yaml(
            self._config.input_dir / "ik_config_used.yaml"
        )

        self._viser_server = self._config.viser.create()

        # episode jsons
        self._episode_files: list[Path] = sorted(
            self._config.input_dir.glob("*recording*.json")
        )
        if len(self._episode_files) == 0:
            raise ValueError(
                f"No episode files found in {self._config.input_dir}"
            )
        self._current_file = self._episode_files[0]

        # episode states
        self._current_targets: list = []
        self._current_solutions: list[IKSolution] | None = None
        self._load_current_file()
        self._current_frame_idx = 0

        # replay states
        self._is_playing = False
        self._playback_speed = 1.0  # playback speed multiplier

        # build GUI
        self._build_viser_gui()
        self._gui_update_event = asyncio.Event()

        # event loop
        self._event_loop = asyncio.get_event_loop()
        self._shutdown_event = asyncio.Event()

    def run(self):
        """Run the replayer."""
        self._playback_task = self._event_loop.create_task(
            self._playback_loop()
        )
        self._scan_files_task = self._event_loop.create_task(
            self._scan_files_loop()
        )
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._event_loop.add_signal_handler(
                sig, lambda: self._shutdown_event.set()
            )

        # run until shutdown
        self._event_loop.run_until_complete(self._shutdown_event.wait())

    def _build_viser_gui(self):
        """
        Build the Viser GUI for replay controls.
        Includes file dropdown for episode selection,
        play/pause button, speed slider, and frame slider.
        """
        gui = self._viser_server.server.gui
        with gui.add_folder("Replay Recorded Episodes", order=0):
            # file dropdown
            self._file_dropdown = gui.add_dropdown(
                "Episode File",
                options=[f.name for f in self._episode_files],
                initial_value=self._current_file.name,
            )
            # play/pause button
            self._play_pause_button = gui.add_button("Play")
            # speed slider
            self._speed_slider = gui.add_slider(
                "Playback Speed (xFPS)",
                min=0.1,
                max=4.0,
                step=0.05,
                initial_value=1.0,
            )
            # frame slider
            self._frame_slider = gui.add_slider(
                "Frame",
                min=0,
                max=max(self._episode_length() - 1, 0),
                step=1,
                initial_value=0,
            )
        # add callbacks
        self._file_dropdown.on_update(
            lambda _event: self._on_file_dropdown_update()
        )
        self._play_pause_button.on_click(
            lambda _event: self._on_play_pause_button_click()
        )
        self._speed_slider.on_update(
            lambda _event: self._on_speed_slider_update()
        )
        self._frame_slider.on_update(
            lambda _event: self._on_frame_slider_update()
        )

    def _load_current_file(self):
        """Load targets and any IK solutions from the current file."""
        selected_file_name = self._current_file.name
        target_path = self._config.input_dir / selected_file_name
        self._current_targets = load_target_episode(target_path)
        try:
            self._current_solutions = load_ik_episode(target_path)
        except Exception:
            self._current_solutions = None

    def _episode_length(self) -> int:
        if self._current_solutions is not None:
            return len(self._current_solutions)
        return len(self._current_targets)

    def _on_file_dropdown_update(self):
        """Load selected episode file."""
        self._gui_update_event.set()
        selected_file_name = self._file_dropdown.value
        # load new episode
        self._current_file = self._config.input_dir / selected_file_name
        self._load_current_file()
        self._current_frame_idx = 0
        # update frame slider max
        self._frame_slider.max = max(self._episode_length() - 1, 0)
        self._frame_slider.value = 0
        self._gui_update_event.clear()

    def _on_play_pause_button_click(self):
        """Toggle play/pause state."""
        self._is_playing = not self._is_playing
        new_label = "Pause" if self._is_playing else "Play"
        self._play_pause_button.label = new_label

    def _on_speed_slider_update(self):
        """Update playback speed."""
        new_speed = self._speed_slider.value
        self._gui_update_event.set()
        self._playback_speed = new_speed
        self._gui_update_event.clear()

    def _on_frame_slider_update(self):
        """Update current frame index."""
        new_frame_idx = int(self._frame_slider.value)
        self._gui_update_event.set()
        self._current_frame_idx = new_frame_idx
        self._gui_update_event.clear()

    async def _playback_loop(self):
        """Main playback loop."""
        while not self._shutdown_event.is_set():
            if self._gui_update_event.is_set():
                await asyncio.sleep(0.1)
                continue
            # get current frame
            target_frame = self._current_targets[self._current_frame_idx]
            ik_frame: IKSolution | None = None
            if self._current_solutions is not None:
                ik_frame = self._current_solutions[self._current_frame_idx]
            # visualize frame
            if ik_frame is not None:
                self._viser_server.visualize_ik_solution(ik_frame)
            else:
                self._viser_server.visualize_target(target_frame)
            if not self._is_playing:
                await asyncio.sleep(0.1)
                continue
            # update frame index and sleep time
            next_frame_idx = self._current_frame_idx + 1
            if next_frame_idx >= self._episode_length():
                # back to start
                next_frame_idx = 0
                sleep_time = 0
            else:
                next_target = self._current_targets[next_frame_idx]
                curr_ts = (
                    ik_frame.target.timestamp
                    if ik_frame is not None
                    else target_frame.timestamp
                )
                next_ts = (
                    self._current_solutions[next_frame_idx].target.timestamp
                    if self._current_solutions is not None
                    else next_target.timestamp
                )
                time_diff = max(next_ts - curr_ts, 0.0)
                sleep_time = time_diff / self._playback_speed
            self._current_frame_idx = next_frame_idx
            self._frame_slider.value = self._current_frame_idx
            await asyncio.sleep(sleep_time)

    async def _scan_files_loop(self):
        """Periodically scan for new episode files."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(1.0)
            if not self._gui_update_event.is_set():
                # only scan when not updating GUI
                current_files = sorted(
                    self._config.input_dir.glob("*recording*.json")
                )
                if len(current_files) != len(self._episode_files):
                    self._episode_files = current_files
                    # update dropdown options
                    self._file_dropdown.options = [
                        f.name for f in self._episode_files
                    ]


def main(
    config: ReplayConfig,
):
    """Replay recorded episodes using the Viser server."""
    replayer = config.crate()
    replayer.run()


if __name__ == "__main__":
    config = tyro.cli(ReplayConfig)
    main(config)


def cli():
    config = tyro.cli(ReplayConfig)
    main(config)
