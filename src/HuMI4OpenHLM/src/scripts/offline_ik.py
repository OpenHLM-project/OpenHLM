import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated

import tyro
from mink.exceptions import NoSolutionFound
from tqdm import tqdm

from ikumi.config import IKConfig, get_config
from ikumi.episodes import (
    load_episode_prompt,
    load_ik_episode,
    save_ik_episode,
)
from ikumi.solution import IKSolution
from ikumi.solver import IKSolver


@dataclass(frozen=True)
class OfflineIKConfig:
    """Configuration for offline IK."""

    ik: IKConfig = IKConfig()

    trim_start_index: Annotated[int, tyro.conf.arg(aliases=["-ts"])] = 20
    """Trim the start of the episode to remove jerky initial frames after reset."""

    input_dir: Annotated[Path, tyro.conf.arg(aliases=["-i"])] = tyro.MISSING
    """Directory containing recorded episodes in JSON format."""
    output_dir: Annotated[Path | None, tyro.conf.arg(aliases=["-o"])] = None
    """
    Directory to save the recomputed IK solutions.
    If None, write in input_dir with suffix "_ik_recomputed".
    """
    max_workers: Annotated[int | None, tyro.conf.arg(aliases=["-j"])] = None
    """
    Maximum concurrent episodes to process. If None, defaults to CPU core count.
    """

    inner_bar: bool = False
    """Whether to show inner frame progress bars."""


default_configs = {
    "proposal": (
        "Proposal task.",
        OfflineIKConfig(ik=get_config("proposal")),
    ),
    "walk": (
        "Walk task.",
        OfflineIKConfig(ik=get_config("walk")),
    ),
    "squat_pick_ground": (
        "Pick ground squat task.",
        OfflineIKConfig(ik=get_config("squat_pick_ground")),
    ),
    "pick-low": (
        "Pick low task.",
        OfflineIKConfig(ik=get_config("pick-low")),
    ),
    "toss": (
        "Toss task.",
        OfflineIKConfig(ik=get_config("toss")),
    ),
    "unsheathe": (
        "Unsheathe sword task.",
        OfflineIKConfig(ik=get_config("unsheathe")),
    ),
    "pick": (
        "Pick something and put it on the mouse pad.",
        OfflineIKConfig(ik=get_config("pick")),
    ),
    "shelf": (
        "Shelf task derived from pick without using the waist tracker.",
        OfflineIKConfig(ik=get_config("shelf")),
    ),
}


def process_episode(
    ep_id: int,
    ep_path: Path,
    trim_start_index: int,
    output_dir: Path,
    ik_solver: IKSolver,
    total_episodes: int,
    show_inner_bar: bool,
) -> None:
    """Process a single episode: recompute IK for all frames and save.

    Args:
        ep_id: Episode index (0-based).
        ep_path: Path to the episode JSON file.
        trim_start_index: Index to start processing the episode from.
        output_dir: Directory to save recomputed episode.
        ik_solver: IK solver instance to use for recomputation.
        total_episodes: Total count for display only.
        show_inner_bar: Whether to show inner progress bar.
    """
    output_path = output_dir / ep_path.name
    skip = False
    if output_path.is_file():
        # If it's a normal json file, skip existing
        try:
            load_ik_episode(json_path=output_path)
        except Exception:
            pass
        else:
            skip = True
    if skip:
        return
    ik_config = ik_solver.config
    target_episode = ik_solver.load_targets(json_path=ep_path)
    prompt = load_episode_prompt(ep_path)
    pbar = tqdm(
        total=len(target_episode),
        desc=f"Episode {ep_id + 1}/{total_episodes}: {ep_path.name}",
        position=ep_id + 1,  # keep 0 for overall episodes bar
        dynamic_ncols=True,
        leave=False,
        mininterval=0.2,
        disable=not show_inner_bar,
    )
    recomputed_frames: list[IKSolution] = []
    last_timestamp: float | None = None
    for f_id, target_update in enumerate(target_episode):
        if f_id == 0:
            ik_solver.recompute_reference_transform(
                target_update=target_update
            )
        # Compute actual dt from timestamps; default to config dt for first frame
        if last_timestamp is None:
            dt_actual = ik_config.dt
        else:
            dt_actual = max(target_update.timestamp - last_timestamp, 1e-6)
        last_timestamp = target_update.timestamp
        try:
            new_sol = ik_solver.solve_one_step(target_update, dt=dt_actual)
        except NoSolutionFound as e:
            raise RuntimeError(
                f"Failed to compute IK for frame {f_id} in episode {ep_path}"
            ) from e
        if f_id >= trim_start_index:
            recomputed_frames.append(new_sol)
        pbar.update(1)
    save_ik_episode(
        recomputed_frames,
        ik_config=ik_config,
        output_path=output_path,
        prompt=prompt,
    )
    pbar.close()


def main(
    config: OfflineIKConfig,
):
    """Run offline IK recomputation on recorded episodes."""

    episode_files = sorted(config.input_dir.glob("recording*.json"))
    if len(episode_files) == 0:
        raise ValueError(f"No episode files found in {config.input_dir}")
    output_dir = (
        config.output_dir
        if config.output_dir is not None
        else config.input_dir.with_name(
            config.input_dir.name
            + "_ik_recomputed"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ik_config = config.ik
    # Restore ik config
    ik_config.save_as_yaml(output_dir / "ik_config_used.yaml")

    ik_solver = ik_config.create()

    # Parallelize episodes using ProcessPoolExecutor sized to requested or CPU cores
    requested_workers = (
        config.max_workers
        if config.max_workers is not None
        else (os.cpu_count() or 1)
    )
    if requested_workers == -1:
        # Serial mode: no multiprocessing
        overall_bar = tqdm(
            total=len(episode_files),
            desc="Episodes",
            position=0,
            dynamic_ncols=True,
            leave=False,
        )
        for ep_id, ep in enumerate(episode_files):
            process_episode(
                ep_id=ep_id,
                ep_path=ep,
                trim_start_index=config.trim_start_index,
                output_dir=output_dir,
                ik_solver=ik_solver,
                total_episodes=len(episode_files),
                show_inner_bar=config.inner_bar,
            )
            overall_bar.update(1)
        overall_bar.close()
        return
    if requested_workers < 1:
        raise ValueError("max_workers must be >= 1 or -1 for serial mode")
    max_workers = min(len(episode_files), requested_workers)
    # Use a 'fork' context so child processes inherit the shared tqdm lock
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=ctx
    ) as executor:
        futures = [
            executor.submit(
                process_episode,
                ep_id,
                ep,
                config.trim_start_index,
                output_dir,
                ik_solver,
                len(episode_files),
                config.inner_bar,
            )
            for ep_id, ep in enumerate(episode_files)
        ]
        overall_bar = tqdm(
            total=len(futures),
            desc="Episodes",
            position=0,
            dynamic_ncols=True,
            leave=False,
        )
        for fut in as_completed(futures):
            overall_bar.update(1)
            # Propagate exceptions if any
            try:
                fut.result()
            except Exception as e:
                print(f"Error with job {fut}: {e}")
        overall_bar.close()


def cli():
    cfg = tyro.extras.overridable_config_cli(default_configs)
    main(cfg)


if __name__ == "__main__":
    cli()
