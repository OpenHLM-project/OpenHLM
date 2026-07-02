import json
from pathlib import Path
from typing import Annotated, TypedDict

import tyro


def trim_episode(
    input_json_path: Annotated[Path, tyro.conf.arg(aliases=["-i"])],
    start_index: Annotated[int, tyro.conf.arg(aliases=["-s"])],
    end_index: Annotated[int, tyro.conf.arg(aliases=["-e"])],
    output_json_path: Annotated[
        Path | None, tyro.conf.arg(aliases=["-o"])
    ] = None,
):
    """Trim an episode JSON file to only include frames between start_index and end_index.

    Args:
        input_json_path (Path): Path to the input JSON file.
        start_index (int): The starting index (inclusive).
        end_index (int): The ending index (exclusive).
        output_json_path (Path | None, optional): Output path for the trimmed JSON file.
            If None, saves to new directory with '_trimmed' suffix. Defaults to None.
    """

    class EpisodeDict(TypedDict):
        episode: list[dict]

    with open(input_json_path, "r") as f:
        input_dict: EpisodeDict = json.load(f)

    episode = input_dict["episode"]
    trimmed_episode = episode[start_index:end_index]

    input_dict["episode"] = trimmed_episode

    if output_json_path is None:
        input_dir = input_json_path.parent
        output_dir = input_dir.with_name(input_dir.name + "_trimmed")
        output_json_path = output_dir / input_json_path.name

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving trimmed episode to {output_json_path}")
    with open(output_json_path, "w") as f:
        json.dump(input_dict, f, indent=4)


if __name__ == "__main__":
    tyro.cli(trim_episode)


def cli():
    tyro.cli(trim_episode)
