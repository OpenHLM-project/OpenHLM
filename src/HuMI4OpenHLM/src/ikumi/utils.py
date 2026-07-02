from __future__ import annotations

import mujoco


def geom_group_from_bodies(
    model: mujoco.MjModel, body_names: list[str]
) -> list[int]:
    """Get a list of unique geom IDs associated with the given body names."""
    geom_ids: list[int] = []
    for body_name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body {body_name!r} not found in model")
        geom_adr = model.body_geomadr[body_id]
        geom_num = model.body_geomnum[body_id]
        for gid in range(geom_adr, geom_adr + geom_num):
            geom_ids.append(gid)
    unique: list[int] = []
    seen: set[int] = set()
    for gid in geom_ids:
        if gid not in seen:
            unique.append(gid)
            seen.add(gid)
    return unique


__all__ = ["geom_group_from_bodies"]
