#!/usr/bin/env python3
"""
Transform TSN's celestial-settings.json into the orbital parameters JSON
used by tychos_skyfield.

Usage:
    python scripts/sync_params.py

Reads from the TSN submodule at TSN/src/settings/celestial-settings.json
and writes to tychos_skyfield/orbital_params.json.

This script does three things:
  1. Filters to only the bodies tychos_skyfield uses
  2. Renames TSN display names to tychos_skyfield internal keys
  3. Renames TSN's camelCase property names to snake_case

The output is pure JSON — no code generation.
"""

import json
import sys
from pathlib import Path

# TSN display name -> tychos_skyfield internal key.
# Bodies present in TSN but absent here (e.g. SystemCenter, Pluto) are skipped.
TSN_TO_TYCHOS = {
    "Earth": "earth",
    "Moon deferent A": "moon_def_a",
    "Moon deferent B": "moon_def_b",
    "Moon": "moon",
    "Sun deferent": "sun_def",
    "Sun": "sun",
    "Mercury def A": "mercury_def_a",
    "Mercury def B": "mercury_def_b",
    "Mercury": "mercury",
    "Venus deferent A": "venus_def_a",
    "Venus deferent B": "venus_def_b",
    "Venus": "venus",
    "Mars E deferent": "mars_def_e",
    "Mars S deferent": "mars_def_s",
    "Mars": "mars",
    "Phobos": "phobos",
    "Deimos": "deimos",
    "Jupiter deferent": "jupiter_def",
    "Jupiter": "jupiter",
    "Saturn deferent": "saturn_def",
    "Saturn": "saturn",
    "Uranus deferent": "uranus_def",
    "Uranus": "uranus",
    "Neptune deferent": "neptune_def",
    "Neptune": "neptune",
    "Halleys deferent": "halleys_def",
    "Halleys": "halleys",
    "Eros deferent A": "eros_def_a",
    "Eros deferent B": "eros_def_b",
    "Eros": "eros",
}

# Properties to extract from TSN and their snake_case renames.
PARAM_MAP = {
    "orbitRadius": "orbit_radius",
    "orbitCentera": "orbit_center_a",
    "orbitCenterb": "orbit_center_b",
    "orbitCenterc": "orbit_center_c",
    "orbitTilta": "orbit_tilt_a",
    "orbitTiltb": "orbit_tilt_b",
    "startPos": "start_pos",
    "speed": "speed",
}

ZERO_PARAMS = {v: 0.0 for v in PARAM_MAP.values()}


def transform(settings):
    """Transform TSN celestial-settings into tychos_skyfield format."""
    params = {}
    for entry in settings:
        tsn_name = entry["name"]
        if tsn_name not in TSN_TO_TYCHOS:
            continue
        key = TSN_TO_TYCHOS[tsn_name]
        params[key] = {
            snake: entry.get(camel, 0.0)
            for camel, snake in PARAM_MAP.items()
        }

    # polar_axis is not in TSN — it's a synthetic object with zero params.
    params["polar_axis"] = dict(ZERO_PARAMS)

    return params


def main():
    repo_root = Path(__file__).resolve().parent.parent
    json_path = repo_root / "TSN" / "src" / "settings" / "celestial-settings.json"

    if not json_path.exists():
        print(
            f"ERROR: {json_path} not found.\n"
            "Make sure the TSN submodule is initialized:\n"
            "  git submodule update --init",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Reading: {json_path}")
    with open(json_path) as f:
        settings = json.load(f)

    params = transform(settings)

    out_path = repo_root / "tychos_skyfield" / "orbital_params.json"
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
        f.write("\n")

    print(f"Written: {out_path} ({len(params)} bodies)")


if __name__ == "__main__":
    main()
