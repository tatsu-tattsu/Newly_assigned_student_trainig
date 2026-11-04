from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

def require_executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(
            f"Required executable '{name}' was not found in PATH. "
            "Activate the monomer-viscosity conda environment first."
        )
    return path

def run_command(
    cmd: list[str],
    cwd: Path,
    log_file: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    cwd.mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print("$", " ".join(map(str, cmd)))
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w") as fh:
            p = subprocess.run(cmd, cwd=cwd, env=merged, stdout=fh, stderr=subprocess.STDOUT)
    else:
        p = subprocess.run(cmd, cwd=cwd, env=merged)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {p.returncode}: {' '.join(map(str, cmd))}"
        )

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def atomic_mass(symbol: str) -> float:
    masses = {
        "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
        "F": 18.998403163, "P": 30.973761998, "S": 32.06,
        "Cl": 35.45, "Br": 79.904, "I": 126.90447,
    }
    if symbol not in masses:
        raise ValueError(f"Unsupported element for mass table: {symbol}")
    return masses[symbol]

def rotation_matrix(rng):
    # Uniform random 3-D rotation using a random unit quaternion.
    import numpy as np
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])

def which_or_raise(names: Iterable[str]) -> dict[str, str]:
    return {name: require_executable(name) for name in names}
