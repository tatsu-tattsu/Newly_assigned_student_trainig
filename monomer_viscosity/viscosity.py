from __future__ import annotations

from pathlib import Path
import re
import numpy as np


# Real-unit LAMMPS pressure is atm and volume is Angstrom^3.
# Green-Kubo prefactor gives pressure^2*time*volume/(kBT).
# Conversion:
# 1 atm * Angstrom^3 = 1.01325e-25 J
# 1 ps = 1e-12 s
# J*s/m^3 = Pa*s
ATM_A3_TO_J = 1.01325e-25
PS_TO_S = 1.0e-12
ANGSTROM3_TO_M3 = 1.0e-30


def _read_fix_ave_correlate(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 5:
            continue
        try:
            vals = [float(x) for x in p]
        except ValueError:
            continue
        rows.append(vals)
    if not rows:
        raise RuntimeError(f"No numerical data found in {path}")
    return np.asarray(rows)


def _read_volume(path: Path):
    vals = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        try:
            # step V T ...
            if len(p) >= 3:
                vals.append(float(p[1]))
        except ValueError:
            pass
    if not vals:
        raise RuntimeError("No volume data found.")
    return float(np.mean(vals[int(len(vals) * 0.5):]))


def green_kubo(
    corr_file: Path,
    volume_A3: float,
    temperature_K: float,
    dt_sample_ps: float,
):
    data = _read_fix_ave_correlate(corr_file)

    # fix ave/correlate type=auto writes:
    # Index, TimeDelta, Ncount, C11, C22, C33
    # for three input quantities. With ave=running, multiple blocks are
    # appended; the final block contains the most completely averaged
    # correlation function and is used here.
    if data.shape[1] < 6:
        raise RuntimeError(
            "Unexpected fix ave/correlate format. "
            f"Expected >=6 columns, got {data.shape[1]}."
        )
    # Find the last contiguous correlation block. Each block starts at
    # TimeDelta=0, so the final rows from the last zero are sufficient.
    zero_idx = np.where(np.isclose(data[:, 1], 0.0))[0]
    if len(zero_idx):
        data = data[zero_idx[-1]:]
    time_steps = data[:, 1]
    cxy = data[:, 3]
    cxz = data[:, 4]
    cyz = data[:, 5]

    t_ps = time_steps * dt_sample_ps

    def integrate(c):
        return np.trapezoid(c, t_ps)

    ixy = integrate(cxy)
    ixz = integrate(cxz)
    iyz = integrate(cyz)

    # C(t) has atm^2. V is A^3 and dt is ps.
    # atm^2 * A^3 * ps -> Pa*s
    # (1 atm)^2*(A^3)*(ps) / (kB*T)
    # Convert one atm to Pa and A^3 to m^3 directly.
    kB = 1.380649e-23
    atm_to_pa = 101325.0
    prefactor = (volume_A3 * ANGSTROM3_TO_M3) * (atm_to_pa**2) * (PS_TO_S) / (kB * temperature_K)

    eta_xy = prefactor * ixy
    eta_xz = prefactor * ixz
    eta_yz = prefactor * iyz
    eta_avg = (eta_xy + eta_xz + eta_yz) / 3.0

    return {
        "eta_xy_Pa_s": float(eta_xy),
        "eta_xz_Pa_s": float(eta_xz),
        "eta_yz_Pa_s": float(eta_yz),
        "eta_avg_Pa_s": float(eta_avg),
        "eta_xy_mPa_s": float(eta_xy * 1000.0),
        "eta_xz_mPa_s": float(eta_xz * 1000.0),
        "eta_yz_mPa_s": float(eta_yz * 1000.0),
        "eta_avg_mPa_s": float(eta_avg * 1000.0),
        "volume_A3": float(volume_A3),
        "temperature_K": float(temperature_K),
    }
