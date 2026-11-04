from __future__ import annotations

from pathlib import Path
import os
import subprocess
from collections import OrderedDict

import parmed as pmd


def _float_key(*values, ndigits=10):
    """Floating-point parametersを安定して辞書キー化する。"""
    return tuple(round(float(v), ndigits) for v in values)


def _get_atom_lj(atom):
    """
    ParmEd AtomからLJ parametersを取得する。

    ParmEdではsigmaはLJの通常のsigma [Å]、
    epsilonはkcal/molとして扱われる。
    """
    atom_type = atom.atom_type

    if atom_type is None:
        raise ValueError(
            f"Atom {atom.idx + 1} ({atom.name}) has no AtomType."
        )

    try:
        sigma = float(atom_type.sigma)
        epsilon = float(atom_type.epsilon)
    except AttributeError as exc:
        raise ValueError(
            f"Could not obtain LJ parameters for atom "
            f"{atom.idx + 1} ({atom.name}, type={atom.type})."
        ) from exc

    return sigma, epsilon


def _assign_atom_types(parm):
    """
    ParmEdのatom typesをLAMMPS atom typeへ変換する。

    LAMMPS側では同じmass/sigma/epsilonを持つものを
    同一atom typeとして扱う。
    """
    type_map = OrderedDict()
    atom_type_ids = {}

    for atom in parm.atoms:
        sigma, epsilon = _get_atom_lj(atom)

        key = _float_key(
            atom.mass,
            sigma,
            epsilon,
        )

        if key not in type_map:
            type_map[key] = len(type_map) + 1

        atom_type_ids[atom.idx] = type_map[key]

    return type_map, atom_type_ids


def _assign_bond_types(parm):
    """
    BondTypeをLAMMPS bond typeへ変換。
    """
    type_map = OrderedDict()
    bond_type_ids = {}

    for bond in parm.bonds:
        if bond.type is None:
            raise ValueError(
                f"Bond between atoms "
                f"{bond.atom1.idx + 1}-{bond.atom2.idx + 1} "
                f"has no parameters."
            )

        key = _float_key(
            bond.type.k,
            bond.type.req,
        )

        if key not in type_map:
            type_map[key] = len(type_map) + 1

        bond_type_ids[id(bond)] = type_map[key]

    return type_map, bond_type_ids


def _assign_angle_types(parm):
    """
    AngleTypeをLAMMPS angle typeへ変換。
    """
    type_map = OrderedDict()
    angle_type_ids = {}

    for angle in parm.angles:
        if angle.type is None:
            raise ValueError(
                f"Angle "
                f"{angle.atom1.idx + 1}-"
                f"{angle.atom2.idx + 1}-"
                f"{angle.atom3.idx + 1} "
                f"has no parameters."
            )

        key = _float_key(
            angle.type.k,
            angle.type.theteq,
        )

        if key not in type_map:
            type_map[key] = len(type_map) + 1

        angle_type_ids[id(angle)] = type_map[key]

    return type_map, angle_type_ids


def _dihedral_terms(dihedral):
    """
    ParmEd DihedralからFourier termsを取得する。

    ParmEdでは多項式torsionがDihedralTypeListとして
    複数termを持つことがある。
    """
    dtype = dihedral.type

    if dtype is None:
        raise ValueError(
            f"Dihedral "
            f"{dihedral.atom1.idx + 1}-"
            f"{dihedral.atom2.idx + 1}-"
            f"{dihedral.atom3.idx + 1}-"
            f"{dihedral.atom4.idx + 1} "
            f"has no parameters."
        )

    # 多項項の場合
    if hasattr(dtype, "__iter__") and not hasattr(dtype, "phi_k"):
        terms = list(dtype)
    else:
        terms = [dtype]

    result = []

    for term in terms:
        result.append(
            (
                float(term.phi_k),
                int(term.per),
                float(term.phase),
            )
        )

    return result


def _assign_dihedral_types(parm):
    """
    ParmEdのdihedralをLAMMPS fourier typeへ変換。

    同一のFourier展開を持つものを同じtypeにまとめる。
    """
    type_map = OrderedDict()
    dihedral_type_ids = {}

    for dihedral in parm.dihedrals:
        terms = _dihedral_terms(dihedral)

        key = tuple(
            _float_key(k, n, phase)
            for k, n, phase in terms
        )

        if key not in type_map:
            type_map[key] = len(type_map) + 1

        dihedral_type_ids[id(dihedral)] = type_map[key]

    return type_map, dihedral_type_ids


def _assign_improper_types(parm):
    """
    ParmEdのImproperTypeをLAMMPS improper harmonicへ変換。
    """
    type_map = OrderedDict()
    improper_type_ids = {}

    for improper in parm.impropers:
        if improper.type is None:
            raise ValueError(
                f"Improper involving atoms "
                f"{improper.atom1.idx + 1}-"
                f"{improper.atom2.idx + 1}-"
                f"{improper.atom3.idx + 1}-"
                f"{improper.atom4.idx + 1} "
                f"has no parameters."
            )

        key = _float_key(
            improper.type.psi_k,
            improper.type.psi_eq,
        )

        if key not in type_map:
            type_map[key] = len(type_map) + 1

        improper_type_ids[id(improper)] = type_map[key]

    return type_map, improper_type_ids


def _write_section_header(fh, title):
    fh.write(f"\n{title}\n\n")


def _write_lammps_data(
    parm,
    data_file: Path,
):
    """
    ParmEd Structureを直接LAMMPS data fileへ変換する。

    mBuildには依存しない。
    LAMMPS units real / atom_style fullを想定。

    Dihedral:
        LAMMPS dihedral_style fourier

    Improper:
        LAMMPS improper_style harmonic
    """

    atom_types, atom_type_ids = _assign_atom_types(parm)
    bond_types, bond_type_ids = _assign_bond_types(parm)
    angle_types, angle_type_ids = _assign_angle_types(parm)
    dihedral_types, dihedral_type_ids = _assign_dihedral_types(parm)
    improper_types, improper_type_ids = _assign_improper_types(parm)

    natoms = len(parm.atoms)
    nbonds = len(parm.bonds)
    nangles = len(parm.angles)
    ndihedrals = len(parm.dihedrals)
    nimpropers = len(parm.impropers)

    # Periodic box
    if parm.box is None:
        raise ValueError(
            "The ParmEd structure does not contain a periodic box."
        )

    if len(parm.box) < 6:
        raise ValueError(
            f"Invalid periodic box information: {parm.box}"
        )

    a, b, c, alpha, beta, gamma = map(float, parm.box[:6])

    # Amber/tleapから来るboxは通常orthorhombic。
    # triclinicも扱えるようにLAMMPS restricted triclinicへ変換する。
    import math

    alpha_r = math.radians(alpha)
    beta_r = math.radians(beta)
    gamma_r = math.radians(gamma)

    lx = a
    xy = b * math.cos(gamma_r)

    xz = c * math.cos(beta_r)

    sin_gamma = math.sin(gamma_r)

    if abs(sin_gamma) < 1.0e-12:
        raise ValueError("Invalid periodic box: gamma is too close to 0/180 degrees.")

    ly = b * sin_gamma

    yz = (
        c * (math.cos(alpha_r) - math.cos(beta_r) * math.cos(gamma_r))
        / sin_gamma
    )

    lz_sq = c * c - xz * xz - yz * yz

    if lz_sq <= 0:
        raise ValueError(
            "Invalid periodic box: calculated Lz^2 <= 0."
        )

    lz = math.sqrt(lz_sq)

    xlo = 0.0
    ylo = 0.0
    zlo = 0.0

    xhi = lx
    yhi = ly
    zhi = lz

    data_file.parent.mkdir(parents=True, exist_ok=True)

    with data_file.open("w") as fh:

        fh.write(
            "LAMMPS data file generated directly from ParmEd\n"
        )

        fh.write(f"\n{natoms} atoms\n")
        fh.write(f"{nbonds} bonds\n")
        fh.write(f"{nangles} angles\n")
        fh.write(f"{ndihedrals} dihedrals\n")
        fh.write(f"{nimpropers} impropers\n")

        fh.write(f"\n{len(atom_types)} atom types\n")
        fh.write(f"{len(bond_types)} bond types\n")
        fh.write(f"{len(angle_types)} angle types\n")
        fh.write(f"{len(dihedral_types)} dihedral types\n")
        fh.write(f"{len(improper_types)} improper types\n")

        fh.write(f"\n{xlo:.10f} {xhi:.10f} xlo xhi\n")
        fh.write(f"{ylo:.10f} {yhi:.10f} ylo yhi\n")
        fh.write(f"{zlo:.10f} {zhi:.10f} zlo zhi\n")

        # Restricted triclinic box parameters
        if (
            abs(xy) > 1.0e-12
            or abs(xz) > 1.0e-12
            or abs(yz) > 1.0e-12
        ):
            fh.write(
                f"{xy:.10f} {xz:.10f} {yz:.10f} xy xz yz\n"
            )

        # ---------------------------------------------------------
        # Masses
        # ---------------------------------------------------------
        _write_section_header(fh, "Masses")

        mass_by_type = {}

        for atom in parm.atoms:
            tid = atom_type_ids[atom.idx]
            mass_by_type[tid] = float(atom.mass)

        for tid in sorted(mass_by_type):
            fh.write(
                f"{tid:6d} "
                f"{mass_by_type[tid]:16.8f}\n"
            )

        # ---------------------------------------------------------
        # Pair Coeffs
        # ---------------------------------------------------------
        _write_section_header(fh, "Pair Coeffs")

        pair_by_type = {}

        for atom in parm.atoms:
            tid = atom_type_ids[atom.idx]
            sigma, epsilon = _get_atom_lj(atom)
            pair_by_type[tid] = (epsilon, sigma)

        for tid in sorted(pair_by_type):
            epsilon, sigma = pair_by_type[tid]

            fh.write(
                f"{tid:6d} "
                f"{epsilon:16.10f} "
                f"{sigma:16.10f}\n"
            )

        # ---------------------------------------------------------
        # Bond Coeffs
        # ---------------------------------------------------------
        _write_section_header(fh, "Bond Coeffs")

        for (k, req), tid in sorted(
            ((key, value) for key, value in bond_types.items()),
            key=lambda x: x[1],
        ):
            fh.write(
                f"{tid:6d} "
                f"{k:16.10f} "
                f"{req:16.10f}\n"
            )

        # ---------------------------------------------------------
        # Angle Coeffs
        # ---------------------------------------------------------
        _write_section_header(fh, "Angle Coeffs")

        for (k, theta), tid in sorted(
            ((key, value) for key, value in angle_types.items()),
            key=lambda x: x[1],
        ):
            fh.write(
                f"{tid:6d} "
                f"{k:16.10f} "
                f"{theta:16.10f}\n"
            )

        # ---------------------------------------------------------
        # Dihedral Coeffs
        # ---------------------------------------------------------
        if dihedral_types:
            _write_section_header(fh, "Dihedral Coeffs")

            for key, tid in sorted(
                dihedral_types.items(),
                key=lambda x: x[1],
            ):
                fh.write(
                    f"{tid:6d} "
                    f"{len(key):d}"
                )

                for k, n, phase in key:
                    fh.write(
                        f" {k:.10f} {int(n)} {phase:.10f}"
                    )

                fh.write("\n")

        # ---------------------------------------------------------
        # Improper Coeffs
        # ---------------------------------------------------------
        if improper_types:
            _write_section_header(fh, "Improper Coeffs")

            for (k, psi_eq), tid in sorted(
                improper_types.items(),
                key=lambda x: x[1],
            ):
                fh.write(
                    f"{tid:6d} "
                    f"{k:16.10f} "
                    f"{psi_eq:16.10f}\n"
                )

        # ---------------------------------------------------------
        # Atoms
        # atom_style full:
        # atom-ID molecule-ID atom-type charge x y z
        # ---------------------------------------------------------
        _write_section_header(fh, "Atoms # full")

        for atom in parm.atoms:

            atom_id = atom.idx + 1
            molecule_id = (
                atom.residue.idx + 1
                if atom.residue is not None
                else 1
            )

            atom_type = atom_type_ids[atom.idx]
            charge = float(atom.charge)

            x = float(atom.xx)
            y = float(atom.xy)
            z = float(atom.xz)

            fh.write(
                f"{atom_id:8d} "
                f"{molecule_id:8d} "
                f"{atom_type:6d} "
                f"{charge:18.10f} "
                f"{x:18.10f} "
                f"{y:18.10f} "
                f"{z:18.10f}\n"
            )

        # ---------------------------------------------------------
        # Bonds
        # ---------------------------------------------------------
        _write_section_header(fh, "Bonds")

        for i, bond in enumerate(parm.bonds, start=1):
            a1 = bond.atom1.idx + 1
            a2 = bond.atom2.idx + 1
            tid = bond_type_ids[id(bond)]

            fh.write(
                f"{i:8d} {tid:6d} "
                f"{a1:8d} {a2:8d}\n"
            )

        # ---------------------------------------------------------
        # Angles
        # ---------------------------------------------------------
        _write_section_header(fh, "Angles")

        for i, angle in enumerate(parm.angles, start=1):
            a1 = angle.atom1.idx + 1
            a2 = angle.atom2.idx + 1
            a3 = angle.atom3.idx + 1
            tid = angle_type_ids[id(angle)]

            fh.write(
                f"{i:8d} {tid:6d} "
                f"{a1:8d} {a2:8d} {a3:8d}\n"
            )

        # ---------------------------------------------------------
        # Dihedrals
        # ---------------------------------------------------------
        if ndihedrals > 0:
            _write_section_header(fh, "Dihedrals")

            for i, dihedral in enumerate(parm.dihedrals, start=1):
                a1 = dihedral.atom1.idx + 1
                a2 = dihedral.atom2.idx + 1
                a3 = dihedral.atom3.idx + 1
                a4 = dihedral.atom4.idx + 1

                tid = dihedral_type_ids[id(dihedral)]

                fh.write(
                    f"{i:8d} {tid:6d} "
                    f"{a1:8d} {a2:8d} "
                    f"{a3:8d} {a4:8d}\n"
                )

        # ---------------------------------------------------------
        # Impropers
        # ---------------------------------------------------------
        if nimpropers > 0:
            _write_section_header(fh, "Impropers")

            for i, improper in enumerate(parm.impropers, start=1):
                a1 = improper.atom1.idx + 1
                a2 = improper.atom2.idx + 1
                a3 = improper.atom3.idx + 1
                a4 = improper.atom4.idx + 1

                tid = improper_type_ids[id(improper)]

                fh.write(
                    f"{i:8d} {tid:6d} "
                    f"{a1:8d} {a2:8d} "
                    f"{a3:8d} {a4:8d}\n"
                )

    return data_file


def amber_to_lammps(
    prmtop: Path,
    inpcrd: Path,
    outdir: Path,
):
    """
    AmberTools/GAFF2のprmtop + inpcrdを
    mBuildなしでLAMMPS dataへ変換する。

    Returns
    -------
    data : Path
        Generated LAMMPS data file.

    parm : parmed.Structure
        Loaded ParmEd structure.
    """

    outdir.mkdir(parents=True, exist_ok=True)

    parm = pmd.load_file(
        str(prmtop),
        str(inpcrd),
    )

    data = outdir / "system.data"

    _write_lammps_data(
        parm,
        data,
    )

    return data, parm


def write_lammps_input(
    data_file: Path,
    out_file: Path,
    temperature: float,
    pressure_atm: float,
    timestep_fs: float,
    nvt_steps: int,
    npt_steps: int,
    gk_steps: int,
    thermo_every: int,
    corr_every: int,
    corr_points: int,
    neighbor_skin_A: float,
    lj_cutoff_A: float,
    pppm_accuracy: float,
):
    """
    LAMMPS input scriptを生成する。

    流れ:
        1. read_data
        2. Energy minimization
        3. 低温NVTによる初期構造緩和
        4. 指定温度NVT
        5. 指定温度NPT
        6. Green-Kubo NVE

    GAFF2 / AMBER convention:
        special_bonds amber

    Green-Kubo:
        pressure tensor autocorrelation
        Pxy, Pxz, Pyz
    """

    total_corr = corr_every * corr_points

    # data_fileは通常input fileと同じディレクトリに置く。
    data_name = data_file.name

    out_file.write_text(
        f"""# Auto-generated monomer viscosity calculation
# Generated without mBuild

units           real
atom_style      full
boundary        p p p

# ------------------------------------------------------------
# Force-field styles
# ------------------------------------------------------------

pair_style      lj/cut/coul/long {lj_cutoff_A:.3f}

bond_style      harmonic
angle_style     harmonic
dihedral_style  fourier
improper_style  harmonic

pair_modify     mix arithmetic

# AMBER convention:
# 1-2 and 1-3 nonbonded interactions excluded
# 1-4 electrostatic = 1/1.2 = 0.833333...
# 1-4 LJ            = 1/2.0 = 0.5
special_bonds    amber

kspace_style    pppm {pppm_accuracy:.1e}

read_data       {data_name}

neighbor        {neighbor_skin_A:.3f} bin
neigh_modify    delay 0 every 1 check yes

# ------------------------------------------------------------
# Thermodynamic output
# ------------------------------------------------------------

thermo          {thermo_every}

thermo_style    custom step temp press density vol etotal pe ke

# ============================================================
# Initial structure relaxation
# ============================================================

# The packed structure can contain severe intermolecular
# atomic overlaps. Do NOT assign 300 K velocities before
# removing these overlaps.

reset_timestep  0

# ------------------------------------------------------------
# Energy minimization
# ------------------------------------------------------------

print           "==============================================="
print           "Initial energy minimization"
print           "==============================================="

min_style       cg

minimize        1.0e-4 1.0e-6 10000 100000

# ------------------------------------------------------------
# Write minimized structure for inspection
# ------------------------------------------------------------

write_data      minimized.data

# ============================================================
# Low-temperature structural relaxation
# ============================================================

# Start from low temperature so that residual close contacts
# do not immediately produce very large atomic velocities.

velocity        all create 50.0 20260929 mom yes rot yes dist gaussian

fix             RELAX1 all nvt temp 50.0 50.0 100.0

run             50000

unfix           RELAX1

# ------------------------------------------------------------
# Gradual temperature increase
# ------------------------------------------------------------

fix             RELAX2 all nvt temp 50.0 {temperature:.3f} 100.0

run             100000

unfix           RELAX2

# ============================================================
# NVT equilibration at target temperature
# ============================================================

fix             EQ1 all nvt temp {temperature:.3f} {temperature:.3f} 100.0

run             {nvt_steps}

unfix           EQ1

# ============================================================
# NPT equilibration
# ============================================================

fix             EQ2 all npt temp {temperature:.3f} {temperature:.3f} 100.0 iso {pressure_atm:.6f} {pressure_atm:.6f} 1000.0

run             {npt_steps}

unfix           EQ2

# ------------------------------------------------------------
# Save equilibrated structure
# ------------------------------------------------------------

write_data      equilibrated.data

# ============================================================
# Green-Kubo viscosity production
# ============================================================

reset_timestep  0

compute         TALL all temp
compute         PALL all pressure TALL

variable        pxy equal c_PALL[4]
variable        pxz equal c_PALL[5]
variable        pyz equal c_PALL[6]

# Pressure autocorrelation.
#
# Pxy = c_PALL[4]
# Pxz = c_PALL[5]
# Pyz = c_PALL[6]
#
# Units conversion is performed in Python.

fix             CORR all ave/correlate {corr_every} {corr_points} {total_corr} c_PALL[4] c_PALL[5] c_PALL[6] type auto ave running file viscosity_corr.dat

# Green-Kubo production is NVE.
fix             GK all nve

variable        V equal vol
variable        T equal temp

fix             OUT all print {thermo_every} "$(step) $(v_V) $(v_T) $(v_pxy) $(v_pxz) $(v_pyz)" file pressure_timeseries.dat screen no title "step V T Pxy Pxz Pyz"

run             {gk_steps}

unfix           GK
unfix           CORR
unfix           OUT

write_data      final.data
"""
    )


def run_lammps(
    input_file: Path,
    workdir: Path,
    threads: int = 4,
):
    """
    LAMMPSをOpenMPで実行する。

    M1 4-core向け:
        OMP_NUM_THREADS=4
        -sf omp
        -pk omp 4
    """

    lmp = require_executable("lmp")

    env = os.environ.copy()

    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"] = "spread"
    env["OMP_PLACES"] = "cores"

    log = workdir / "lammps.log"

    with log.open("w") as fh:

        p = subprocess.run(
            [
                lmp,
                "-sf",
                "omp",
                "-pk",
                "omp",
                str(threads),
                "-in",
                input_file.name,
            ],
            cwd=workdir,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )

    if p.returncode != 0:
        raise RuntimeError(
            f"LAMMPS failed; inspect {log}"
        )


def require_executable(name: str):
    """
    PATH上の実行ファイルを探す。
    """

    import shutil

    path = shutil.which(name)

    if path is None:
        raise FileNotFoundError(
            f"Required executable '{name}' was not found in PATH."
        )

    return path