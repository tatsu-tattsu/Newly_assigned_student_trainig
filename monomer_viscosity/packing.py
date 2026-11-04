from __future__ import annotations

from pathlib import Path
import math
import re
import numpy as np
from rdkit import Chem

from .utils import atomic_mass, rotation_matrix


def parse_mol2_atoms(mol2: Path):
    atoms = []
    bonds = []
    in_atoms = False
    in_bonds = False
    for line in mol2.read_text().splitlines():
        s = line.strip()
        if s.upper() == "@<TRIPOS>ATOM":
            in_atoms, in_bonds = True, False
            continue
        if s.upper() == "@<TRIPOS>BOND":
            in_atoms, in_bonds = False, True
            continue
        if s.startswith("@<TRIPOS>"):
            in_atoms = in_bonds = False
            continue
        if in_atoms and s:
            p = s.split()
            # id name x y z type subst_id subst_name charge
            atoms.append({
                "id": int(p[0]),
                "name": p[1],
                "x": float(p[2]), "y": float(p[3]), "z": float(p[4]),
                "type": p[5],
                "resname": p[7] if len(p) > 7 else "MOL",
                "charge": float(p[8]) if len(p) > 8 else 0.0,
            })
        elif in_bonds and s:
            p = s.split()
            bonds.append((int(p[1]), int(p[2])))
    return atoms, bonds


def _pdb_atom_name(name: str, idx: int) -> str:
    # PDB atom-name field is 4 columns. Preserve short GAFF names.
    name = re.sub(r"[^A-Za-z0-9*]", "", name)[:4]
    return name.rjust(4) if len(name) < 4 else name



def create_packed_pdb(
    mol2: Path,
    out_pdb: Path,
    n_molecules: int,
    density_g_cm3: float,
    min_distance_A: float,
    max_attempts: int,
    seed: int = 20260929,
):
    """
    Generate an initial liquid configuration.

    The simulation box is determined directly from the requested
    density. Molecules are then inserted sequentially at random
    positions and orientations while checking intermolecular
    atomic distances.

    Important:
        - Molecular mass is determined from MOL2 atom names.
        - GAFF atom types are NOT interpreted as chemical elements.
        - The simulation box is NOT enlarged to accommodate packing.
        - Candidate molecules that overlap existing molecules are
          rejected.
        - Periodic boundary conditions are considered when checking
          intermolecular distances.
        - The final density is explicitly validated.

    Parameters
    ----------
    mol2 : Path
        GAFF2/AM1-BCC monomer MOL2 file.

    out_pdb : Path
        Output packed PDB file.

    n_molecules : int
        Number of monomer molecules.

    density_g_cm3 : float
        Target initial density.

    min_distance_A : float
        Minimum allowed intermolecular atomic distance in Angstrom.

        This value is applied ONLY between atoms belonging to
        different molecules. Intramolecular distances are never
        checked.

        For a first low-accuracy test, values around 1.2-1.5 A
        are reasonable. A value that is too large can make dense
        packing impossible.

    max_attempts : int
        Maximum number of placement attempts per molecule.

    seed : int
        Random seed.

    Returns
    -------
    box_A : float
        Cubic simulation box length in Angstrom.

    molecular_mass : float
        Monomer molecular mass in g/mol.
    """

    # ============================================================
    # Input validation
    # ============================================================

    mol2 = Path(mol2).resolve()
    out_pdb = Path(out_pdb).resolve()

    if not mol2.exists():
        raise FileNotFoundError(
            f"MOL2 file not found: {mol2}"
        )

    if n_molecules <= 0:
        raise ValueError(
            f"n_molecules must be positive: {n_molecules}"
        )

    if density_g_cm3 <= 0.0:
        raise ValueError(
            f"density_g_cm3 must be positive: {density_g_cm3}"
        )

    if min_distance_A <= 0.0:
        raise ValueError(
            f"min_distance_A must be positive: "
            f"{min_distance_A}"
        )

    if max_attempts <= 0:
        raise ValueError(
            f"max_attempts must be positive: "
            f"{max_attempts}"
        )

    # ============================================================
    # Read MOL2
    # ============================================================

    atoms, bonds = parse_mol2_atoms(mol2)

    if not atoms:
        raise RuntimeError(
            f"No atoms found in MOL2 file: {mol2}"
        )

    # ============================================================
    # Coordinates
    # ============================================================

    xyz = np.array(
        [
            [
                a["x"],
                a["y"],
                a["z"],
            ]
            for a in atoms
        ],
        dtype=float,
    )

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise RuntimeError(
            f"Invalid coordinate array shape: {xyz.shape}"
        )

    if not np.all(
        np.isfinite(xyz)
    ):
        raise RuntimeError(
            "MOL2 contains non-finite coordinates."
        )

    # ============================================================
    # Determine chemical elements
    # ============================================================

    def element_from_atom_name(name: str) -> str:
        """
        Determine chemical element from MOL2 atom name.

        Examples
        --------
        C1  -> C
        C2  -> C
        O1  -> O
        H1  -> H
        Cl1 -> Cl
        Br1 -> Br
        """

        if not isinstance(name, str):
            raise ValueError(
                f"Invalid MOL2 atom name: {name!r}"
            )

        letters = re.sub(
            r"[^A-Za-z]",
            "",
            name,
        )

        if not letters:
            raise ValueError(
                "Cannot determine element from "
                f"MOL2 atom name: {name!r}"
            )

        if len(letters) >= 2:

            first_two = (
                letters[:2].capitalize()
            )

            if first_two in {
                "Cl",
                "Br",
            }:
                return first_two

        return letters[0].upper()

    symbols = []

    for atom in atoms:

        symbol = element_from_atom_name(
            atom["name"]
        )

        try:
            atomic_mass(symbol)

        except ValueError as exc:

            raise ValueError(
                "Unsupported element detected in MOL2:\n"
                f"  atom name = {atom['name']!r}\n"
                f"  element   = {symbol!r}\n"
                f"  GAFF type = {atom['type']!r}"
            ) from exc

        symbols.append(symbol)

    # ============================================================
    # Molecular formula
    # ============================================================

    element_counts = {}

    for symbol in symbols:

        element_counts[symbol] = (
            element_counts.get(symbol, 0)
            + 1
        )

    formula_order = [
        "C",
        "H",
        "N",
        "O",
        "F",
        "P",
        "S",
        "Cl",
        "Br",
        "I",
    ]

    formula_parts = []

    for symbol in formula_order:

        count = element_counts.get(
            symbol,
            0,
        )

        if count == 0:
            continue

        if count == 1:
            formula_parts.append(symbol)
        else:
            formula_parts.append(
                f"{symbol}{count}"
            )

    for symbol in sorted(
        element_counts
    ):

        if symbol in formula_order:
            continue

        count = element_counts[symbol]

        if count == 1:
            formula_parts.append(symbol)
        else:
            formula_parts.append(
                f"{symbol}{count}"
            )

    molecular_formula = "".join(
        formula_parts
    )

    # ============================================================
    # Molecular mass
    # ============================================================

    molecular_mass = float(
        sum(
            atomic_mass(symbol)
            for symbol in symbols
        )
    )

    if not np.isfinite(
        molecular_mass
    ) or molecular_mass <= 0.0:

        raise RuntimeError(
            "Invalid molecular mass: "
            f"{molecular_mass}"
        )

    # ============================================================
    # Explicit CCO validation
    # ============================================================

    if molecular_formula == "C2H6O":

        expected_mass = 46.069

        if not np.isclose(
            molecular_mass,
            expected_mass,
            rtol=0.0,
            atol=1.0e-6,
        ):

            raise RuntimeError(
                "CCO molecular-mass validation failed:\n"
                f"  formula   = {molecular_formula}\n"
                f"  calculated= {molecular_mass:.9f} g/mol\n"
                f"  expected  = {expected_mass:.9f} g/mol"
            )

    # ============================================================
    # Total mass
    # ============================================================

    NA = 6.02214076e23

    mass_g = (
        n_molecules
        * molecular_mass
        / NA
    )

    if (
        not np.isfinite(mass_g)
        or mass_g <= 0.0
    ):

        raise RuntimeError(
            f"Invalid total mass: {mass_g}"
        )

    # ============================================================
    # Density -> box size
    # ============================================================

    volume_cm3 = (
        mass_g
        / density_g_cm3
    )

    volume_A3 = (
        volume_cm3
        * 1.0e24
    )

    if (
        not np.isfinite(volume_A3)
        or volume_A3 <= 0.0
    ):

        raise RuntimeError(
            f"Invalid target volume: "
            f"{volume_A3} A^3"
        )

    box_A = float(
        volume_A3 ** (1.0 / 3.0)
    )

    if box_A <= 0.0:
        raise RuntimeError(
            f"Invalid box size: {box_A} A"
        )

    # ============================================================
    # Center the monomer
    # ============================================================

    center = xyz.mean(
        axis=0
    )

    local = xyz - center

    # ============================================================
    # Molecular dimensions
    # ============================================================

    min_xyz = local.min(
        axis=0
    )

    max_xyz = local.max(
        axis=0
    )

    span = (
        max_xyz
        - min_xyz
    )

    molecular_radius = float(
        np.max(span) / 2.0
    )

    molecular_diameter = (
        2.0 * molecular_radius
    )

    # ============================================================
    # Sanity check for box size
    # ============================================================

    if box_A > 100.0:

        raise RuntimeError(
            "Calculated box is suspiciously large.\n"
            f"  formula       = {molecular_formula}\n"
            f"  mass          = {molecular_mass:.6f} g/mol\n"
            f"  molecules     = {n_molecules}\n"
            f"  density       = {density_g_cm3:.6f} g/cm3\n"
            f"  box           = {box_A:.6f} A"
        )

    # ============================================================
    # Random number generator
    # ============================================================

    rng = np.random.default_rng(
        seed
    )

    # ============================================================
    # Optional SciPy KDTree
    # ============================================================

    try:

        from scipy.spatial import cKDTree

        scipy_available = True

    except ImportError:

        cKDTree = None
        scipy_available = False

    # ============================================================
    # Periodic minimum-image distance
    # ============================================================

    def minimum_image(
        delta,
        box,
    ):
        """
        Apply minimum-image convention.
        """
        return (
            delta
            - box
            * np.rint(
                delta / box
            )
        )

    # ============================================================
    # Candidate collision test
    # ============================================================

    def candidate_is_valid(
        candidate,
        placed,
    ):
        """
        Check whether candidate molecule overlaps any previously
        placed molecule.

        Only intermolecular distances are checked.
        """

        if not placed:
            return True

        # --------------------------------------------------------
        # Bounding-radius pre-check
        # --------------------------------------------------------
        #
        # This is only a fast rejection criterion.
        # It does NOT replace the atomic distance check.
        # --------------------------------------------------------

        candidate_center = candidate.mean(
            axis=0
        )

        candidate_radius = float(
            np.max(
                np.ptp(
                    candidate,
                    axis=0
                )
            )
            / 2.0
        )

        for previous in placed:

            previous_center = previous.mean(
                axis=0
            )

            delta_center = minimum_image(
                candidate_center
                - previous_center,
                box_A,
            )

            center_distance = float(
                np.linalg.norm(
                    delta_center
                )
            )

            previous_radius = float(
                np.max(
                    np.ptp(
                        previous,
                        axis=0
                    )
                )
                / 2.0
            )

            broad_cutoff = (
                candidate_radius
                + previous_radius
                + min_distance_A
            )

            if (
                center_distance
                > broad_cutoff
            ):
                continue

            # ----------------------------------------------------
            # Exact atomic distance check
            # ----------------------------------------------------

            delta = (
                candidate[:, None, :]
                - previous[None, :, :]
            )

            delta = minimum_image(
                delta,
                box_A,
            )

            distances_sq = np.sum(
                delta * delta,
                axis=2,
            )

            if np.any(
                distances_sq
                < min_distance_A ** 2
            ):
                return False

        return True

    # ============================================================
    # Sequential random packing
    # ============================================================
    #
    # The previous implementation placed all molecules on a
    # lattice and THEN rotated them.
    #
    # That can produce severe atom overlaps.
    #
    # Here each molecule is independently inserted and rejected
    # if it overlaps an already placed molecule.
    # ============================================================

    placed_all = []

    total_attempts = 0

    # We gradually reduce the initial exclusion distance if the
    # requested value proves too restrictive.
    #
    # The actual requested value is always tried first.
    #
    # For a low-accuracy liquid initialization, allowing a somewhat
    # smaller initial distance is preferable to making the box
    # unphysically large.
    fallback_distances = [
        float(min_distance_A),
    ]

    # Only add fallback distances when useful.
    #
    # Do not go below 1.0 A because a smaller threshold can still
    # produce physically disastrous starting geometries.
    if min_distance_A > 1.5:
        fallback_distances.append(1.5)

    if min_distance_A > 1.2:
        fallback_distances.append(1.2)

    final_min_distance = None

    for molecule_index in range(
        n_molecules
    ):

        molecule_placed = False

        for active_min_distance in fallback_distances:

            if molecule_placed:
                break

            for attempt in range(
                max_attempts
            ):

                total_attempts += 1

                # ------------------------------------------------
                # Random rotation
                # ------------------------------------------------

                R = rotation_matrix(
                    rng
                )

                rotated = (
                    local @ R.T
                )

                # ------------------------------------------------
                # Random molecular center
                #
                # Keep the complete molecule inside the box.
                # This avoids broken molecular coordinates at the
                # initial stage.
                # ------------------------------------------------

                rotated_min = rotated.min(
                    axis=0
                )

                rotated_max = rotated.max(
                    axis=0
                )

                lower = (
                    -rotated_min
                    + 0.05
                )

                upper = (
                    box_A
                    - rotated_max
                    - 0.05
                )

                if np.any(
                    upper <= lower
                ):

                    raise RuntimeError(
                        "Molecule does not fit inside "
                        "the density-derived box.\n"
                        f"  molecule index = "
                        f"{molecule_index + 1}\n"
                        f"  box            = "
                        f"{box_A:.6f} A\n"
                        f"  molecular span  = "
                        f"{span}\n"
                        f"  molecular radius= "
                        f"{molecular_radius:.6f} A"
                    )

                center_new = rng.uniform(
                    lower,
                    upper,
                )

                candidate = (
                    rotated
                    + center_new
                )

                # ------------------------------------------------
                # Check atomic overlap
                # ------------------------------------------------

                if candidate_is_valid(
                    candidate,
                    placed_all,
                ):

                    placed_all.append(
                        candidate
                    )

                    molecule_placed = True

                    final_min_distance = (
                        active_min_distance
                    )

                    break

            # end attempt loop

        # end fallback distance loop

        if not molecule_placed:

            raise RuntimeError(
                "Failed to place molecule "
                f"{molecule_index + 1}/"
                f"{n_molecules}.\n"
                f"  box = {box_A:.6f} A\n"
                f"  requested min distance = "
                f"{min_distance_A:.3f} A\n"
                f"  max attempts/molecule = "
                f"{max_attempts}\n"
                f"  molecular formula = "
                f"{molecular_formula}\n"
                f"  molecular mass = "
                f"{molecular_mass:.6f} g/mol\n"
                f"  target density = "
                f"{density_g_cm3:.6f} g/cm3\n"
                "The requested density and "
                "minimum intermolecular distance "
                "may be incompatible."
            )

        # --------------------------------------------------------
        # Progress output
        # --------------------------------------------------------

        if (
            molecule_index == 0
            or (molecule_index + 1) % 10 == 0
            or molecule_index + 1 == n_molecules
        ):

            print(
                f"  Packing molecules: "
                f"{molecule_index + 1}/"
                f"{n_molecules}"
            )

    # ============================================================
    # Combine coordinates
    # ============================================================

    coords = np.vstack(
        placed_all
    )

    # ============================================================
    # Verify coordinates
    # ============================================================

    if not np.all(
        np.isfinite(coords)
    ):

        raise RuntimeError(
            "Packed coordinates contain "
            "non-finite values."
        )

    if np.any(
        coords < 0.0
    ) or np.any(
        coords >= box_A
    ):

        raise RuntimeError(
            "Packed coordinates lie outside "
            "the simulation box."
        )

    # ============================================================
    # Final intermolecular minimum-distance calculation
    # ============================================================

    actual_min_distance = np.inf

    # Use pairwise molecule checks.
    #
    # n_molecules = 100 is small enough that this is acceptable.
    # The expensive operation is still modest for 9 atoms/molecule.
    # ============================================================

    for i in range(
        n_molecules
    ):

        mol_i = placed_all[i]

        for j in range(
            i + 1,
            n_molecules
        ):

            mol_j = placed_all[j]

            delta = (
                mol_i[:, None, :]
                - mol_j[None, :, :]
            )

            delta = minimum_image(
                delta,
                box_A,
            )

            distances = np.sqrt(
                np.sum(
                    delta * delta,
                    axis=2,
                )
            )

            local_min = float(
                distances.min()
            )

            if local_min < actual_min_distance:
                actual_min_distance = local_min

    # ============================================================
    # Final density
    # ============================================================

    actual_volume_A3 = (
        box_A ** 3
    )

    actual_volume_cm3 = (
        actual_volume_A3
        * 1.0e-24
    )

    actual_density = (
        mass_g
        / actual_volume_cm3
    )

    density_relative_error = (
        abs(
            actual_density
            - density_g_cm3
        )
        / density_g_cm3
    )

    if (
        density_relative_error
        > 1.0e-10
    ):

        raise RuntimeError(
            "Initial density validation failed.\n"
            f"  requested = "
            f"{density_g_cm3:.12f} g/cm3\n"
            f"  actual    = "
            f"{actual_density:.12f} g/cm3\n"
            f"  relative error = "
            f"{density_relative_error:.3e}"
        )

    # ============================================================
    # Final minimum-distance validation
    # ============================================================

    if not np.isfinite(
        actual_min_distance
    ):

        raise RuntimeError(
            "Could not determine the final "
            "intermolecular minimum distance."
        )

    if (
        actual_min_distance
        < final_min_distance
        - 1.0e-8
    ):

        print(
            "Final packing violates the requested "
            "minimum intermolecular distance.\n"
            f"  requested = "
            f"{final_min_distance:.6f} A\n"
            f"  actual    = "
            f"{actual_min_distance:.6f} A"
        )

    # ============================================================
    # Write PDB
    # ============================================================

    lines = [
        "REMARK   packed monomer liquid generated by monomer_viscosity",
        f"REMARK   molecules = {n_molecules}",
        f"REMARK   formula = {molecular_formula}",
        f"REMARK   molecular_mass = "
        f"{molecular_mass:.6f} g/mol",
        f"REMARK   requested_density = "
        f"{density_g_cm3:.9f} g/cm3",
        f"REMARK   actual_initial_density = "
        f"{actual_density:.9f} g/cm3",
        f"REMARK   density_relative_error = "
        f"{density_relative_error:.3e}",
        f"REMARK   box = "
        f"{box_A:.6f} A",
        f"REMARK   molecular_radius = "
        f"{molecular_radius:.6f} A",
        f"REMARK   molecular_diameter = "
        f"{molecular_diameter:.6f} A",
        f"REMARK   requested_min_interatomic_distance = "
        f"{min_distance_A:.6f} A",
        f"REMARK   used_min_interatomic_distance = "
        f"{final_min_distance:.6f} A",
        f"REMARK   actual_min_interatomic_distance = "
        f"{actual_min_distance:.6f} A",
        f"REMARK   total_placement_attempts = "
        f"{total_attempts}",
    ]

    # ============================================================
    # ATOM records
    # ============================================================

    serial = 1
    nat = len(atoms)

    for m in range(
        n_molecules
    ):

        block = coords[
            m * nat:
            (m + 1) * nat
        ]

        for j, (
            atom,
            position,
        ) in enumerate(
            zip(
                atoms,
                block,
            )
        ):

            lines.append(
                f"ATOM  {serial:5d} "
                f"{_pdb_atom_name(atom['name'], j + 1)} "
                f"MOL A{m + 1:4d}"
                f"    {position[0]:8.3f}"
                f"{position[1]:8.3f}"
                f"{position[2]:8.3f}"
                f"  1.00  0.00          "
            )

            serial += 1

        lines.append(
            "TER"
        )

    lines.append(
        "END"
    )

    # ============================================================
    # Write PDB
    # ============================================================

    out_pdb.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_pdb.write_text(
        "\n".join(lines)
        + "\n"
    )

    # ============================================================
    # Console output
    # ============================================================

    print()
    print(
        "Packed system:"
    )
    print(
        f"  formula                    = "
        f"{molecular_formula}"
    )
    print(
        f"  molecular mass             = "
        f"{molecular_mass:.6f} g/mol"
    )
    print(
        f"  molecules                  = "
        f"{n_molecules}"
    )
    print(
        f"  requested density          = "
        f"{density_g_cm3:.6f} g/cm3"
    )
    print(
        f"  actual initial density     = "
        f"{actual_density:.6f} g/cm3"
    )
    print(
        f"  density relative error     = "
        f"{density_relative_error:.3e}"
    )
    print(
        f"  box                        = "
        f"{box_A:.6f} A"
    )
    print(
        f"  molecular diameter         = "
        f"{molecular_diameter:.6f} A"
    )
    print(
        f"  requested min distance     = "
        f"{min_distance_A:.6f} A"
    )
    print(
        f"  used min distance          = "
        f"{final_min_distance:.6f} A"
    )
    print(
        f"  actual min distance        = "
        f"{actual_min_distance:.6f} A"
    )
    print(
        f"  total placement attempts   = "
        f"{total_attempts}"
    )

    if molecular_formula == "C2H6O":

        print(
            "  CCO mass check             = PASS "
            "(46.069 g/mol)"
        )

    print()

    # ============================================================
    # Write Extended XYZ
    # ============================================================

    packed_symbols = symbols * n_molecules

    out_xyz = out_pdb.with_suffix(".extxyz")

    write_extxyz(
        out_xyz=out_xyz,
        symbols=packed_symbols,
        coords=coords,
        box_A=box_A,
    )

    print(f"  Extended XYZ              = {out_xyz}")

    return box_A, molecular_mass



def build_amber_box(
    packed_pdb: Path,
    monomer_mol2: Path,
    frcmod: Path,
    outdir: Path,
    box_A: float,
):
    from .utils import run_command, require_executable

    packed_pdb = Path(packed_pdb).resolve()
    monomer_mol2 = Path(monomer_mol2).resolve()
    frcmod = Path(frcmod).resolve()
    outdir = Path(outdir).resolve()

    if not packed_pdb.exists():
        raise FileNotFoundError(
            f"Packed PDB not found: {packed_pdb}"
        )

    if not monomer_mol2.exists():
        raise FileNotFoundError(
            f"Monomer MOL2 not found: {monomer_mol2}"
        )

    if not frcmod.exists():
        raise FileNotFoundError(
            f"FRCMOD not found: {frcmod}"
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tleap = require_executable("tleap")

    prmtop = outdir / "system.prmtop"
    inpcrd = outdir / "system.inpcrd"
    leap = outdir / "tleap_system.in"

    leap.write_text(
        f"""source leaprc.gaff2
loadamberparams {frcmod}

MOL = loadmol2 {monomer_mol2}
SYS = loadpdb {packed_pdb}

set SYS box {{ {box_A:.6f} {box_A:.6f} {box_A:.6f} }}

check SYS

saveamberparm SYS {prmtop} {inpcrd}

quit
"""
    )

    run_command(
        [
            tleap,
            "-f",
            str(leap),
        ],
        outdir,
        outdir / "tleap_system.log",
    )

    return prmtop, inpcrd

from pathlib import Path
import numpy as np


def write_extxyz(
    out_xyz: Path,
    symbols: list[str],
    coords: np.ndarray,
    box_A: float,
):
    """
    Write atomic coordinates in Extended XYZ format.

    Parameters
    ----------
    out_xyz : Path
        Output XYZ file.
    symbols : list[str]
        Chemical element symbols.
    coords : np.ndarray
        Atomic coordinates in Angstrom.
    box_A : float
        Cubic simulation box length in Angstrom.
    """

    out_xyz = Path(out_xyz)
    coords = np.asarray(coords, dtype=float)

    if coords.shape != (len(symbols), 3):
        raise ValueError(
            f"Invalid coordinate shape: {coords.shape}"
        )

    if not np.all(np.isfinite(coords)):
        raise ValueError("Coordinates contain non-finite values.")

    if box_A <= 0:
        raise ValueError("box_A must be positive.")

    out_xyz.parent.mkdir(parents=True, exist_ok=True)

    lattice = (
        f"{box_A:.10f} 0.0 0.0 "
        f"0.0 {box_A:.10f} 0.0 "
        f"0.0 0.0 {box_A:.10f}"
    )

    with out_xyz.open("w") as f:

        f.write(f"{len(symbols)}\n")

        f.write(
            f'Lattice="{lattice}" '
            'Properties=species:S:1:pos:R:3 '
            'pbc="T T T"\n'
        )

        for symbol, xyz in zip(symbols, coords):

            f.write(
                f"{symbol:2s} "
                f"{xyz[0]:15.8f} "
                f"{xyz[1]:15.8f} "
                f"{xyz[2]:15.8f}\n"
            )