from pathlib import Path


def read_mol2_atoms(mol2_path):
    """
    MOL2ファイルから原子座標、GAFF2 atom type、部分電荷を取得する。
    """

    atoms = []
    in_atom_section = False

    with open(mol2_path, "r", encoding="utf-8") as f:
        for line in f:

            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_section = True
                continue

            if line.startswith("@<TRIPOS>"):
                if in_atom_section:
                    break

            if not in_atom_section:
                continue

            fields = line.split()

            if len(fields) < 9:
                continue

            atom = {
                "id": int(fields[0]),
                "name": fields[1],
                "x": float(fields[2]),
                "y": float(fields[3]),
                "z": float(fields[4]),
                "atom_type": fields[5],
                "charge": float(fields[8]),
            }

            atoms.append(atom)

    if not atoms:
        raise ValueError(f"No atoms found in MOL2 file: {mol2_path}")

    return add_vdw_radii(atoms)

def read_mol2_bonds(mol2_path):
    bonds = []
    in_bond_section = False

    with open(mol2_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("@<TRIPOS>BOND"):
                in_bond_section = True
                continue

            if line.startswith("@<TRIPOS>"):
                if in_bond_section:
                    break

            if not in_bond_section:
                continue

            fields = line.split()
            if len(fields) < 4:
                continue

            bonds.append((int(fields[1]), int(fields[2])))

    return bonds

# Bondi van der Waals radii [Å]
VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
    "Si": 2.10,
}


def element_from_gaff_type(atom_type):
    """
    GAFF2 atom typeから元素記号を判定する。
    """

    atom_type = atom_type.lower()

    if atom_type.startswith("cl"):
        return "Cl"
    elif atom_type.startswith("br"):
        return "Br"
    elif atom_type.startswith("si"):
        return "Si"

    mapping = {
        "c": "C",
        "n": "N",
        "o": "O",
        "s": "S",
        "p": "P",
        "f": "F",
        "i": "I",
        "h": "H",
    }

    element = mapping.get(atom_type[0])

    if element is None:
        raise ValueError(
            f"Unsupported GAFF2 atom type: {atom_type}"
        )

    return element


def add_vdw_radii(atoms):
    """
    各原子に元素情報とvan der Waals半径を追加する。
    """

    for atom in atoms:
        element = element_from_gaff_type(atom["atom_type"])

        atom["element"] = element
        atom["radius"] = VDW_RADII[element]

    return atoms


from pathlib import Path

import numpy as np


# Bohr radius [Å]
ANGSTROM_PER_BOHR = 0.529177210903

ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
    "Si": 14,
}


def export_esp_cube(
    mol2_path,
    output_path,
    grid_spacing=0.25,
    padding=4.0,
    softening=0.20,
):
    """
    MOL2のAM1-BCC部分電荷からESPを計算し、
    Gaussian互換のCubeファイルとして出力する。

    Parameters
    ----------
    mol2_path : str or Path
        GAFF2/AM1-BCCパラメータを含むMOL2ファイル
    output_path : str or Path
        出力Cubeファイル
    grid_spacing : float
        グリッド間隔 [Å]
    padding : float
        分子外側に設ける余白 [Å]
    softening : float
        原子位置での発散を避けるための距離下限 [Å]

    Notes
    -----
    ESPの単位はHartree/e (atomic units)。
    点電荷モデルによる近似ESPであり、電子密度ではない。
    """

    atoms = read_mol2_atoms(mol2_path)

    coords = np.array([
        [a["x"], a["y"], a["z"]]
        for a in atoms
    ], dtype=float)

    charges = np.array([
        a["charge"] for a in atoms
    ], dtype=float)

    elements = [
        a["element"] for a in atoms
    ]

    atomic_numbers = np.array([
        ATOMIC_NUMBERS[e] for e in elements
    ], dtype=int)

    # --------------------------------------------------
    # 1. Grid definition [Å]
    # --------------------------------------------------

    lower = coords.min(axis=0) - padding
    upper = coords.max(axis=0) + padding

    axes = [
        np.arange(
            lower[i],
            upper[i] + grid_spacing * 0.5,
            grid_spacing,
        )
        for i in range(3)
    ]

    nx, ny, nz = map(len, axes)

    # Cube format uses Bohr units for coordinates
    origin_bohr = lower / ANGSTROM_PER_BOHR
    spacing_bohr = grid_spacing / ANGSTROM_PER_BOHR
    coords_bohr = coords / ANGSTROM_PER_BOHR

    # --------------------------------------------------
    # 2. Cube header
    # --------------------------------------------------

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as f:

        f.write("AM1-BCC point-charge electrostatic potential\n")
        f.write("ESP generated from GAFF2 MOL2 partial charges\n")

        f.write(
            f"{len(atoms):5d}"
            f"{origin_bohr[0]:12.6f}"
            f"{origin_bohr[1]:12.6f}"
            f"{origin_bohr[2]:12.6f}\n"
        )

        f.write(
            f"{nx:5d}"
            f"{spacing_bohr:12.6f}"
            f"{0.0:12.6f}"
            f"{0.0:12.6f}\n"
        )

        f.write(
            f"{ny:5d}"
            f"{0.0:12.6f}"
            f"{spacing_bohr:12.6f}"
            f"{0.0:12.6f}\n"
        )

        f.write(
            f"{nz:5d}"
            f"{0.0:12.6f}"
            f"{0.0:12.6f}"
            f"{spacing_bohr:12.6f}\n"
        )

        # Atom records: atomic number, nuclear charge, x, y, z
        for z, xyz in zip(atomic_numbers, coords_bohr):
            f.write(
                f"{z:5d}"
                f"{float(z):12.6f}"
                f"{xyz[0]:12.6f}"
                f"{xyz[1]:12.6f}"
                f"{xyz[2]:12.6f}\n"
            )

        # --------------------------------------------------
        # 3. ESP calculation and Cube data output
        # --------------------------------------------------

        # x -> y -> z の順にグリッドを走査
        # メモリ消費を抑えるため、xスライス単位で計算

        values_per_line = 6
        buffer = []

        for ix, x in enumerate(axes[0]):

            Y, Z = np.meshgrid(
                axes[1],
                axes[2],
                indexing="ij",
            )

            points = np.column_stack([
                np.full(Y.size, x),
                Y.ravel(),
                Z.ravel(),
            ])

            delta = (
                points[:, None, :]
                - coords[None, :, :]
            )

            distances = np.linalg.norm(delta, axis=2)
            distances = np.maximum(
                distances,
                softening,
            )

            # ESP [Hartree/e]
            esp = (
                ANGSTROM_PER_BOHR
                * np.sum(
                    charges[None, :] / distances,
                    axis=1,
                )
            )

            # Cube data
            for value in esp:
                buffer.append(f"{value:13.5e}")

                if len(buffer) == values_per_line:
                    f.write(" ".join(buffer) + "\n")
                    buffer.clear()

        if buffer:
            f.write(" ".join(buffer) + "\n")

    print(f"ESP Cube saved: {output_path}")
    print(f"Grid dimensions: {nx} x {ny} x {nz}")
    print(f"Grid spacing: {grid_spacing:.3f} Å")
    print("ESP unit: Hartree/e")
