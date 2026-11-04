from __future__ import annotations

from pathlib import Path

from .utils import run_command, require_executable


from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from monomer_viscosity.visualization import read_mol2_atoms, read_mol2_bonds
from monomer_viscosity.visualization import export_esp_cube

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.colors import Normalize


ELEMENT_COLORS = {
    "C": "#707070",
    "H": "#FFFFFF",
    "N": "#3050F8",
    "O": "#FF2020",
    "F": "#90E050",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "Br": "#A62929",
    "I": "#940094",
    "Si": "#D0A0A0",
}


def export_esp_png(
    mol2_path,
    output_path,
    grid_spacing=0.25,
    surface_alpha=0.85,
    show_charges=False,
):
    """
    AM1-BCC部分電荷に基づくESP表面を生成する。

    分子表面: ESPによる青-白-赤の着色
    原子: 元素ごとの色分け
    結合: MOL2の結合情報に基づいて描画
    """

    atoms = read_mol2_atoms(mol2_path)
    bonds = read_mol2_bonds(mol2_path)

    coords = np.array([
        [a["x"], a["y"], a["z"]]
        for a in atoms
    ])

    charges = np.array([a["charge"] for a in atoms])
    radii = np.array([a["radius"] for a in atoms])

    # --------------------------------------------------
    # 1. 分子全体のvan der Waals表面を生成
    # --------------------------------------------------

    padding = 3.0

    lower = coords.min(axis=0) - radii.max() - padding
    upper = coords.max(axis=0) + radii.max() + padding

    axes = [
        np.arange(lower[i], upper[i] + grid_spacing, grid_spacing)
        for i in range(3)
    ]

    X, Y, Z = np.meshgrid(*axes, indexing="ij")

    field = np.full(X.shape, np.inf, dtype=np.float32)

    for coord, radius in zip(coords, radii):

        distance = np.sqrt(
            (X - coord[0]) ** 2
            + (Y - coord[1]) ** 2
            + (Z - coord[2]) ** 2
        )

        field = np.minimum(
            field,
            distance - radius
        )

    vertices, faces, normals, values = marching_cubes(
        field,
        level=0.0,
        spacing=(grid_spacing,) * 3,
    )

    vertices += lower

    # --------------------------------------------------
    # 2. 表面上のESPを計算
    # --------------------------------------------------

    esp = np.zeros(len(vertices))

    # メモリ消費を抑えるため分割計算
    chunk_size = 5000

    for start in range(0, len(vertices), chunk_size):

        end = min(start + chunk_size, len(vertices))

        delta = (
            vertices[start:end, None, :]
            - coords[None, :, :]
        )

        distances = np.linalg.norm(delta, axis=2)
        distances = np.maximum(distances, 0.25)

        # ESP [Hartree/e]
        esp[start:end] = (
            0.529177210903
            * np.sum(charges[None, :] / distances, axis=1)
        )

    # 表面の極端な値によって色のコントラストが
    # 失われないように、絶対値の95パーセンタイルを使用
    color_limit = max(
        np.percentile(np.abs(esp), 95),
        0.01
    )

    norm = Normalize(
        vmin=-color_limit,
        vmax=color_limit
    )

    cmap = plt.get_cmap("coolwarm")

    face_esp = esp[faces].mean(axis=1)
    face_colors = cmap(norm(face_esp))

    # --------------------------------------------------
    # 3. 描画
    # --------------------------------------------------

    fig = plt.figure(figsize=(11, 9), facecolor="white")

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # 分子表面
    surface = Poly3DCollection(
        vertices[faces],
        facecolors=face_colors,
        edgecolors="none",
        alpha=surface_alpha,
        linewidths=0,
    )

    ax.add_collection3d(surface)

    # --------------------------------------------------
    # 4. 結合を描画
    # --------------------------------------------------

    atom_by_id = {
        a["id"]: a
        for a in atoms
    }

    for atom_id1, atom_id2 in bonds:

        a1 = atom_by_id[atom_id1]
        a2 = atom_by_id[atom_id2]

        p1 = [a1["x"], a1["y"], a1["z"]]
        p2 = [a2["x"], a2["y"], a2["z"]]

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color="#303030",
            linewidth=2.0,
            alpha=0.9,
        )

    # --------------------------------------------------
    # 5. 元素ごとに原子を色分け
    # --------------------------------------------------

    for element, color in ELEMENT_COLORS.items():

        selected = [
            a for a in atoms
            if a["element"] == element
        ]

        if not selected:
            continue

        xyz = np.array([
            [a["x"], a["y"], a["z"]]
            for a in selected
        ])

        # 原子の大きさを元素によらず一定にして
        # 元素ごとの色を判別しやすくする
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=85,
            c=color,
            edgecolors="#202020",
            linewidths=0.6,
            depthshade=False,
            label=element,
        )

    # 部分電荷の数値表示（必要な場合のみ）
    if show_charges:
        for atom in atoms:
            ax.text(
                atom["x"],
                atom["y"],
                atom["z"],
                f'{atom["charge"]:+.2f}',
                fontsize=7,
                color="black",
            )

    # --------------------------------------------------
    # 6. カラーバー・レイアウト
    # --------------------------------------------------

    mappable = cm.ScalarMappable(
        norm=norm,
        cmap=cmap
    )

    mappable.set_array([])

    cbar = fig.colorbar(
        mappable,
        ax=ax,
        shrink=0.65,
        pad=0.08,
    )

    cbar.set_label(
        "Electrostatic potential (Hartree/e)",
        fontsize=11,
    )

    ax.legend(
        title="Elements",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=9,
    )

    ax.set_box_aspect((
        np.ptp(coords[:, 0]) + 2,
        np.ptp(coords[:, 1]) + 2,
        np.ptp(coords[:, 2]) + 2,
    ))

    ax.view_init(elev=25, azim=35)
    ax.set_axis_off()

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def make_gaff2_parameters(
    optimized_sdf: Path,
    workdir: Path,
    net_charge: int,
    forcefield: str = "gaff2",
    charge_method: str = "bcc",
):
    # run_command() changes the working directory to workdir.
    # Therefore, all paths passed to external AmberTools programs
    # should be converted to absolute paths first.
    
    optimized_sdf = Path(optimized_sdf).resolve()
    workdir = Path(workdir).resolve()

    if not optimized_sdf.exists():
        raise FileNotFoundError(
            f"Optimized SDF not found: {optimized_sdf}"
        )

    workdir.mkdir(parents=True, exist_ok=True)

    antechamber = require_executable("antechamber")
    parmchk2 = require_executable("parmchk2")
    tleap = require_executable("tleap")

    mol2 = workdir / "monomer.mol2"
    frcmod = workdir / "monomer.frcmod"
    prepi = workdir / "monomer.prepi"
    lib = workdir / "monomer.lib"
    prmtop = workdir / "monomer.prmtop"
    inpcrd = workdir / "monomer.inpcrd"

    # AM1-BCC is the inexpensive automatic charge model.
    # Psi4 has already supplied the optimized geometry.
    run_command(
        [
            antechamber,
            "-i", str(optimized_sdf),
            "-fi", "sdf",
            "-o", str(mol2),
            "-fo", "mol2",
            "-at", forcefield,
            "-c", charge_method,
            "-nc", str(net_charge),
            "-rn", "MOL",
            "-pf", "y",
        ],
        workdir,
        workdir / "antechamber.log",
    )

    run_command(
        [
            parmchk2,
            "-i", str(mol2),
            "-f", "mol2",
            "-o", str(frcmod),
            "-s", "2",
        ],
        workdir,
        workdir / "parmchk2.log",
    )

    leap = workdir / "tleap_monomer.in"

    leap.write_text(
        f"""source leaprc.{forcefield}
loadamberparams {frcmod}
MOL = loadmol2 {mol2}
check MOL
saveamberparm MOL {prmtop} {inpcrd}
saveoff MOL {lib}
quit
"""
    )

    run_command(
        [tleap, "-f", str(leap)],
        workdir,
        workdir / "tleap.log",
    )

    export_esp_cube(
        mol2_path=mol2,
        output_path=workdir / "visualization" / "esp.cube",
        grid_spacing=0.25,
        padding=4.0,
    )

    return mol2, frcmod, prmtop, inpcrd