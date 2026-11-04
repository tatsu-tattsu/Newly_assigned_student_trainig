from __future__ import annotations

from pathlib import Path
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import AllChem, Draw


def mol_from_smiles(smiles: str, mol_dir):
    mol = Chem.MolFromSmiles(smiles)
    result_dir = mol_dir / "result"
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(
        mol,
        randomSeed=20260929,
        useRandomCoords=False,
        enforceChirality=True,
    )
    if status != 0:
        status = AllChem.EmbedMolecule(
            mol, randomSeed=20260929, useRandomCoords=True
        )
    if status != 0:
        raise RuntimeError("RDKit 3-D embedding failed.")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass
    os.makedirs(result_dir, exist_ok=True)
    Draw.MolToImage(mol).save(os.path.join(result_dir, "rdkit_3d_mol.png"))
    return mol


def rdkit_xyz(mol) -> str:
    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), "RDKit 3D structure"]
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()} {p.x:.10f} {p.y:.10f} {p.z:.10f}")
    return "\n".join(lines) + "\n"


def psi4_optimize(mol, workdir: Path, method: str, basis: str, memory_gb: float, threads: int):
    import psi4

    workdir.mkdir(parents=True, exist_ok=True)
    result_dir = workdir / "result"
    psi4.core.set_num_threads(threads)
    psi4.set_memory(f"{memory_gb} GB")
    psi4.core.set_output_file(str(workdir / "psi4.out"), False)

    xyz = rdkit_xyz(mol)
    xyz_path = result_dir / "start.xyz"
    xyz_path.write_text(rdkit_xyz(mol), encoding="utf-8")
    q = Chem.GetFormalCharge(Chem.RemoveHs(mol))
    xyz_lines = xyz.splitlines()[2:]
    psi_mol = psi4.geometry(
        f"{q} 1\n" + "\n".join(xyz_lines) + "\nunits angstrom"
    )
    psi_mol.update_geometry()

    psi4.set_options({
        "basis": basis,
        "scf_type": "df",
        "geom_maxiter": 80,
        "g_convergence": "gau_tight",
    })
    energy, wfn = psi4.optimize(
        f"{method}/{basis}",
        molecule=psi_mol,
        return_wfn=True,
    )
    # Psi4 molecule coordinates are in bohr after internal processing.
    geom_bohr = np.asarray(psi_mol.geometry().np)
    geom_ang = geom_bohr * 0.529177210903

    out = mol.GetConformer()
    for i, xyz_i in enumerate(geom_ang):
        out.SetAtomPosition(i, xyz_i.tolist())

    sdf = workdir / "optimized.sdf"
    writer = Chem.SDWriter(str(sdf))
    writer.write(mol)
    writer.close()

    (result_dir / "optimized.xyz").write_text(
        Chem.MolToXYZBlock(mol)
    )
    return float(energy), sdf


def formal_charge(smiles: str) -> int:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError("Invalid SMILES")
    return int(Chem.GetFormalCharge(m))
