from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from monomer_viscosity.config import Config
from monomer_viscosity.molecule import mol_from_smiles, psi4_optimize, formal_charge
from monomer_viscosity.forcefield import make_gaff2_parameters
from monomer_viscosity.packing import create_packed_pdb, build_amber_box
from monomer_viscosity.lammps import amber_to_lammps, write_lammps_input, run_lammps
from monomer_viscosity.viscosity import green_kubo
from monomer_viscosity.utils import write_json


def main():
    ap = argparse.ArgumentParser(description="SMILES-only monomer viscosity predictor")
    ap.add_argument("smiles")
    ap.add_argument("--smoke", action="store_true",
                    help="Very short pipeline test; not a meaningful viscosity.")
    ap.add_argument("--nmol", type=int, default=None,
                    help="Override molecule count for testing.")
    args = ap.parse_args()

    cfg = Config()
    if args.nmol:
        cfg = Config(n_molecules=args.nmol)
    if args.smoke:
        cfg = Config(
            n_molecules=20,
            nvt_steps=10_000,
            npt_steps=20_000,
            gk_steps=50_000,
            thermo_every=1000,
            corr_every=10,
            corr_points=1000,
        )

    job_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.smiles)[:50] + "_viscosity"
    root = cfg.job_dir(job_id)
    mol_dir = root / "01_molecule"
    ff_dir = root / "02_forcefield"
    pack_dir = root / "03_packing"
    md_dir = root / "04_md"
    result_dir = root / "05_result"
    for d in [mol_dir, ff_dir, pack_dir, md_dir, result_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Monomer viscosity predictor")
    print("=" * 70)
    print("SMILES:", args.smiles)
    print("Molecules:", cfg.n_molecules)
    print("Target temperature:", cfg.temperature_K, "K")
    print("MD:", cfg.total_ns, "ns")
    print()

    # 1. RDKit
    mol = mol_from_smiles(args.smiles, mol_dir)
    charge = formal_charge(args.smiles)
    (mol_dir / "input.smi").write_text(args.smiles + "\n")
    from rdkit import Chem
    Chem.MolToMolFile(mol, str(mol_dir / "rdkit_3d.mol"))
    from rdkit.Chem import Draw
    import os
    Draw.MolToImage(mol).save(os.path.join(mol_dir, "rdkit_3d_mol.png"))

    # 2. Psi4
    print("[1/5] Psi4 geometry optimization")
    energy, optimized_sdf = psi4_optimize(
        mol, mol_dir, cfg.psi4_method, cfg.psi4_basis,
        cfg.psi4_memory_gb, cfg.psi4_threads
    )
    write_json(mol_dir / "psi4_result.json", {
        "energy_hartree": energy,
        "method": cfg.psi4_method,
        "basis": cfg.psi4_basis,
    })

    # 3. GAFF2/AmberTools
    print("[2/5] GAFF2 parameterization")
    mol2, frcmod, mono_prmtop, mono_inpcrd = make_gaff2_parameters(
        optimized_sdf, ff_dir, charge, cfg.forcefield, cfg.charge_method
    )

    # 4. Pack
    print("[3/5] Liquid packing")
    packed_pdb = pack_dir / "packed.pdb"
    box_A, molecular_mass = create_packed_pdb(
        mol2, packed_pdb, cfg.n_molecules,
        cfg.initial_density_g_cm3,
        cfg.min_interatomic_distance_A,
        cfg.max_pack_attempts,
    )
    sys_prmtop, sys_inpcrd = build_amber_box(
        packed_pdb, mol2, frcmod, pack_dir, box_A
    )

    # 5. LAMMPS
    print("[4/5] Amber -> LAMMPS")
    data_file, parm = amber_to_lammps(sys_prmtop, sys_inpcrd, md_dir)
    in_file = md_dir / "in.lammps"
    write_lammps_input(
        data_file, in_file,
        cfg.temperature_K, cfg.pressure_atm, cfg.timestep_fs,
        cfg.nvt_steps, cfg.npt_steps, cfg.gk_steps,
        cfg.thermo_every, cfg.corr_every, cfg.corr_points,
        cfg.neighbor_skin_A, cfg.lj_cutoff_A, cfg.pppm_accuracy,
    )

    print("[5/5] LAMMPS/OpenMP MD")
    run_lammps(in_file, md_dir, cfg.omp_threads)

    corr = md_dir / "viscosity_corr.dat"
    pressure_ts = md_dir / "pressure_timeseries.dat"
    volume = None
    if pressure_ts.exists():
        from monomer_viscosity.viscosity import _read_volume
        volume = _read_volume(pressure_ts)
    else:
        # ParmEd box fallback
        volume = box_A ** 3

    result = green_kubo(
        corr, volume, cfg.temperature_K,
        cfg.corr_every * cfg.timestep_fs / 1000.0,
    )
    result.update({
        "smiles": args.smiles,
        "n_molecules": cfg.n_molecules,
        "molecular_mass_g_mol": molecular_mass,
        "initial_box_A": box_A,
        "psi4_energy_hartree": energy,
        "psi4_method": cfg.psi4_method,
        "psi4_basis": cfg.psi4_basis,
        "charge_method": cfg.charge_method,
        "forcefield": cfg.forcefield,
        "smoke": args.smoke,
    })
    write_json(result_dir / "result.json", result)

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Viscosity = {result['eta_avg_mPa_s']:.6g} mPa s")
    print(f"xy         = {result['eta_xy_mPa_s']:.6g} mPa s")
    print(f"xz         = {result['eta_xz_mPa_s']:.6g} mPa s")
    print(f"yz         = {result['eta_yz_mPa_s']:.6g} mPa s")
    print("Result:", result_dir / "result.json")


if __name__ == "__main__":
    main()
