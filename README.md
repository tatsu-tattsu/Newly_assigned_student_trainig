# Monomer viscosity predictor

SMILES only -> RDKit 3D -> Psi4 optimization -> AmberTools/GAFF2 -> random liquid packing -> LAMMPS/OpenMP -> Green-Kubo viscosity.

This is a screening/test implementation. The default system is 100 monomers and about 1.5 ns MD (0.2 ns NVT + 0.5 ns NPT + 0.8 ns Green-Kubo production).

## New environment

```bash
conda env create -f environment.yml
conda activate monomer-viscosity
```

Check:

```bash
python -c "import rdkit, psi4, parmed; print('python stack OK')"
lmp -h | head
antechamber -h | head
parmchk2 -h | head
tleap -h | head
```

Check OpenMP:

```bash
lmp -h | grep -i omp
```

The runner uses one MPI task with 4 OpenMP threads by default:

```bash
OMP_NUM_THREADS=4 lmp -sf omp -pk omp 4 -in in.lammps
```

## Run

```bash
python run.py "CCO"
```

The only required scientific input is the SMILES. All test parameters are in `config.py`.

For a fast smoke test:

```bash
python run.py "CCO" --smoke
```

For the default test:

```bash
python run.py "CCO"
```

Results are written below `runs/<job-id>/`.

## Important scientific limitation

The default charge model is AmberTools AM1-BCC, not RESP. Psi4 is used for quantum-mechanical geometry optimization. This is deliberate for a fully automatic, relatively inexpensive test workflow. The result should be treated as a screening/test value, not a publication-quality viscosity without convergence tests, multiple independent trajectories, finite-size assessment, and force-field validation.

