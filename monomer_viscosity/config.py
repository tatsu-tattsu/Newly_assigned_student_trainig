from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    n_molecules: int = 100
    temperature_K: float = 300.0
    pressure_atm: float = 1.0
    timestep_fs: float = 1.0

    # MD: 0.2 ns NVT + 0.5 ns NPT + 0.8 ns GK
    nvt_steps: int = 200_000
    npt_steps: int = 500_000
    gk_steps: int = 800_000

    thermo_every: int = 5000
    dump_every: int = 0

    # GK correlation sampling
    corr_every: int = 10
    corr_points: int = 500

    # Initial packing density; NPT determines the final density.
    initial_density_g_cm3: float = 0.80
    min_interatomic_distance_A: float = 1.55
    max_pack_attempts: int = 200_000

    # QM
    psi4_method: str = "wb97m-d3bj"
    psi4_basis: str = "def2-svp"
    psi4_memory_gb: float = 1.0
    psi4_threads: int = 4

    # Force field
    charge_method: str = "bcc"
    forcefield: str = "gaff2"
    net_charge: int = 0

    # LAMMPS
    omp_threads: int = 4
    neighbor_skin_A: float = 2.0
    lj_cutoff_A: float = 12.0
    pppm_accuracy: float = 1.0e-4

    root: Path = Path("runs")

    @property
    def nvt_ns(self):
        return self.nvt_steps * self.timestep_fs / 1e6

    @property
    def npt_ns(self):
        return self.npt_steps * self.timestep_fs / 1e6

    @property
    def gk_ns(self):
        return self.gk_steps * self.timestep_fs / 1e6

    @property
    def total_ns(self):
        return self.nvt_ns + self.npt_ns + self.gk_ns

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id
