# Appendix F: PySCF Interface Reference - Companion Code

This directory contains Python scripts demonstrating the PySCF interface for quantum chemistry calculations, complementing Appendix F of the Advanced Quantum Chemistry lecture notes.

## Scripts Overview

| Script | Description |
|--------|-------------|
| `molecule_construction.py` | All methods for specifying molecular geometry |
| `integral_inventory.py` | Comprehensive integral extraction (S, T, V, ERIs) |
| `eri_symmetry_demo.py` | Numerical verification of 8-fold ERI symmetries |
| `shell_access_demo.py` | Shell-by-shell integral access with `intor_by_shell` |
| `scf_from_scratch.py` | Minimal SCF using only PySCF integrals |
| `jk_build_comparison.py` | Manual J/K construction vs PySCF |
| `diis_demo.py` | DIIS acceleration for SCF convergence |
| `density_fitting_timing.py` | Conventional vs DF-HF comparison |
| `dipole_calculation.py` | Property calculation from density matrix |
| `debug_utilities.py` | Validation functions for student implementations |

## Running the Scripts

### Setup

```bash
# From repository root
source .venv/bin/activate

# Or use the venv Python directly
.venv/bin/python script_name.py
```

### Run Individual Scripts

```bash
cd /home/vyv/Documents/2302638/LectureNote/code/appendix_f

# Molecule construction examples
python molecule_construction.py

# Integral extraction
python integral_inventory.py

# ERI symmetries
python eri_symmetry_demo.py

# Shell-by-shell access
python shell_access_demo.py

# SCF from scratch
python scf_from_scratch.py

# J/K matrix construction
python jk_build_comparison.py

# DIIS acceleration
python diis_demo.py

# Density fitting comparison
python density_fitting_timing.py

# Dipole moment calculation
python dipole_calculation.py

# Debug utilities
python debug_utilities.py
```

## Key Concepts Demonstrated

### 1. Molecule Construction (`molecule_construction.py`)
- String format: `"H 0 0 0; H 0 0 0.74"`
- List format: `[('H', (0, 0, 0)), ('H', (0, 0, 0.74))]`
- Unit specification (Angstrom vs Bohr)
- Charged and open-shell systems
- Accessing molecule attributes

### 2. Integral Extraction (`integral_inventory.py`)
- One-electron: `mol.intor("int1e_ovlp")`, `mol.intor("int1e_kin")`, `mol.intor("int1e_nuc")`
- Two-electron: `mol.intor("int2e", aosym="s1")`
- Dipole integrals: `mol.intor("int1e_r")`
- Matrix property verification

### 3. ERI Symmetries (`eri_symmetry_demo.py`)
Eight-fold symmetry (chemist's notation):
```
(ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)
```

Storage options:
- `aosym='s1'`: Full N^4 tensor
- `aosym='s4'`: 4-fold symmetry
- `aosym='s8'`: 8-fold symmetry (most compact)

### 4. SCF Implementation (`scf_from_scratch.py`)
Key equations:
```python
# Core Hamiltonian
h = T + V

# Coulomb and Exchange
J = np.einsum('ijkl,kl->ij', eri, P)
K = np.einsum('ikjl,kl->ij', eri, P)

# Fock matrix
F = h + J - 0.5 * K

# Energy
E = 0.5 * np.einsum('ij,ij->', P, h + F) + E_nuc
```

### 5. DIIS Acceleration (`diis_demo.py`)
- Error vector: `e = FPS - SPF` (vanishes at convergence)
- Extrapolation: `F_new = sum_i c_i F_i`
- Constraint: `sum_i c_i = 1`

### 6. Density Fitting (`density_fitting_timing.py`)
Factorization:
```
(ij|kl) ~ sum_Q B_ij^Q B_kl^Q
```
Reduces scaling from O(N^4) to O(N^2 * N_aux)

### 7. Property Calculation (`dipole_calculation.py`)
```python
# Electronic dipole
mu_elec = -np.einsum('xij,ji->x', r_ints, dm)

# Nuclear dipole
mu_nuc = sum(Z_A * R_A for all atoms)

# Total
mu_total = mu_elec + mu_nuc
```

## Debug Utilities (`debug_utilities.py`)

Validation functions for student implementations:

```python
from debug_utilities import (
    validate_electron_count,
    validate_density_matrix,
    validate_energy,
    validate_convergence,
    check_hf_calculation,
    compare_implementations
)

# Check electron count
validate_electron_count(dm, S, n_elec)

# Comprehensive density matrix check
validate_density_matrix(dm, S, n_elec)

# Energy validation
validate_energy(dm, h, J, K, E_nuc, E_ref)

# Convergence check
validate_convergence(F, P, S)

# Complete HF calculation check
check_hf_calculation(mol, mf)

# Compare against PySCF
compare_implementations(mol, dm_user, E_user)
```

## Test Systems

All scripts use standard test molecules:

| Molecule | Basis | Purpose |
|----------|-------|---------|
| H2 | STO-3G | Minimal, exact verification |
| H2O | STO-3G, cc-pVDZ | Realistic closed-shell |
| HF | cc-pVDZ | Polar molecule |
| LiH | cc-pVDZ | Ionic character |
| (H2O)_2 | cc-pVDZ | Larger system timing |

## Common PySCF Patterns

### Molecule Setup
```python
from pyscf import gto, scf

mol = gto.M(
    atom="H 0 0 0; H 0 0 0.74",
    basis="sto-3g",
    unit="Angstrom",
    verbose=0
)
```

### Integral Extraction
```python
S = mol.intor("int1e_ovlp")      # Overlap
T = mol.intor("int1e_kin")       # Kinetic
V = mol.intor("int1e_nuc")       # Nuclear attraction
eri = mol.intor("int2e")         # Two-electron integrals
```

### Running HF
```python
mf = scf.RHF(mol)
E = mf.kernel()

# Access results
dm = mf.make_rdm1()    # Density matrix
C = mf.mo_coeff        # MO coefficients
eps = mf.mo_energy     # Orbital energies
```

## References

- Sun, Q.; et al. "Recent developments in the PySCF program package", J. Chem. Phys. 153 (2020) 024109. [DOI: 10.1063/5.0006074](https://doi.org/10.1063/5.0006074)
  - Documentation: https://pyscf.org
- Sun, Q. "Libcint: An efficient general integral library for Gaussian basis functions", J. Comput. Chem. 36 (2015) 1664-1671. [DOI: 10.1002/jcc.23981](https://doi.org/10.1002/jcc.23981)
  - Source: https://github.com/sunqm/libcint
- Lecture notes: Chapter 6 (SCF) and Appendix F
