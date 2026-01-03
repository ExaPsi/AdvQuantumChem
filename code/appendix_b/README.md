# Appendix B: Atomic Units and Conversions - Companion Code

This directory contains Python scripts demonstrating atomic unit conventions and unit conversions in quantum chemistry, complementing Appendix B of the Advanced Quantum Chemistry lecture notes.

## Scripts Overview

| Script | Description |
|--------|-------------|
| `geometry_units.py` | Geometry input unit specification in PySCF (Angstrom vs Bohr) |
| `unit_conversions.py` | CODATA 2018 conversion factors and utility functions |
| `unit_error_detection.py` | Common unit mistakes and geometry sanity checks |
| `dipole_calculation.py` | Dipole moment calculation with unit conversion to Debye |

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
cd /home/vyv/Documents/2302638/LectureNote/code/appendix_b

# Geometry unit specification
python geometry_units.py

# Unit conversion factors
python unit_conversions.py

# Unit error detection
python unit_error_detection.py

# Dipole moment calculation
python dipole_calculation.py
```

## Key Concepts Demonstrated

### 1. Geometry Unit Specification (`geometry_units.py`)

PySCF accepts molecular geometries in either Angstrom or Bohr:

```python
# Angstrom (most common in chemistry)
mol = gto.M(
    atom="H 0 0 0; H 0 0 0.74",
    basis="sto-3g",
    unit="Angstrom"
)

# Bohr (atomic units)
mol = gto.M(
    atom="H 0 0 0; H 0 0 1.398",
    basis="sto-3g",
    unit="Bohr"
)
```

**Key point**: Internally, PySCF always stores coordinates in Bohr.

### 2. Unit Conversion Factors (`unit_conversions.py`)

CODATA 2018 conversion factors:

| Quantity | Conversion | Value |
|----------|------------|-------|
| Length | 1 a0 -> Angstrom | 0.529177 |
| Length | 1 Angstrom -> a0 | 1.889726 |
| Energy | 1 Eh -> eV | 27.2114 |
| Energy | 1 Eh -> kcal/mol | 627.509 |
| Energy | 1 Eh -> kJ/mol | 2625.50 |
| Energy | 1 Eh -> cm^-1 | 219475 |
| Dipole | 1 e*a0 -> Debye | 2.5417 |

Utility functions:
```python
from unit_conversions import (
    bohr_to_angstrom,
    angstrom_to_bohr,
    hartree_to_ev,
    hartree_to_kcalmol,
    hartree_to_kjmol,
    dipole_au_to_debye
)

# Example usage
E_ev = hartree_to_ev(E_hartree)
R_ang = bohr_to_angstrom(R_bohr)
```

### 3. Unit Error Detection (`unit_error_detection.py`)

Common pitfall: interpreting Angstrom values as Bohr (or vice versa)

```python
R = 0.74  # Intended as Angstrom

# CORRECT
mol = gto.M(atom=f"H 0 0 0; H 0 0 {R}", unit="Angstrom")

# WRONG - catastrophic error!
mol = gto.M(atom=f"H 0 0 0; H 0 0 {R}", unit="Bohr")
# This gives R = 0.39 Angstrom (way too short!)
```

Sanity check function:
```python
def check_geometry_sanity(mol):
    """Flag bonds outside 0.5-5.0 Angstrom range."""
    coords = mol.atom_coords()  # Always in Bohr
    # Check all pairwise distances...
```

### 4. Dipole Moment Calculation (`dipole_calculation.py`)

Computing dipole moments from density matrix:

```python
# Electronic contribution
with mol.with_common_orig((0, 0, 0)):
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
mu_elec = -np.einsum('xij,ji->x', ao_dip, dm)

# Nuclear contribution
charges = mol.atom_charges()
coords = mol.atom_coords()
mu_nuc = np.einsum('i,ix->x', charges, coords)

# Total (in atomic units)
mu_au = mu_elec + mu_nuc

# Convert to Debye
mu_debye = mu_au * 2.5417
```

Validation against PySCF:
```python
mu_pyscf = mf.dip_moment(verbose=0)  # Returns Debye directly
```

## Atomic Units Reference

In atomic units, the following fundamental constants equal 1:

| Constant | Symbol | SI Value |
|----------|--------|----------|
| Electron mass | m_e | 9.109 x 10^-31 kg |
| Elementary charge | e | 1.602 x 10^-19 C |
| Reduced Planck constant | hbar | 1.055 x 10^-34 J*s |
| Coulomb constant | 4*pi*epsilon_0 | 1.113 x 10^-10 F/m |

Derived atomic units:

| Quantity | Unit | Name | SI Value |
|----------|------|------|----------|
| Length | a0 | Bohr radius | 0.529 Angstrom |
| Energy | Eh | Hartree | 27.2 eV |
| Time | hbar/Eh | | 2.42 x 10^-17 s |

## Common Errors to Avoid

1. **Mixing Angstrom and Bohr**: Always check the `unit` parameter in `gto.M()`
2. **Forgetting internal units**: `mol.atom_coords()` always returns Bohr
3. **Dipole sign convention**: Electronic contribution is negative (electron has negative charge)
4. **Origin dependence**: Dipole integrals depend on the choice of origin

## Test Systems

| Molecule | Purpose |
|----------|---------|
| H2 | Unit conversion verification |
| H2O | Dipole moment calculation |

## References

### CODATA 2018 Fundamental Constants

- Tiesinga, E., Mohr, P. J., Newell, D. B., & Taylor, B. N. (2021). CODATA recommended values of the fundamental physical constants: 2018. *Rev. Mod. Phys.*, 93, 025010.
  [DOI: 10.1103/RevModPhys.93.025010](https://doi.org/10.1103/RevModPhys.93.025010)
- NIST Reference: https://physics.nist.gov/cuu/Constants/

### Software

- Sun, Q., et al. (2020). Recent developments in the PySCF program package. *J. Chem. Phys.*, 153, 024109.
  [DOI: 10.1063/5.0006074](https://doi.org/10.1063/5.0006074)
- PySCF documentation: https://pyscf.org

### Course Materials

- Lecture notes: Appendix B (Atomic Units and Conversions)
