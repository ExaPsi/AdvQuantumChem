# Chapter 1: Electron-Integral View of Quantum Chemistry

This directory contains companion Python code for Chapter 1 of the Advanced Quantum Chemistry lecture notes (Course 2302638).

## Overview

Chapter 1 introduces the integral-driven perspective of quantum chemistry. The central insight is that Hartree-Fock and almost all electronic structure methods reduce to:

1. Computing one- and two-electron integrals
2. Linear algebra operations (traces, contractions, diagonalization)

The code implements:

- Molecule construction and basis set specification in PySCF
- One-electron integral extraction (S, T, V, h)
- Two-electron integral (ERI) extraction with symmetry options
- Integral symmetry verification
- Electron count validation via Tr[PS]
- Energy reconstruction from integrals (Algorithm 1.1)
- Memory analysis for integral storage

## Files

| File | Lab/Exercise | Description |
|------|--------------|-------------|
| `lab1a_integrals.py` | Lab 1A | AO integral inventory and sanity checks |

## Dependencies

```bash
# Activate virtual environment
source ../../.venv/bin/activate

# Required packages
pip install numpy scipy pyscf
```

## Running the Code

```bash
# Activate environment first
source ../../.venv/bin/activate

# Lab 1A: Integral extraction and validation
python lab1a_integrals.py
```

## Key Concepts

### Integral Inventory

Hartree-Fock requires four types of integrals:

| Integral | Symbol | PySCF call | Physical meaning |
|----------|--------|------------|------------------|
| Overlap | S_uv | `mol.intor('int1e_ovlp')` | Basis function overlap |
| Kinetic | T_uv | `mol.intor('int1e_kin')` | Electron kinetic energy |
| Nuclear | V_uv | `mol.intor('int1e_nuc')` | Electron-nucleus attraction |
| ERI | (uv\|ls) | `mol.intor('int2e')` | Electron-electron repulsion |

The core Hamiltonian is h = T + V.

### Algorithm 1.1: Integral-Driven HF Energy

```
Input: Molecule geometry, basis set, converged density matrix P

1. Compute h = T + V  (one-electron integrals)
2. Compute (uv|ls)    (two-electron integrals)
3. Contract J_uv = sum_{ls} (uv|ls) P_ls  (Coulomb)
4. Contract K_uv = sum_{ls} (ul|vs) P_ls  (Exchange)
5. E_elec = Tr[P*h] + (1/2) Tr[P*(J - 0.5*K)]
6. E_tot = E_elec + E_nuc

Equivalently: E_elec = (1/2) Tr[P*(h + F)] where F = h + J - 0.5*K
```

### Electron Count Validation

In a non-orthonormal AO basis, the electron count is:

```
N_e = Tr[P*S] = sum_{uv} P_uv S_vu
```

This is NOT Tr[P] because the basis functions overlap. This serves as a critical sanity check.

### ERI Symmetries

For real orbitals, ERIs obey 8-fold permutation symmetry:

```
(uv|ls) = (vu|ls) = (uv|sl) = (vu|sl)
        = (ls|uv) = (sl|uv) = (ls|vu) = (sl|vu)
```

This reduces storage from N^4 to approximately N^4/8 unique elements.

### ERI Notation

PySCF uses chemist's notation (Mulliken notation):

```
(uv|ls) = integral chi_u(1) chi_v(1) (1/r12) chi_l(2) chi_s(2) dr1 dr2
```

This is NOT physicist's notation <ul|vs>. The code and lecture notes consistently use chemist's notation.

## Learning Objectives

After completing this lab, students should be able to:

1. **Build molecules in PySCF** using `gto.M()` with atom strings and basis set names
2. **Extract integrals** using `mol.intor()` for one- and two-electron integrals
3. **Understand ERI storage options** (s1, s4, s8) and their memory implications
4. **Verify integral symmetries** programmatically
5. **Validate electron count** using Tr[PS] as a sanity check
6. **Reconstruct HF energy** from integrals and density matrix
7. **Appreciate the integral-driven perspective**: HF = Integrals + Linear Algebra

## Expected Output

Running `lab1a_integrals.py` should produce:

```
============================================================
Lab 1A: AO Integral Inventory and Sanity Checks
============================================================

--- Building H2 molecule ---
Molecule: H2
Basis: STO-3G
Number of AOs: 2
Number of electrons: 2

--- Extracting one-electron integrals ---
Overlap matrix S shape: (2, 2)
...

--- Running RHF calculation ---
RHF converged: True
RHF total energy: -1.1167593073 Eh

--- Validating electron count ---
Tr[P*S] = 2.0000000000
Expected: 2
Electron count valid: True

--- Reconstructing energy from integrals ---
E_tot from SCF:     -1.1167593073 Eh
E_tot rebuilt:      -1.1167593073 Eh
Difference:         0.00e+00 Eh
Energy validated:   True

All validation checks PASSED!
```

## Validation

All implementations are validated against PySCF reference calculations:

- `lab1a_integrals.py`: Validates S and h symmetry, 8-fold ERI symmetry, electron count via Tr[PS], and energy reconstruction to machine precision

## Connection to Lecture Notes

This code corresponds to:

- Section 1.2: The electronic Hamiltonian
- Section 1.3: LCAO basis expansion
- Section 1.4: Integral inventory for HF
- Section 1.5: Computational pipeline overview
- Section 1.6: ERI scaling and symmetry
- Section 1.7: Hands-on Python Labs

The implementations serve as foundation for all subsequent chapters, establishing the integral-first perspective that guides the entire course.

## References

- Szabo, A.; Ostlund, N. S. *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory*, Chapters 2-3. Dover Publications, 1996. ISBN: 0-486-69186-1
- Helgaker, T.; Jorgensen, P.; Olsen, J. *Molecular Electronic-Structure Theory*, Chapter 9. Wiley, 2000. ISBN: 978-0-471-96755-2
- Sun, Q. et al. "Recent developments in the PySCF program package." *J. Chem. Phys.* **153**, 024109 (2020). [DOI: 10.1063/5.0006074](https://doi.org/10.1063/5.0006074)
- PySCF documentation: https://pyscf.org/
