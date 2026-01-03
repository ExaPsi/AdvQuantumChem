# Chapter 7: Scaling and Properties

This directory contains companion Python code for Chapter 7 of the Advanced Quantum Chemistry lecture notes (Course 2302638).

## Overview

Chapter 7 analyzes the computational scaling of Hartree-Fock and introduces density fitting as a practical acceleration technique. The code also demonstrates how molecular properties are computed from the density matrix and integrals. Key implementations include:

- Conventional HF vs density-fitted HF comparison
- Dipole moment calculation from density matrix
- Virial theorem as a basis quality diagnostic
- Basis set conditioning analysis
- Scaling analysis of HF components

## Files

| File | Lab/Exercise | Description |
|------|--------------|-------------|
| `lab7a_df_hf_comparison.py` | Lab 7A | Conventional vs DF-HF timing and accuracy comparison |
| `lab7b_dipole_calculation.py` | Lab 7B | Dipole moment from density matrix and position integrals |
| `lab7c_virial_diagnostic.py` | Lab 7C | Virial theorem (2T + V = 0) as basis quality check |
| `basis_conditioning.py` | Exercise 7.4 | Overlap matrix conditioning and eigenvalue thresholding |
| `scaling_analysis.py` | Exercise 7.7 | Integral counts, memory estimates, and timing comparison |

## Dependencies

```bash
# Activate virtual environment
source ../../.venv/bin/activate

# Required packages
pip install numpy scipy pyscf
```

## Running the Code

Each file can be run as a standalone script:

```bash
# Activate environment first
source ../../.venv/bin/activate

# Lab 7A: DF-HF comparison
python lab7a_df_hf_comparison.py

# Lab 7B: Dipole moment
python lab7b_dipole_calculation.py

# Lab 7C: Virial theorem
python lab7c_virial_diagnostic.py

# Exercise 7.4: Basis conditioning
python basis_conditioning.py

# Exercise 7.7: Scaling analysis
python scaling_analysis.py
```

## Key Concepts

### Density Fitting / Resolution of Identity (Lab 7A)

The density fitting approximation factorizes the 4-index ERI tensor:

```
(mn|ls) = sum_Q B_mn^Q B_ls^Q

where B_mn^Q = sum_P (mn|P)(P|Q)^{-1/2}
```

Key properties:
- Reduces storage from O(N^4) to O(N^2 N_aux) ~ O(N^3)
- DF error (10^-5 to 10^-6 Eh) is much smaller than basis set error
- Speedup increases with system size

### One-Electron Properties (Lab 7B)

The general one-electron property formula:

```
<O> = Tr[P * o]

where P is the density matrix and o is the operator integral matrix.
```

For the dipole moment:

```
mu = mu_nuc - mu_el
mu_nuc = sum_A Z_A * R_A        (nuclear contribution)
mu_el = Tr[P * r]               (electronic contribution)
```

Key observations:
- Origin-independent for neutral molecules
- Origin-dependent for charged species (ions)
- Conversion: 1 a.u. = 2.5417 Debye

### Virial Theorem (Lab 7C)

For exact eigenstates of Coulombic Hamiltonians at equilibrium:

```
2<T> + <V> = 0

Virial ratio: eta = -<V>/<T> = 2
```

Diagnostic interpretation:
- |eta - 2| < 10^-2: Good basis quality at equilibrium
- |eta - 2| > 10^-1: Basis incompleteness or non-equilibrium geometry
- The theorem is exact only at stationary points of the energy

### Basis Set Conditioning (Exercise 7.4)

The condition number of the overlap matrix:

```
kappa(S) = s_max / s_min
```

Guidelines:
- log_10(kappa) < 6: Excellent conditioning
- log_10(kappa) < 8: Good conditioning
- log_10(kappa) < 10: May need eigenvalue thresholding
- log_10(kappa) > 10: Near-linear dependence, thresholding required

Eigenvalue thresholding removes redundant basis functions by discarding
eigenvectors of S with eigenvalues below a threshold (PySCF default: 10^-8).

### Scaling Analysis (Exercise 7.7)

Theoretical scaling of HF components:

```
One-electron integrals:     O(N^2)
Two-electron integrals:     O(N^4)
Fock matrix build:          O(N^4) conventional, O(N^3) DF
Matrix diagonalization:     O(N^3)
```

Memory estimates (8 bytes per double):
- Full ERI tensor (N=100): ~0.8 GB
- Full ERI tensor (N=200): ~12.8 GB
- Full ERI tensor (N=500): ~500 GB

This motivates density fitting for systems with N_AO > 100.

## Validation

All implementations are validated against PySCF reference calculations:

- `lab7a_df_hf_comparison.py`: Validates DF error < 10^-4 Eh and SCF convergence
- `lab7b_dipole_calculation.py`: Validates dipole against PySCF mf.dip_moment()
- `lab7c_virial_diagnostic.py`: Validates energy components and correct signs
- `basis_conditioning.py`: Validates eigenvalue analysis and thresholding
- `scaling_analysis.py`: Validates integral count formulas and timing measurements

## Connection to Lecture Notes

This code corresponds to:

- Section 7.2: Scaling Anatomy of HF
- Section 7.3: Density Fitting / Resolution of Identity
- Section 7.4: Basis Sets, Conditioning, and Practical Accuracy Controls
- Section 7.5: Virial Theorem as a Diagnostic
- Section 7.7: One-Electron Properties from Density and Integrals
- Section 7.8: Hands-on Python Labs

## Learning Objectives

After completing these labs, students should be able to:

1. Explain why density fitting reduces computational cost and when it is appropriate
2. Compute molecular properties from the density matrix using the Tr[P*o] formula
3. Use the virial theorem to assess basis set quality and geometry optimization
4. Diagnose and handle near-linear dependence in basis sets
5. Estimate memory requirements for HF calculations and identify bottlenecks

## References

- Szabo & Ostlund, "Modern Quantum Chemistry", Chapters 3 and 4
- Helgaker, Jorgensen, Olsen, "Molecular Electronic-Structure Theory", Chapters 8 and 9
- F. Weigend, "A fully direct RI-HF algorithm: Implementation, optimised auxiliary basis sets, demonstration of accuracy and efficiency", Phys. Chem. Chem. Phys. 4, 4285-4291 (2002). [DOI: 10.1039/b204199p](https://doi.org/10.1039/b204199p)
- T. H. Dunning, "Gaussian basis sets for use in correlated molecular calculations. I. The atoms boron through neon and hydrogen", J. Chem. Phys. 90, 1007-1023 (1989). [DOI: 10.1063/1.456153](https://doi.org/10.1063/1.456153)
- Q. Sun et al., "Recent developments in the PySCF program package", J. Chem. Phys. 153, 024109 (2020). [DOI: 10.1063/5.0006074](https://doi.org/10.1063/5.0006074)
