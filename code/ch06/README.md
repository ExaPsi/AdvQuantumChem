# Chapter 6: Hartree-Fock SCF from Integrals

This directory contains companion Python code for Chapter 6 of the Advanced Quantum Chemistry lecture notes (Course 2302638).

## Overview

Chapter 6 implements the complete Hartree-Fock Self-Consistent Field (SCF) algorithm from AO integrals. The code demonstrates:

- Roothaan-Hall equations: FC = SC epsilon
- Symmetric orthogonalization: X = S^{-1/2}
- SCF iteration: P -> F(P) -> (epsilon, C) -> P_new -> repeat
- SCF convergence residual: R = FPS - SPF
- Pulay DIIS acceleration for faster convergence
- In-core vs direct SCF comparison

## Files

| File | Lab/Exercise | Description |
|------|--------------|-------------|
| `lab6a_rhf_scf.py` | Lab 6A | Minimal RHF SCF from AO integrals (Algorithm 6.1) |
| `lab6b_diis.py` | Lab 6B | Pulay DIIS implementation and difficult SCF cases |
| `lab6c_jk_comparison.py` | Lab 6C | Compare in-core vs direct J/K building |

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

# Lab 6A: Minimal RHF SCF
python lab6a_rhf_scf.py

# Lab 6B: DIIS implementation
python lab6b_diis.py

# Lab 6C: J/K building comparison
python lab6c_jk_comparison.py
```

## Key Concepts

### RHF SCF Iteration (Lab 6A)

The SCF loop implements a fixed-point iteration:

```
1. Build Fock matrix:    F = h + J - (1/2) K
2. Solve Roothaan-Hall:  F C = S C epsilon
3. Build density:        P = 2 * C_occ @ C_occ.T
4. Check convergence:    |E - E_old| < tol and ||R|| < tol
5. If not converged, go to step 1
```

Key equations (RHF closed-shell):
- Density matrix: `P = 2 * C_occ @ C_occ.T` (factor of 2 for doubly occupied)
- Fock matrix: `F = h + J - (1/2) K`
- Electronic energy: `E_elec = (1/2) Tr[P(h + F)]`
- SCF residual: `R = FPS - SPF` (vanishes at convergence)

### Symmetric Orthogonalization

Transform the generalized eigenvalue problem to standard form:
```
F C = S C epsilon  -->  F' C' = C' epsilon
where F' = X.T @ F @ X, C = X @ C', and X = S^{-1/2}
```

Linear dependence is handled by discarding small eigenvalues of S.

### Pulay DIIS (Lab 6B)

DIIS (Direct Inversion in the Iterative Subspace) accelerates SCF convergence by constructing an improved Fock matrix as a linear combination of previous iterations:

```
F_DIIS = sum_i c_i F_i

The coefficients c_i minimize:
    ||sum_i c_i r_i||^2  subject to  sum_i c_i = 1

This leads to the linear system:
    [B  -1] [c ]   [0 ]
    [-1  0] [lambda] = [-1]

where B_ij = r_i . r_j (dot product of error vectors)
```

DIIS typically reduces SCF iterations by a factor of 2-5.

### In-Core vs Direct SCF (Lab 6C)

Two strategies for building J and K matrices:

**In-core SCF:**
- Store full ERI tensor (pq|rs) in memory
- Contract with einsum: `J = einsum('pqrs,rs->pq', eri, P)`
- Memory: O(N^4), fast contractions
- Best for small systems (< 100 AOs)

**Direct SCF:**
- Compute ERIs on-the-fly, no storage
- Use Schwarz screening to skip negligible integrals
- Memory: O(1), recompute integrals each iteration
- Essential for large systems

## Learning Objectives

After completing this chapter's labs, students should be able to:

1. **Implement RHF SCF from scratch** using AO integrals from PySCF
2. **Understand orthogonalization** and why it converts F C = S C epsilon to a standard eigenvalue problem
3. **Explain SCF convergence** using the commutator residual R = FPS - SPF
4. **Implement Pulay DIIS** and understand why it accelerates convergence
5. **Compare in-core vs direct SCF** and explain the trade-offs
6. **Diagnose difficult SCF cases** (stretched bonds, small HOMO-LUMO gaps)

## Validation

All implementations are validated against PySCF reference calculations:

- `lab6a_rhf_scf.py`: Validates H2 and H2O energies against PySCF RHF (< 1e-8 Hartree)
- `lab6b_diis.py`: Tests convergence on difficult cases (stretched H2, N2)
- `lab6c_jk_comparison.py`: Verifies J/K matrices match PySCF's get_jk()

## Connection to Lecture Notes

This code corresponds to:

- Section 6.2: The Fock operator and Roothaan-Hall equations
- Section 6.3: Orthogonalization and the generalized eigenvalue problem
- Section 6.4: SCF iteration framework
- Section 6.5: SCF convergence and the commutator residual
- Section 6.6: DIIS acceleration (Algorithm 6.2)
- Section 6.7: In-core vs direct SCF
- Section 6.8: Hands-on Python Labs

## Exercises Covered

| Exercise | Description | Implementation |
|----------|-------------|----------------|
| 6.1 | Derive RHF energy expression | `lab6a_rhf_scf.py` |
| 6.2 | Why does R = FPS - SPF vanish at convergence? | `lab6a_rhf_scf.py` |
| 6.3 | DIIS coefficient interpretation | `lab6b_diis.py` |
| 6.4 | Stretched H2 convergence study | `lab6b_diis.py` |
| 6.5 | In-core vs direct timing comparison | `lab6c_jk_comparison.py` |

## Example Output

Running `lab6a_rhf_scf.py` produces:

```
RHF SCF Calculation
  Basis functions: 7
  Electrons: 10
  Occupied orbitals: 5
----------------------------------------------------------------------
  it=  1  E=-74.942079...  |R|=...  |dP|=...
  it=  2  E=-74.961...  dE=...  |R|=...  |dP|=...
  ...
  SCF converged in N iterations
  Final energy: -74.9659011695 Hartree
----------------------------------------------------------------------
Comparison
  Educational RHF: -74.9659011695 Hartree
  PySCF RHF:       -74.9659011695 Hartree
  Difference:      +1.00e-10 Hartree
----------------------------------------------------------------------
VALIDATION PASSED: Energy agrees within 1e-8 Hartree
```

## References

- Szabo & Ostlund, "Modern Quantum Chemistry", Chapter 3
- Helgaker, Jorgensen, Olsen, "Molecular Electronic-Structure Theory", Chapters 10-11
- P. Pulay, "Convergence acceleration of iterative sequences. The case of SCF iteration", Chem. Phys. Lett. 73, 393-398 (1980) - Original DIIS paper. [DOI: 10.1016/0009-2614(80)80396-4](https://doi.org/10.1016/0009-2614(80)80396-4)
- P. Pulay, "Improved SCF convergence acceleration", J. Comp. Chem. 3, 556-560 (1982) - DIIS improvements. [DOI: 10.1002/jcc.540030413](https://doi.org/10.1002/jcc.540030413)
- Q. Sun et al., "Recent developments in the PySCF program package", J. Chem. Phys. 153, 024109 (2020). [DOI: 10.1063/5.0006074](https://doi.org/10.1063/5.0006074)
