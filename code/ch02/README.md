# Chapter 2: Gaussian Basis Sets and Orthonormalization

This directory contains companion Python code for Chapter 2 of the Advanced Quantum Chemistry lecture notes (Course 2302638).

## Overview

Chapter 2 explores how non-orthogonal atomic orbital (AO) basis sets lead to the generalized eigenvalue problem FC = SCe, and how to solve it via orthonormalization. The code implements:

- Overlap matrix eigenvalue analysis and conditioning diagnostics
- Three orthogonalization methods: canonical, symmetric (Lowdin), and Gram-Schmidt
- Eigenvalue thresholding for handling near-linear dependence
- Transformation of generalized eigenproblems to standard form
- Validation against scipy and PySCF reference implementations

## Files

| File | Lab/Exercise | Description |
|------|--------------|-------------|
| `lab2a_overlap_conditioning.py` | Lab 2A | Overlap matrix eigenvalue spectra and condition number analysis |
| `lab2b_orthogonalizers.py` | Lab 2B | Build X such that X^T S X = I using three methods |
| `lab2c_gen_eigenproblem.py` | Lab 2C | Solve FC = SCe via orthogonalization |

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

# Lab 2A: Overlap conditioning analysis
python lab2a_overlap_conditioning.py

# Lab 2B: Build orthogonalizers
python lab2b_orthogonalizers.py

# Lab 2C: Generalized eigenvalue problem
python lab2c_gen_eigenproblem.py
```

## Key Concepts

### Overlap Matrix and Conditioning (Lab 2A)

The overlap matrix S measures how basis functions overlap:
```
S_uv = <u|v> = integral u*(r) v(r) dr
```

Key properties:
- S is symmetric and positive definite for linearly independent basis functions
- Eigenvalues of S measure the "size" of the basis in different directions
- Small eigenvalues indicate near-linear dependence (ill-conditioning)
- Condition number kappa(S) = lambda_max / lambda_min measures numerical stability

Effect of basis set choice:
- Minimal basis (STO-3G): Well-conditioned, kappa(S) ~ 10^1
- Split-valence (cc-pVDZ): Moderate conditioning, kappa(S) ~ 10^2
- Diffuse functions (aug-cc-pVDZ): Worse conditioning, kappa(S) ~ 10^4 - 10^6

### Orthogonalizers (Lab 2B)

Three methods to build X such that X^T S X = I:

**Canonical Orthogonalizer:**
```
Given S = U diag(e) U^T, build X = U_kept diag(e_kept^{-1/2})
```
- Most stable for general use
- Explicitly removes near-linear dependent directions
- Standard choice in production codes

**Symmetric (Lowdin) Orthogonalizer:**
```
X = S^{-1/2} = U diag(e^{-1/2}) U^T
```
- Produces orthonormal functions "closest" to original AOs
- Minimizes sum of squared differences from original basis
- With thresholding, equivalent to canonical in practice

**Gram-Schmidt (S-metric):**
```
Sequential orthogonalization: v_new = v - sum_j (x_j^T S v) x_j
```
- Pedagogical value only
- Numerically unstable for ill-conditioned S
- Order-dependent results

### Generalized Eigenvalue Problem (Lab 2C)

The Roothaan-Hall equations FC = SCe require solving a generalized eigenproblem.

**Algorithm:**
```
1. Build orthogonalizer X such that X^T S X = I
2. Transform to orthonormal basis: F' = X^T F X
3. Solve ordinary eigenproblem: F' C' = C' e
4. Back-transform: C = X C'
```

**Verification criteria:**
- Orthonormality: C^T S C = I (Frobenius error < 1e-10)
- Eigenvalue equation: ||FC - SCe||_F < 1e-10
- Agreement with scipy.linalg.eigh and PySCF

### Eigenvalue Thresholding

For near-linear dependent basis sets:
```
Keep eigenvalues where e > threshold (typically 1e-10)
Dropped dimensions indicate redundant basis directions
```

Threshold selection trade-offs:
- Too small (1e-14): May retain numerical noise
- Too large (1e-6): May discard physically meaningful directions
- Recommended (1e-10): Balances stability and flexibility

## Learning Objectives

After completing these labs, students should be able to:

1. **Analyze basis set conditioning** by computing overlap matrix eigenvalue spectra
2. **Understand the relationship** between diffuse functions and linear dependence
3. **Implement orthogonalizers** using eigendecomposition with thresholding
4. **Transform generalized eigenproblems** to standard form and back
5. **Verify numerical correctness** using orthonormality and residual checks
6. **Compare implementations** against scipy and PySCF references

## Validation

All implementations are validated against reference calculations:

- `lab2a_overlap_conditioning.py`: Compares eigenvalue spectra across basis sets
- `lab2b_orthogonalizers.py`: Verifies X^T S X = I to machine precision for all methods
- `lab2c_gen_eigenproblem.py`: Validates eigenvalues against scipy.linalg.eigh and PySCF

## Connection to Lecture Notes

This code corresponds to:

- Section 2.4: The S-metric and generalized inner product
- Section 2.5: Gram-Schmidt orthonormalization (pedagogical)
- Section 2.6: Symmetric and canonical orthogonalizers
- Section 2.7: Algorithm 2.2 (stable orthonormalizer with thresholding)
- Section 2.8: Algorithm 2.3 (solving FC = SCe via orthogonalization)
- Section 2.10: Hands-on Python Labs

The implementations build foundations for Chapter 6 (Hartree-Fock SCF), where the generalized eigenvalue problem is solved iteratively with the Fock matrix.

## References

- Szabo & Ostlund, "Modern Quantum Chemistry", Section 3.4.5
- Helgaker, Jorgensen, Olsen, "Molecular Electronic-Structure Theory", Section 1.3
- Lowdin, P.-O., J. Chem. Phys. 18, 365 (1950) (Symmetric orthogonalization)
