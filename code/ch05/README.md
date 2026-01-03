# Chapter 5: Rys Quadrature in Practice

This directory contains companion Python code for Chapter 5 of the Advanced Quantum Chemistry lecture notes (Course 2302638).

## Overview

Chapter 5 puts the Rys quadrature theory from Chapter 4 into practice. The code implements:

- Boys function evaluation and moment computation
- Algorithm 5.1: Rys quadrature nodes/weights from moments via Hankel matrices
- Moment matching verification
- First angular momentum ERI: (p s|ss) via derivative identity
- Coulomb (J) and Exchange (K) matrix construction
- Complete HF energy validation from integrals

## Files

| File | Lab/Exercise | Description |
|------|--------------|-------------|
| `boys_moments.py` | Lab 5A | Boys function F_n(T) and moments m_n(T) = 2*F_n(T) |
| `rys_quadrature.py` | Lab 5A | Algorithm 5.1: Hankel + Cholesky + Golub-Welsch |
| `moment_matching.py` | Exercise 5.2 | Verify quadrature exactness for n = 0, ..., 2*n_roots - 1 |
| `psss_eri.py` | Lab 5C | (p_xi s\|ss) ERI from derivative identity |
| `jk_build.py` | Lab 5D | Build J and K matrices from ERIs |
| `hf_energy_validation.py` | Exercise 5.5 | Complete HF energy from integrals |

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

# Lab 5A: Boys moments
python boys_moments.py

# Lab 5A continued: Rys quadrature construction
python rys_quadrature.py

# Exercise 5.2: Moment matching verification
python moment_matching.py

# Lab 5C: (p s|ss) ERI
python psss_eri.py

# Lab 5D: J and K matrices
python jk_build.py

# Exercise 5.5: Complete HF energy
python hf_energy_validation.py
```

## Key Concepts

### Boys Function as Moments (Lab 5A)

The Boys function can be written as a moment of a weighted integral:

```
F_n(T) = integral_0^1 t^{2n} exp(-T t^2) dt

With substitution x = t^2:

F_n(T) = (1/2) * integral_0^1 x^n * x^{-1/2} * exp(-Tx) dx
       = (1/2) * m_n(T)

where m_n(T) = integral_0^1 x^n * w_T(x) dx
with weight w_T(x) = x^{-1/2} * exp(-Tx)
```

### Algorithm 5.1: Rys Quadrature from Moments

```
Input: T >= 0, number of roots n_r
Output: nodes {x_i} in (0,1) and weights {W_i} > 0

1. Compute moments m_k = 2*F_k(T) for k = 0, 1, ..., 2*n_r - 1

2. Build Hankel matrices:
   H_ij = m_{i+j}         (Gram matrix of monomials)
   H^(1)_ij = m_{i+j+1}   (shifted Hankel)

3. Cholesky factorize: H = L L^T
   Compute C = L^{-1}

4. Build Jacobi matrix: J = C H^(1) C^T

5. Diagonalize J: eigenvalues are nodes x_i

6. Weights: W_i = m_0 * (V_{0,i})^2
   where V_{0,i} is first component of i-th eigenvector
```

### Moment Matching (Exercise 5.2)

For n_r-point Gaussian quadrature, the rule is exact for polynomials of degree <= 2*n_r - 1:

```
m_n = sum_i W_i * x_i^n    for n = 0, 1, ..., 2*n_r - 1

Equivalently:
F_n(T) = (1/2) * sum_i W_i * x_i^n    (exact for n <= 2*n_r - 1)
```

### (p_xi s|ss) ERI Formula (Lab 5C)

Using the derivative identity to promote s-type to p-type:

```
(p_xi s|ss) = (1/(2*alpha)) * d/dA_xi (ss|ss)

Closed form:
(p_xi s|ss) = prefactor * exp(-mu*R_AB^2) * exp(-nu*R_CD^2)
              * [-(beta/p)(A_xi - B_xi) F_0(T) - (rho/p)(P_xi - Q_xi) F_1(T)]

Key insight: Angular momentum increases the Boys order required!
- (ss|ss) needs only F_0
- (ps|ss) needs F_0 and F_1
- General rule: n_roots = floor(L/2) + 1, where L = sum of angular momenta
```

### J and K Matrix Construction (Lab 5D)

Coulomb and Exchange matrices from ERIs in chemist's notation:

```
J_mn = sum_ls (mn|ls) P_ls    (classical Coulomb)
K_mn = sum_ls (ml|ns) P_ls    (quantum Exchange)

Note the "crossed" indices in K: reflects electron exchange!

In numpy einsum:
J = einsum('ijkl,lk->ij', eri, P)
K = einsum('ikjl,lk->ij', eri, P)
```

### HF Energy Formula (Exercise 5.5)

```
E_1e = Tr[P*h]                              (one-electron)
E_2e = (1/2) Tr[P*J] - (1/4) Tr[P*K]        (two-electron)
E_elec = E_1e + E_2e = (1/2) Tr[P*(h + F)]  (electronic)
E_total = E_elec + E_nuc                    (total)

where F = h + J - (1/2)*K is the Fock matrix.

This demonstrates: HF = Integrals + Linear Algebra!
```

## Validation

All implementations are validated against PySCF reference calculations:

- `boys_moments.py`: Validates against scipy's hyp1f1 and direct integration
- `rys_quadrature.py`: Verifies moment matching to machine precision
- `moment_matching.py`: Comprehensive sweep over T from 1e-8 to 100
- `psss_eri.py`: Validates (p s|ss) ERI against PySCF int2e
- `jk_build.py`: Compares J and K against PySCF get_jk()
- `hf_energy_validation.py`: Full energy decomposition against PySCF SCF

## Connection to Lecture Notes

This code corresponds to:

- Section 5.2: From Boys functions to moments
- Section 5.3: Orthonormal polynomials and the Jacobi matrix
- Section 5.4: Algorithm 5.1 (nodes/weights from moments)
- Section 5.5: Evaluating Boys functions by Rys quadrature
- Section 5.6: (p_xi s|ss) from derivatives
- Section 5.7: Building J and K from ERIs
- Section 5.8: Hands-on Python Labs

## Exercises Covered

| Exercise | Description | Implementation |
|----------|-------------|----------------|
| 5.1 | One-root quadrature formulas | `rys_quadrature.py` |
| 5.2 | Two-root moment matching | `moment_matching.py` |
| 5.3 | (ss\|ss) by three routes | Uses code from ch04/ |
| 5.4 | (p_xi s\|ss) derivation | `psss_eri.py` |
| 5.5 | J/K build and energy check | `hf_energy_validation.py` |

## References

- Szabo & Ostlund, "Modern Quantum Chemistry", Chapter 3
- Helgaker, Jorgensen, Olsen, "Molecular Electronic-Structure Theory", Chapter 9
- Golub & Welsch, Math. Comp. 23 (1969) 221-230 (Gaussian quadrature from moments)
- Dupuis, Rys, King, J. Chem. Phys. 65 (1976) 111 (Rys quadrature for ERIs)
