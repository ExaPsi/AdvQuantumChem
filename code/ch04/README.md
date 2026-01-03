# Chapter 4: Two-Electron Integrals and Rys Quadrature Foundations

This directory contains companion Python code for Chapter 4 of the Advanced Quantum Chemistry lecture notes (Course 2302638).

## Overview

Chapter 4 introduces electron repulsion integrals (ERIs) and the mathematical machinery needed to evaluate them efficiently. The code implements:

- Boys function evaluation with multiple numerical strategies
- The closed-form (ss|ss) primitive ERI formula
- Schwarz inequality for integral screening
- ERI symmetry verification
- Scaling analysis for ERI storage

## Files

| File | Lab/Exercise | Description |
|------|--------------|-------------|
| `boys_function.py` | Lab 4A | Boys function F_n(T) with series, recursion, and hybrid methods |
| `ssss_eri.py` | Lab 4B | Primitive (ss|ss) ERI calculation with PySCF validation |
| `boys_quadrature_comparison.py` | Lab 4C | Gauss-Legendre quadrature for Boys function |
| `schwarz_screening.py` | Exercise 4.3 | Schwarz inequality screening demonstration |
| `eri_symmetry.py` | Exercise 4.6 | 8-fold ERI symmetry verification |
| `eri_scaling.py` | Exercise 4.7 | ERI storage and computation scaling analysis |

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

# Lab 4A: Boys function
python boys_function.py

# Lab 4B: (ss|ss) ERI
python ssss_eri.py

# Lab 4C: Quadrature comparison
python boys_quadrature_comparison.py

# Exercise 4.3: Schwarz screening
python schwarz_screening.py

# Exercise 4.6: ERI symmetry
python eri_symmetry.py

# Exercise 4.7: ERI scaling
python eri_scaling.py
```

## Key Concepts

### Boys Function (Lab 4A)

The Boys function is defined as:
```
F_n(T) = integral_0^1 t^(2n) exp(-T t^2) dt
```

Key properties:
- F_n(0) = 1/(2n+1)
- F_0(T) = (1/2) sqrt(pi/T) erf(sqrt(T)) for T > 0
- Upward recursion: F_{n+1}(T) = [(2n+1)F_n(T) - exp(-T)] / (2T)
- Downward recursion: F_n(T) = [2T F_{n+1}(T) + exp(-T)] / (2n+1)

The hybrid evaluation strategy uses:
- Series expansion for small T (stable for all n)
- erf + upward recursion for large T (stable when T > 25)

### (ss|ss) ERI Formula (Lab 4B)

The fundamental primitive ERI for s-type Gaussians:
```
(ab|cd) = (2 pi^{5/2}) / (p q sqrt(p+q)) * exp(-mu R_AB^2) * exp(-nu R_CD^2) * F_0(T)
```

where:
- p = alpha + beta, q = gamma + delta (composite exponents)
- mu = alpha*beta/p, nu = gamma*delta/q (reduced exponents)
- P = (alpha*A + beta*B)/p, Q = (gamma*C + delta*D)/q (composite centers)
- rho = p*q/(p+q)
- T = rho * |P - Q|^2

### Schwarz Screening (Exercise 4.3)

The Schwarz inequality provides upper bounds for ERIs:
```
|(mu nu|lambda sigma)| <= sqrt((mu nu|mu nu)) * sqrt((lambda sigma|lambda sigma))
```

This allows rapid identification of negligible integrals without computing them.

### ERI Symmetries (Exercise 4.6)

For real orbitals, ERIs obey 8-fold permutation symmetry:
```
(mu nu|lambda sigma) = (nu mu|lambda sigma) = (mu nu|sigma lambda) = (nu mu|sigma lambda)
                     = (lambda sigma|mu nu) = (sigma lambda|mu nu) = (lambda sigma|nu mu) = (sigma lambda|nu mu)
```

### Scaling (Exercise 4.7)

- Full ERI tensor: O(N^4) storage
- With 8-fold symmetry: ~N^4/8 unique elements
- Memory becomes prohibitive for N_AO > ~100 (exceeds 1 GB)

## Validation

All implementations are validated against PySCF reference calculations:

- `boys_function.py`: Validates F_n(0) = 1/(2n+1) and F_0(1.0) against exact values
- `ssss_eri.py`: Validates all 16 elements of 2x2x2x2 ERI tensor against PySCF
- `schwarz_screening.py`: Verifies bounds are never violated for random ERI samples
- `eri_symmetry.py`: Checks all 8-fold symmetries to machine precision

## Connection to Lecture Notes

This code corresponds to:
- Section 4.5: The Fundamental Primitive ERI
- Section 4.6: The Boys Function
- Section 4.7: From Boys Functions to Quadrature
- Section 4.8: Hands-on Python Labs
- Section 4.10: Exercises

The implementations serve as building blocks for Chapter 5 (Rys Quadrature in Practice), where we extend to higher angular momentum integrals.

## References

### Textbooks

- Szabo, A. & Ostlund, N. S. *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory*, Dover Publications (1996), Appendix A. ISBN: 978-0-486-69186-2
- Helgaker, T., Jorgensen, P., & Olsen, J. *Molecular Electronic-Structure Theory*, Wiley (2000), Section 9.2. ISBN: 978-0-471-96755-2

### Primary Literature

- Dupuis, M., Rys, J., & King, H. F. Evaluation of molecular integrals over Gaussian basis functions. *J. Chem. Phys.* **65**, 111-116 (1976). [DOI: 10.1063/1.432807](https://doi.org/10.1063/1.432807)

### Software

- Sun, Q. et al. Recent developments in the PySCF program package. *J. Chem. Phys.* **153**, 024109 (2020). [DOI: 10.1063/5.0006074](https://doi.org/10.1063/5.0006074)
- Sun, Q. Libcint: An efficient general integral library for Gaussian basis functions. *J. Comput. Chem.* **36**, 1664-1671 (2015). [DOI: 10.1002/jcc.23981](https://doi.org/10.1002/jcc.23981)
