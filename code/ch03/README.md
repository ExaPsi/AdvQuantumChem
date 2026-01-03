# Chapter 3: One-Electron Integrals and Gaussian Product Theorem

This directory contains companion Python code for Chapter 3 of the Advanced Quantum Chemistry lecture notes (Course 2302638).

## Overview

Chapter 3 introduces the fundamental building blocks for molecular integral evaluation. The code implements:

- Gaussian Product Theorem (GPT) parameter computation and verification
- Analytic s-s overlap, kinetic, and nuclear attraction integrals
- Boys function evaluation with multiple numerical strategies
- Overlap decay and integral screening concepts
- Dipole moment calculation from density matrix and one-electron integrals

## Files

| File | Lab/Exercise | Description |
|------|--------------|-------------|
| `gaussian_product_theorem.py` | Section 3.2 | GPT parameter exploration and numerical verification |
| `lab3a_integral_sanity.py` | Lab 3A | One-electron integral extraction and property checks |
| `lab3b_analytic_vs_pyscf.py` | Lab 3B | Analytic S, T, V formulas validated against PySCF |
| `lab3c_dipole_moment.py` | Lab 3C | Dipole moment from density and operator integrals |
| `overlap_screening.py` | Exercise 3.2 | Overlap decay and screening radius analysis |
| `boys_function_exploration.py` | Exercise 3.8 | Boys function F_n(T) evaluation methods |

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

# GPT exploration
python gaussian_product_theorem.py

# Lab 3A: Integral sanity checks
python lab3a_integral_sanity.py

# Lab 3B: Analytic vs PySCF
python lab3b_analytic_vs_pyscf.py

# Lab 3C: Dipole moment
python lab3c_dipole_moment.py

# Exercise 3.2: Overlap screening
python overlap_screening.py

# Exercise 3.8: Boys function
python boys_function_exploration.py
```

## Key Concepts

### Gaussian Product Theorem (GPT)

The product of two Gaussians centered at different points is itself a Gaussian:

```
exp(-alpha|r-A|^2) * exp(-beta|r-B|^2) = exp(-p|r-P|^2) * exp(-mu*R_AB^2)

where:
    p = alpha + beta              (total exponent)
    mu = alpha*beta/(alpha+beta)  (reduced exponent)
    P = (alpha*A + beta*B)/p      (composite center)
    R_AB = |A - B|                (distance between centers)
```

Key insights:
- Product remains Gaussian (computational convenience)
- Composite center P is weighted average of A and B
- Tighter Gaussians (larger exponent) anchor P toward their center
- Prefactor K = exp(-mu*R^2) enables screening

### Analytic s-s Integrals (Lab 3B)

For normalized s-type primitive Gaussians:

```
Overlap:   S_ab = N_a * N_b * (pi/p)^(3/2) * exp(-mu * R_AB^2)

Kinetic:   T_ab = mu * (3 - 2*mu*R_AB^2) * S_ab

Nuclear:   V_ab = -Z * N_a * N_b * (2*pi/p) * exp(-mu*R_AB^2) * F_0(p*|P-C|^2)
```

where N_a = (2*alpha/pi)^(3/4) is the normalization constant.

### Boys Function (Exercise 3.8)

The Boys function appears in Coulomb integrals:

```
F_n(T) = integral_0^1 t^(2n) exp(-T*t^2) dt
```

Key properties:
- F_n(0) = 1/(2n+1)
- F_0(T) = (1/2) sqrt(pi/T) erf(sqrt(T)) for T > 0
- Upward recursion: F_{n+1}(T) = [(2n+1)F_n(T) - exp(-T)] / (2T)
- Downward recursion: F_n(T) = [2T F_{n+1}(T) + exp(-T)] / (2n+1)

Numerical considerations:
- Series expansion stable for small T
- Upward recursion unstable for small T and large n
- Downward recursion numerically stable

### Overlap Screening (Exercise 3.2)

Overlap decays exponentially with distance:

```
S_ab ~ exp(-mu * R_AB^2)
```

Screening implications:
- Distant orbitals have negligible overlap
- Screening radius depends on exponents (diffuse orbitals have longer range)
- Enables sparse matrix storage for large molecules
- Foundation for linear-scaling methods

### Dipole Moment (Lab 3C)

Molecular dipole from density matrix and position integrals:

```
mu = sum_A Z_A * (R_A - O) - Tr[P * r(O)]
   = nuclear contribution - electronic contribution
```

Key insights:
- One-electron property (no ERIs needed)
- Origin-independent for neutral molecules
- Individual contributions change with origin, but total is invariant
- Conversion: 1 e*a_0 = 2.541746 Debye

## Learning Objectives

After completing these labs, students should be able to:

1. **Apply the GPT** to compute composite center, total exponent, and prefactor
2. **Derive and implement** analytic s-s integral formulas
3. **Evaluate Boys functions** using series expansion and recurrence relations
4. **Understand numerical stability** of different evaluation strategies
5. **Predict screening behavior** based on Gaussian exponents
6. **Compute molecular properties** from density matrix and one-electron integrals

## Validation

All implementations are validated against PySCF reference calculations:

- `gaussian_product_theorem.py`: Verifies GPT identity to machine precision at random points
- `lab3a_integral_sanity.py`: Checks integral symmetries, signs, and electron count
- `lab3b_analytic_vs_pyscf.py`: Validates S, T, V elements against PySCF int1e
- `lab3c_dipole_moment.py`: Compares dipole with PySCF dip_moment()
- `overlap_screening.py`: Validates decay formula S ~ exp(-mu*R^2)
- `boys_function_exploration.py`: Cross-validates series, recursion, and scipy implementations

## Connection to Lecture Notes

This code corresponds to:

- Section 3.2: The Gaussian Product Theorem
- Section 3.3: Overlap Integrals for s-Type Gaussians
- Section 3.4: Kinetic Energy Integrals
- Section 3.5: Nuclear Attraction Integrals and Boys Function
- Section 3.6: Operator Integrals for Properties
- Section 3.7: Hands-on Python Labs

The implementations serve as building blocks for Chapter 4 (Two-Electron Integrals) and Chapter 5 (Rys Quadrature), where we extend to four-center integrals and higher angular momentum.

## References

- Szabo & Ostlund, "Modern Quantum Chemistry", Appendix A
- Helgaker, Jorgensen, Olsen, "Molecular Electronic-Structure Theory", Chapter 9
- Taketa, Huzinaga, O-ohata, J. Phys. Soc. Japan 21 (1966) 2313 (Gaussian integral formulas)
