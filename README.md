# Advanced Quantum Chemistry - Companion Code

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Course**: 2302638 Advanced Quantum Chemistry
**Institution**: Department of Chemistry, Faculty of Science, Chulalongkorn University
**Part I**: Electron Integrals and Hartree-Fock with Rys Quadrature (7 weeks)

**Author**: Viwat Vchirawongkwin
**Contact**: viwat.v@chula.ac.th

## Overview

This repository contains Python implementations accompanying the lecture notes for Advanced Quantum Chemistry. The code demonstrates computational quantum chemistry from an **integral-driven perspective**, emphasizing that electronic structure theory reduces to computing one- and two-electron integrals followed by linear algebra.

All scripts validate against [PySCF](https://pyscf.org/) reference calculations.

## Requirements

```bash
pip install numpy scipy pyscf matplotlib
```

Or use the requirements file:

```bash
pip install -r code/requirements.txt
```

## Repository Structure

```
code/
├── ch01-ch07/        # Weekly lab implementations
├── appendix_b-f/     # Reference implementations
├── requirements.txt  # Python dependencies
└── verify_setup.py   # Setup verification script
```

## Quick Start

```bash
# Verify your setup
python code/verify_setup.py

# Run a specific lab
python code/ch01/lab1a_integrals.py
```

## Chapter Contents

### Chapter 1: Electron-Integral View of Quantum Chemistry
> *The integral inventory that powers all of electronic structure theory*

- Born-Oppenheimer approximation and electronic Hamiltonian
- LCAO basis expansion and AO overlap metric
- Integral inventory: S (overlap), T (kinetic), V (nuclear attraction), ERIs
- Computational pipeline: Geometry → Basis → Integrals → SCF → Energy

**Lab**: Extract S, T, V, ERIs from PySCF; verify symmetries and electron count

### Chapter 2: Gaussian Basis Sets and Orthonormalization
> *Why Gaussians? Because their products are also Gaussians*

- Primitive and contracted Gaussian-type orbitals (GTOs)
- Cartesian vs spherical Gaussians
- Overlap matrix as metric and Gram-Schmidt orthonormalization
- Symmetric/canonical orthogonalization: X = S^(-1/2)
- Generalized eigenproblems and near-linear dependence

**Lab**: Build orthogonalizers, study conditioning vs basis set size

### Chapter 3: One-Electron Integrals and Gaussian Product Theorem
> *The theorem that makes molecular integral evaluation tractable*

- **Gaussian Product Theorem (GPT)**: product of two Gaussians = new Gaussian at composite center
- GPT parameters: p = α+β, μ = αβ/(α+β), P = (αA+βB)/(α+β)
- Analytic s-s overlap and kinetic integral derivation
- Nuclear attraction integrals and Boys function preview
- Dipole integrals for molecular properties

**Lab**: Validate analytic formulas vs PySCF; compute dipole from density

### Chapter 4: Two-Electron Integrals and Rys Quadrature Foundations
> *The O(N⁴) bottleneck and how to tame it*

- ERIs in chemist's notation: (μν|λσ)
- 8-fold ERI symmetry and Schwarz screening
- Primitive (ss|ss) ERI closed form with Boys function
- Boys function F_n(T): definition, series expansion, recurrence relations
- Root count rule: n_roots = floor(L/2) + 1

**Lab**: Implement stable Boys function; verify (ss|ss) against PySCF

### Chapter 5: Rys Quadrature in Practice
> *From moments to nodes: Gaussian quadrature for electron repulsion*

- Boys function as moments: m_n(T) = 2F_n(T)
- Orthogonal polynomial construction from Hankel matrices
- Golub-Welsch algorithm for Rys nodes and weights
- Higher angular momentum ERIs via Rys quadrature
- Building J and K matrices from ERIs

**Lab**: Implement Rys quadrature; compute ERIs; build J/K and validate

### Chapter 6: Hartree-Fock SCF from Integrals
> *Putting it all together: self-consistent field theory*

- HF variational principle and Fock operator
- Roothaan-Hall equations: FC = SCε
- SCF loop: P → F(P) → (ε,C) → P_new → convergence check
- DIIS acceleration for robust convergence
- In-core vs direct SCF strategies

**Lab**: Implement RHF SCF from scratch; add DIIS; validate against PySCF

### Chapter 7: Scaling, Density Fitting, and Properties
> *Making it fast and extracting physical observables*

- Scaling analysis: ERIs O(N⁴), diagonalization O(N³)
- Density Fitting (DF) / Resolution of Identity (RI)
- Three-index tensor factorization: (μν|λσ) ≈ Σ_Q B_μν^Q B_λσ^Q
- Virial theorem diagnostics
- One-electron properties: dipole moments from Tr[Po]

**Lab**: Compare conventional HF vs DF-HF; compute dipole; virial diagnostics

## Key Concepts

### Gaussian Product Theorem
```python
# Product of two Gaussians = new Gaussian at composite center
p = alpha + beta                      # Combined exponent
P = (alpha * A + beta * B) / p        # Composite center
K = exp(-alpha * beta / p * |A-B|^2)  # Pre-exponential factor
```

### Boys Function
```python
# F_n(T) = integral from 0 to 1 of t^(2n) * exp(-T*t^2) dt
# Special cases:
#   F_n(0) = 1/(2n+1)
#   F_0(T) = (1/2) * sqrt(pi/T) * erf(sqrt(T))  for T > 0
```

### Hartree-Fock from Integrals
```python
S = mol.intor("int1e_ovlp")       # Overlap
T = mol.intor("int1e_kin")        # Kinetic
V = mol.intor("int1e_nuc")        # Nuclear attraction
eri = mol.intor("int2e")          # ERIs

h = T + V                         # Core Hamiltonian
J = np.einsum('ijkl,kl->ij', eri, P)  # Coulomb
K = np.einsum('ikjl,kl->ij', eri, P)  # Exchange
F = h + J - 0.5 * K               # Fock matrix
E = 0.5 * np.trace(P @ (h + F))   # Electronic energy
```

## Conventions

| Convention | Description |
|------------|-------------|
| **Units** | Atomic units (m_e = ℏ = e = 4πε₀ = 1) |
| **ERI notation** | Chemist's notation (μν\|λσ) |
| **RHF density** | P_μν = 2 Σᵢ C_μi C_νi (factor of 2 for closed-shell) |
| **Fock matrix** | F = h + J - ½K |
| **HF energy** | E = ½Tr[P(h+F)] + E_nuc |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

### Textbooks

1. Szabo, A. & Ostlund, N. S. *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory* (Dover, 1996). ISBN: 978-0486691862

2. Helgaker, T., Jorgensen, P. & Olsen, J. *Molecular Electronic-Structure Theory* (Wiley, 2000). [DOI: 10.1002/9781119019572](https://doi.org/10.1002/9781119019572)

### Primary Literature

3. Dupuis, M., Rys, J. & King, H. F. Evaluation of molecular integrals over Gaussian basis functions. *J. Chem. Phys.* **65**, 111-116 (1976). [DOI: 10.1063/1.432807](https://doi.org/10.1063/1.432807)

### Software

4. Sun, Q. et al. PySCF: the Python-based simulations of chemistry framework. *WIREs Comput. Mol. Sci.* **8**, e1340 (2018). [DOI: 10.1002/wcms.1340](https://doi.org/10.1002/wcms.1340)

5. Sun, Q. et al. Recent developments in the PySCF program package. *J. Chem. Phys.* **153**, 024109 (2020). [DOI: 10.1063/5.0006074](https://doi.org/10.1063/5.0006074)

6. Sun, Q. Libcint: An efficient general integral library for Gaussian basis functions. *J. Comput. Chem.* **36**, 1664-1671 (2015). [DOI: 10.1002/jcc.23981](https://doi.org/10.1002/jcc.23981)

7. [PySCF Documentation](https://pyscf.org/user/index.html)
