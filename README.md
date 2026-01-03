# Advanced Quantum Chemistry - Companion Code

**Course**: 2302638 Advanced Quantum Chemistry
**Institution**: Department of Chemistry, Faculty of Science, Chulalongkorn University
**Part I**: Electron Integrals and Hartree-Fock with Rys Quadrature (7 weeks)

## Overview

This repository contains Python implementations accompanying the lecture notes for Advanced Quantum Chemistry. The code demonstrates computational quantum chemistry from an integral-driven perspective, emphasizing that electronic structure theory reduces to computing one- and two-electron integrals followed by linear algebra.

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
├── ch01/          # Week 1: Integral inventory and overview
├── ch02/          # Week 2: Basis sets and orthonormalization
├── ch03/          # Week 3: One-electron integrals and GPT
├── ch04/          # Week 4: Two-electron integrals and Boys function
├── ch05/          # Week 5: Rys quadrature in practice
├── ch06/          # Week 6: Hartree-Fock SCF implementation
├── ch07/          # Week 7: Scaling, density fitting, properties
├── appendix_b/    # Atomic units and conversions
├── appendix_c/    # Gaussian integral formulas
├── appendix_d/    # Boys function reference
├── appendix_e/    # Rys quadrature reference
├── appendix_f/    # PySCF/libcint interface
├── requirements.txt
└── verify_setup.py
```

## Quick Start

```bash
# Verify your setup
python code/verify_setup.py

# Run a specific lab
python code/ch01/lab1a_integrals.py
```

## Key Concepts Demonstrated

### Gaussian Product Theorem (Chapter 3)
```python
# Product of two Gaussians = new Gaussian at composite center
p = alpha + beta
P = (alpha * A + beta * B) / p
```

### Boys Function (Chapter 4)
```python
# F_n(T) = integral from 0 to 1 of t^(2n) * exp(-T*t^2) dt
```

### Hartree-Fock from Integrals (Chapter 6)
```python
S = mol.intor("int1e_ovlp")       # Overlap
T = mol.intor("int1e_kin")        # Kinetic
V = mol.intor("int1e_nuc")        # Nuclear attraction
eri = mol.intor("int2e")          # ERIs

h = T + V                         # Core Hamiltonian
J = np.einsum('ijkl,kl->ij', eri, P)  # Coulomb
K = np.einsum('ikjl,kl->ij', eri, P)  # Exchange
F = h + J - 0.5 * K               # Fock matrix
```

## Conventions

- **Units**: Atomic units (m_e = hbar = e = 1)
- **ERI notation**: Chemist's notation (μν|λσ)
- **RHF density**: P_μν = 2 Σᵢ C_μi C_νi (factor of 2 for closed-shell)

## License

Educational use. See course materials for full details.

## References

- Szabo & Ostlund, "Modern Quantum Chemistry"
- Helgaker, Jørgensen, Olsen, "Molecular Electronic-Structure Theory"
- [PySCF Documentation](https://pyscf.org/user.html)
