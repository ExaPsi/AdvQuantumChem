# Appendix C: Gaussian Integral Formula Sheet - Companion Code

This directory contains Python scripts implementing the fundamental Gaussian integrals used in quantum chemistry, complementing Appendix C of the Advanced Quantum Chemistry lecture notes.

## Scripts Overview

| Script | Description |
|--------|-------------|
| `gaussian_product_theorem.py` | GPT implementation and numerical verification |
| `overlap_integral.py` | s-s overlap integral with normalization |
| `kinetic_integral.py` | s-s kinetic energy integral |
| `boys_function.py` | Boys function F_n(T) (re-exports from appendix_d) |
| `nuclear_attraction.py` | Nuclear attraction integral using F_0 |
| `two_electron_integral.py` | (ss\|ss) ERI implementation with 8-fold symmetry check |
| `hf_energy_validation.py` | Complete HF energy from integrals (capstone validation) |

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
cd /home/vyv/Documents/2302638/LectureNote/code/appendix_c

# Gaussian Product Theorem
python gaussian_product_theorem.py

# Overlap integral
python overlap_integral.py

# Kinetic integral
python kinetic_integral.py

# Boys function
python boys_function.py

# Nuclear attraction
python nuclear_attraction.py

# Two-electron integral
python two_electron_integral.py

# Complete HF validation
python hf_energy_validation.py
```

## Key Concepts Demonstrated

### 1. Gaussian Product Theorem (`gaussian_product_theorem.py`)

The product of two Gaussians centered at A and B yields a Gaussian at composite center P:

```
exp(-alpha|r-A|^2) * exp(-beta|r-B|^2) = exp(-p|r-P|^2) * exp(-mu*R_AB^2)
```

GPT parameters:
- `p = alpha + beta` (composite exponent)
- `mu = alpha*beta/(alpha+beta)` (reduced exponent)
- `P = (alpha*A + beta*B)/(alpha+beta)` (composite center)
- `K_AB = exp(-mu*R_AB^2)` (pre-exponential factor)

### 2. Overlap Integral (`overlap_integral.py`)

For s-type Gaussians:

```
S_ab = (pi/p)^(3/2) * exp(-mu*R_AB^2)    (unnormalized)
```

With normalization N_s(alpha) = (2*alpha/pi)^(3/4).

### 3. Kinetic Integral (`kinetic_integral.py`)

Compact closed form for s-type functions:

```
T_ab = mu * (3 - 2*mu*R_AB^2) * S_ab
```

The factor (3 - 2*mu*R^2) arises from applying the Laplacian to Gaussians.

### 4. Boys Function (`boys_function.py`)

Central to all 1/r integrals:

```
F_n(T) = integral from 0 to 1 of t^(2n) * exp(-T*t^2) dt
```

Key properties:
- `F_n(0) = 1/(2n+1)`
- `F_0(T) = (1/2)*sqrt(pi/T)*erf(sqrt(T))` for T > 0
- Upward recurrence: `F_{n+1} = [(2n+1)*F_n - exp(-T)] / (2T)`

### 5. Nuclear Attraction (`nuclear_attraction.py`)

For s-type Gaussians and nucleus at C with charge Z_C:

```
V_ab(C) = -Z_C * (2*pi/p) * exp(-mu*R_AB^2) * F_0(p*|P-C|^2)
```

The negative sign indicates attractive interaction.

### 6. Two-Electron Integral (`two_electron_integral.py`)

The (ss|ss) ERI in chemist's notation:

```
(ab|cd) = (2*pi^(5/2)) / (p*q*sqrt(p+q))
          * exp(-mu_ab*R_AB^2) * exp(-nu_cd*R_CD^2) * F_0(T)
```

where `T = rho*|P-Q|^2` and `rho = p*q/(p+q)`.

8-fold ERI symmetry:
```
(ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)
```

### 7. HF Energy Validation (`hf_energy_validation.py`)

Demonstrates the central course theme: **HF reduces to integrals + linear algebra**

```python
# Core Hamiltonian
h = T + V

# Coulomb and Exchange from ERIs
J = np.einsum('ijkl,kl->ij', eri, P)
K = np.einsum('ikjl,kl->ij', eri, P)

# Fock matrix
F = h + J - 0.5 * K

# Total energy
E_elec = 0.5 * np.einsum('ij,ij->', P, h + F)
E_total = E_elec + E_nuc
```

## Test System

All scripts use H2 / STO-3G at R = 0.74 Angstrom as the standard test case:
- 2 basis functions (minimal)
- Closed-shell (2 electrons)
- Exact PySCF reference available

## Common PySCF Patterns

### Molecule Setup
```python
from pyscf import gto, scf

mol = gto.M(
    atom="H 0 0 0; H 0 0 0.74",
    basis="sto-3g",
    unit="Angstrom",
    verbose=0
)
```

### Integral Extraction
```python
S = mol.intor("int1e_ovlp")      # Overlap
T = mol.intor("int1e_kin")       # Kinetic
V = mol.intor("int1e_nuc")       # Nuclear attraction
eri = mol.intor("int2e")         # Two-electron integrals
```

### HF Reference
```python
mf = scf.RHF(mol)
E = mf.kernel()
dm = mf.make_rdm1()              # Density matrix
```

## Dependencies

- numpy
- scipy
- pyscf

Note: `boys_function.py` imports the stable implementation from `appendix_d/boys_function.py`.

## References

- Lecture notes: Chapter 3 (One-electron integrals), Chapter 4 (Two-electron integrals)
- Szabo & Ostlund, "Modern Quantum Chemistry"
- Helgaker, Jorgensen, Olsen, "Molecular Electronic-Structure Theory"
