# Appendix E: Rys Quadrature Reference - Companion Code

This directory contains Python scripts demonstrating Rys quadrature for molecular integral evaluation, complementing Appendix E of the Advanced Quantum Chemistry lecture notes.

## Scripts Overview

| Script | Description |
|--------|-------------|
| `rys_quadrature.py` | Core Rys quadrature: moments, Hankel matrices, Golub-Welsch algorithm |
| `rys_eri_example.py` | Complete (ss|ss) ERI evaluation using Rys quadrature |
| `rys_validation.py` | Comprehensive validation suite for numerical accuracy |

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
cd /home/vyv/Documents/2302638/LectureNote/code/appendix_e

# Core quadrature implementation
python rys_quadrature.py

# ERI examples with PySCF comparison
python rys_eri_example.py

# Full validation suite
python rys_validation.py
```

## Key Concepts Demonstrated

### 1. Rys Quadrature Theory (`rys_quadrature.py`)

The Rys weight function and its moments:
```
w_T(x) = x^(-1/2) * exp(-T*x)   on [0, 1]
m_n(T) = integral_0^1 x^n * w_T(x) dx = 2 * F_n(T)
```

The quadrature rule with n_r nodes exactly reproduces moments 0 through 2*n_r - 1:
```
m_n(T) = sum_{i=1}^{n_r} W_i * x_i^n   for n = 0, 1, ..., 2*n_r - 1
```

### 2. Golub-Welsch Algorithm (`rys_quadrature.py`)

The moment-based algorithm for computing nodes and weights:
1. Compute moments: `m_k = 2*F_k(T)` for k = 0, ..., 2*n_r - 1
2. Build Hankel matrices: `H_ij = m_{i+j}`, `H^(1)_ij = m_{i+j+1}`
3. Cholesky factorize: `H = L * L^T`, define `C = L^(-1)`
4. Form Jacobi matrix: `J = C * H^(1) * C^T`
5. Diagonalize J: eigenvalues are nodes `x_i`
6. Weights: `W_i = m_0 * (V_{0i})^2` where V is the eigenvector matrix

### 3. Root Count Rule (`rys_quadrature.py`)

Number of Rys roots needed for a shell quartet:
```python
n_roots = L_total // 2 + 1

# where L_total = l_A + l_B + l_C + l_D
```

| Shell Type | L_total | n_roots |
|------------|---------|---------|
| (ss\|ss) | 0 | 1 |
| (ps\|ss) | 1 | 1 |
| (pp\|ss) | 2 | 2 |
| (pp\|pp) | 4 | 3 |
| (dd\|dd) | 8 | 5 |
| (ff\|ff) | 12 | 7 |

### 4. (ss|ss) ERI Formula (`rys_eri_example.py`)

The primitive (ss|ss) integral in chemists' notation:
```
(ab|cd) = (2*pi^(5/2)) / (p*q*sqrt(p+q)) * exp(-mu*R_AB^2) * exp(-nu*R_CD^2) * F_0(T)
```

where:
```
p = alpha + beta,  mu = alpha*beta/p,  P = (alpha*A + beta*B)/p
q = gamma + delta, nu = gamma*delta/q, Q = (gamma*C + delta*D)/q
rho = p*q/(p+q),   T = rho * |P - Q|^2
```

Key insight: `F_0(T) = (1/2) * sum_i W_i` via Rys quadrature.

### 5. ERI Symmetries (`rys_eri_example.py`)

Eight-fold symmetry (chemists' notation):
```
(ab|cd) = (ba|cd) = (ab|dc) = (ba|dc) = (cd|ab) = (dc|ab) = (cd|ba) = (dc|ba)
```

### 6. Validation Tests (`rys_validation.py`)

Comprehensive tests for numerical accuracy:
- Moment matching: `sum W_i x_i^n = m_n` for n = 0, ..., 2*n_r - 1
- Boys function reproduction: `F_n(T) = (1/2) sum W_i x_i^n`
- Node bounds: 0 < x_i < 1
- Weight positivity: W_i > 0
- Weight sum: `sum W_i = m_0 = 2*F_0(T)`
- Stability for small and large T values

## Dependencies

These scripts require the Boys function module from `appendix_d`:

```python
# Import chain
from boys_function import boys, boys_array  # From appendix_d
```

Ensure `appendix_d/boys_function.py` is available before running.

## Test Systems

All scripts test across parameter ranges:

| Parameter | Values |
|-----------|--------|
| T (Rys argument) | 0, 1e-10, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0 |
| n_roots | 1, 2, 3, 4, 5 |
| Gaussian exponents | 0.1 to 3.0 |
| Center separations | 0 to 10 Bohr |

## Example Usage

### Computing Rys Nodes and Weights

```python
from rys_quadrature import rys_roots_weights, root_count_for_angular_momentum

# Determine root count for (pp|pp) shell quartet
L_total = 1 + 1 + 1 + 1  # = 4
n_roots = root_count_for_angular_momentum(L_total)  # = 3

# Compute nodes and weights
T = 2.5  # Rys argument
nodes, weights = rys_roots_weights(T, n_roots)

# Verify: F_n(T) = (1/2) * sum W_i * x_i^n
F_0 = 0.5 * np.sum(weights)
F_1 = 0.5 * np.sum(weights * nodes)
F_2 = 0.5 * np.sum(weights * nodes**2)
```

### Computing (ss|ss) ERI

```python
import numpy as np
from rys_eri_example import eri_ssss_rys, eri_ssss_direct

# Define primitives
alpha, beta = 1.0, 1.0
gamma, delta = 1.0, 1.0
A = np.array([0.0, 0.0, 0.0])
B = np.array([0.0, 0.0, 0.0])
C = np.array([2.0, 0.0, 0.0])
D = np.array([2.0, 0.0, 0.0])

# Compute via Rys quadrature
eri_rys = eri_ssss_rys(alpha, A, beta, B, gamma, C, delta, D)

# Compare with direct Boys evaluation
eri_direct = eri_ssss_direct(alpha, A, beta, B, gamma, C, delta, D)

print(f"|Rys - Direct| = {abs(eri_rys - eri_direct):.2e}")
```

### Running Validation Suite

```python
from rys_validation import run_all_validations, detailed_moment_analysis

# Run full test suite
success = run_all_validations()

# Detailed analysis for specific case
detailed_moment_analysis(T=2.5, n_roots=3)
```

## References

- Golub, G. H.; Welsch, J. H. "Calculation of Gauss Quadrature Rules", Math. Comp. 23 (1969) 221-230. [DOI: 10.1090/S0025-5718-1969-0245201-2](https://doi.org/10.1090/S0025-5718-1969-0245201-2)
- Dupuis, M.; Rys, J.; King, H. F. "Evaluation of molecular integrals over Gaussian basis functions", J. Chem. Phys. 65 (1976) 111-116. [DOI: 10.1063/1.432811](https://doi.org/10.1063/1.432811)
- Lecture notes: Chapter 4-5 (Rys Quadrature) and Appendix E
