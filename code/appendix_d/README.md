# Appendix D: Boys Function Reference - Companion Code

This directory contains Python scripts implementing and validating the Boys function, complementing Appendix D of the Advanced Quantum Chemistry lecture notes.

## Scripts Overview

| Script | Description |
|--------|-------------|
| `boys_function.py` | Core implementation with stable evaluation strategies |
| `boys_stability.py` | Numerical stability analysis and comparison of methods |
| `boys_validation.py` | Validation against SciPy's incomplete gamma function |
| `boys_vs_pyscf.py` | Validation via ERI comparison with PySCF/libcint |

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
cd /home/vyv/Documents/2302638/LectureNote/code/appendix_d

# Core implementation with demonstrations
python boys_function.py

# Numerical stability analysis
python boys_stability.py

# Validation against SciPy
python boys_validation.py

# Validation via PySCF ERI comparison
python boys_vs_pyscf.py
```

## Key Concepts Demonstrated

### 1. Boys Function Definition (`boys_function.py`)

The Boys function is fundamental to all integrals involving 1/r operators:

```
F_n(T) = integral from 0 to 1 of t^(2n) * exp(-T*t^2) dt
```

Key properties:
- `F_n(0) = 1/(2n+1)`
- `F_0(T) = (1/2) * sqrt(pi/T) * erf(sqrt(T))` for T > 0
- Upward recurrence: `F_{n+1}(T) = [(2n+1)*F_n(T) - exp(-T)] / (2T)`
- Downward recurrence: `F_n(T) = [2T*F_{n+1}(T) + exp(-T)] / (2n+1)`

### 2. Stable Evaluation Strategy (`boys_function.py`)

The implementation uses a hybrid approach based on the value of T:

| T Range | Method | Reason |
|---------|--------|--------|
| T = 0 | Exact: `1/(2n+1)` | No computation needed |
| T < 1e-10 | Taylor series | Avoids cancellation in recurrence |
| T > 35 + 5*n | Asymptotic + downward | Stable for large arguments |
| Intermediate | erf + upward recurrence | Fast and accurate |

### 3. Numerical Stability Analysis (`boys_stability.py`)

Demonstrates why different strategies are needed:

**Catastrophic Cancellation in Upward Recurrence:**
```python
# For small T, upward recurrence computes:
#   numerator = (2n+1)*F_n(T) - exp(-T)
# When T -> 0:
#   (2n+1)*F_n(0) = 1  and  exp(0) = 1
# Result: 1 - 1 = 0 with severe digit loss!
```

**Stability of Downward Recurrence:**
```python
# Downward always ADDS positive quantities:
#   F_n = [2T*F_{n+1} + exp(-T)] / (2n+1)
# No cancellation, unconditionally stable.
```

### 4. Validation Methods

**SciPy Reference (`boys_validation.py`):**

The Boys function relates to the incomplete gamma function:
```
F_n(T) = (1/2) * T^{-(n+0.5)} * Gamma(n+0.5) * gammainc(n+0.5, T)
```

Tests include:
- Special case `F_n(0) = 1/(2n+1)`
- Comparison with `scipy.special.gammainc`
- Recurrence consistency
- Derivative identity: `dF_n/dT = -F_{n+1}`
- Asymptotic behavior

**PySCF/libcint Validation (`boys_vs_pyscf.py`):**

Validates through ERI computation:
```
Our boys(n, T) --> Our ERI formula --> Compare with PySCF ERI (libcint)
```

If our Boys function is correct, our (ss|ss) ERIs will match PySCF exactly.

## Usage Examples

### Basic Boys Function Evaluation

```python
from boys_function import boys, boys_array

# Single value
F_0 = boys(0, 1.0)  # F_0(1.0) = 0.746824...
F_2 = boys(2, 0.5)  # F_2(0.5)

# Array of values F_0 through F_n
F_all = boys_array(5, 2.0)  # Returns [F_0(2), F_1(2), ..., F_5(2)]
```

### Verifying Special Cases

```python
from boys_function import boys

# F_n(0) = 1/(2n+1)
for n in range(5):
    assert abs(boys(n, 0) - 1/(2*n+1)) < 1e-14
```

### Computing (ss|ss) ERI

```python
import numpy as np
from boys_function import boys

def eri_ssss(alpha, A, beta, B, gamma, C, delta, D):
    """Compute (ss|ss) ERI using Boys function."""
    # GPT for bra pair
    p = alpha + beta
    mu_ab = alpha * beta / p
    P = (alpha * A + beta * B) / p
    R_AB_sq = np.sum((A - B)**2)

    # GPT for ket pair
    q = gamma + delta
    nu_cd = gamma * delta / q
    Q = (gamma * C + delta * D) / q
    R_CD_sq = np.sum((C - D)**2)

    # Boys argument
    rho = p * q / (p + q)
    T = rho * np.sum((P - Q)**2)

    # ERI formula
    prefactor = (2 * np.pi**2.5) / (p * q * np.sqrt(p + q))
    exp_factor = np.exp(-mu_ab * R_AB_sq) * np.exp(-nu_cd * R_CD_sq)
    F0 = boys(0, T)

    # Apply normalization constants N_a, N_b, N_c, N_d
    N = lambda a: (2*a/np.pi)**0.75
    return N(alpha) * N(beta) * N(gamma) * N(delta) * prefactor * exp_factor * F0
```

## Test Systems

All scripts test across various regimes:

| Regime | T Range | Challenge |
|--------|---------|-----------|
| T = 0 | Exact | Limiting case |
| Very small T | 1e-12 to 1e-8 | Cancellation in upward recurrence |
| Small T | 1e-6 to 0.1 | Series convergence |
| Moderate T | 0.1 to 10 | Standard evaluation |
| Large T | 50 to 200 | Asymptotic regime |

## Mathematical Background

### Taylor Series (Small T)

```
F_n(T) = sum_{k=0}^inf (-T)^k / [k! * (2n+2k+1)]
```

Converges quickly for small T and avoids cancellation.

### Asymptotic Expansion (Large T)

```
F_n(T) -> (2n-1)!! / [2^{n+1} * T^{n+1/2}] * sqrt(pi)
```

Provides starting point for stable downward recurrence.

### Connection to Incomplete Gamma

```
F_n(T) = (1/2) * T^{-(n+0.5)} * Gamma(n+0.5) * P(n+0.5, T)
```

where P(a,x) is the regularized lower incomplete gamma function.

## References

### Textbooks

- Helgaker, T., Jorgensen, P., & Olsen, J. (2000). *Molecular Electronic-Structure Theory*, Chapter 9. Wiley.
  [DOI: 10.1002/9781119019572](https://doi.org/10.1002/9781119019572)

### Software

- Sun, Q. (2015). Libcint: An efficient general integral library for Gaussian basis functions. *J. Comput. Chem.*, 36, 1664-1671.
  [DOI: 10.1002/jcc.23981](https://doi.org/10.1002/jcc.23981)
  - Source reference: `fmt.c` (Boys function implementation)

- Sun, Q., et al. (2020). Recent developments in the PySCF program package. *J. Chem. Phys.*, 153, 024109.
  [DOI: 10.1063/5.0006074](https://doi.org/10.1063/5.0006074)

### Course Materials

- Lecture notes: Chapter 4 (Boys function introduction), Appendix D
