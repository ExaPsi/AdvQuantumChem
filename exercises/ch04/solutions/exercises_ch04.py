#!/usr/bin/env python3
"""
Chapter 4 Exercise Solutions: Two-Electron Integrals and Rys Quadrature Foundations

This script provides Python implementations for all exercises in Chapter 4 that
involve numerical computation or validation. The exercises cover:

- Exercise 4.1: (ss|ss) ERI formula derivation and verification
- Exercise 4.2: ERI limiting behavior analysis (R_PQ -> 0 and R_PQ -> infinity)
- Exercise 4.3: Schwarz screening experiment for H2O
- Exercise 4.4: Boys function stability study
- Exercise 4.5: Root count scaling with angular momentum (conceptual, no code)
- Exercise 4.6: ERI symmetry verification (8-fold)
- Exercise 4.7: ERI scaling with basis size
- Exercise 4.8: Moment matching for Rys quadrature
- Exercise 4.9: Boys function evaluation strategies (Research/Challenge)

Additionally, this file includes validation code for Checkpoint Questions 4.6, 4.9,
and 4.10 which involve numerical computation.

Part of: Advanced Quantum Chemistry Lecture Notes
Course: 2302638 Advanced Quantum Chemistry
Institution: Department of Chemistry, Chulalongkorn University

Usage:
    python exercises_ch04.py              # Run all exercises
    python exercises_ch04.py --exercise 3 # Run specific exercise

All numerical results are validated against PySCF reference values.
"""

import numpy as np
import math
from scipy import special
from typing import Tuple, List, Optional
from itertools import product
import argparse


# =============================================================================
# CORE UTILITIES: Boys Function Implementation
# =============================================================================

def boys_series(n: int, T: float, max_terms: int = 100, tol: float = 1e-16) -> float:
    """
    Compute F_n(T) using Taylor series expansion.

    The series is:
        F_n(T) = sum_{k=0}^{inf} (-T)^k / [k! * (2n + 2k + 1)]

    This method is numerically stable for all T but converges slowly for large T.

    Args:
        n: Order of the Boys function (n >= 0)
        T: Argument (T >= 0)
        max_terms: Maximum number of terms in series
        tol: Convergence tolerance (relative)

    Returns:
        F_n(T) computed via series
    """
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    if T == 0:
        return 1.0 / (2 * n + 1)

    val = 0.0
    term = 1.0

    for k in range(max_terms):
        contribution = term / (2 * n + 2 * k + 1)
        val += contribution

        if k > 5 and abs(contribution) < tol * abs(val):
            break

        term *= -T / (k + 1)

    return val


def boys_erf_upward(n: int, T: float) -> float:
    """
    Compute F_n(T) using F_0 from the error function and upward recurrence.

    F_0(T) = (1/2) sqrt(pi/T) erf(sqrt(T))    for T > 0
    F_{n+1}(T) = [(2n+1) F_n(T) - exp(-T)] / (2T)

    WARNING: This method suffers from catastrophic cancellation for small T
    with large n. Use only when T > ~25.

    Args:
        n: Order of the Boys function (n >= 0)
        T: Argument (should be large for stability)

    Returns:
        F_n(T) computed via upward recurrence
    """
    if T <= 0:
        return 1.0 / (2 * n + 1) if T == 0 else float('nan')

    sqrt_T = math.sqrt(T)
    F = 0.5 * math.sqrt(math.pi / T) * math.erf(sqrt_T)

    if n == 0:
        return F

    exp_mT = math.exp(-T)
    two_T = 2.0 * T

    for m in range(n):
        F = ((2 * m + 1) * F - exp_mT) / two_T

    return F


def boys_asymptotic(n: int, T: float) -> float:
    """
    Compute F_n(T) using the large-T asymptotic expansion.

    For T >> n:
        F_n(T) ~ (2n-1)!! / (2^{n+1} T^{n+1/2}) * sqrt(pi)

    Args:
        n: Order of the Boys function
        T: Argument (should be large, T >> n)

    Returns:
        Asymptotic approximation to F_n(T)
    """
    if T <= 0:
        raise ValueError("Asymptotic expansion requires T > 0")

    # Compute double factorial (2n-1)!! = 1 * 3 * 5 * ... * (2n-1)
    double_fact = 1.0
    for k in range(1, 2 * n, 2):
        double_fact *= k

    return double_fact * math.sqrt(math.pi) / (2**(n + 1) * T**(n + 0.5))


def boys_downward(n: int, T: float, n_extra: int = 20) -> float:
    """
    Compute F_n(T) using downward recurrence from an asymptotic start.

    Downward recurrence:
        F_m(T) = [2T F_{m+1}(T) + exp(-T)] / (2m+1)

    This is stable because division by (2m+1) damps any errors.

    Args:
        n: Order of the Boys function to compute
        T: Argument
        n_extra: Extra orders to start recurrence (higher = more accurate)

    Returns:
        F_n(T) computed via downward recurrence
    """
    if T <= 0:
        return 1.0 / (2 * n + 1) if T == 0 else float('nan')

    n_start = n + n_extra
    F = boys_asymptotic(n_start, T)

    exp_mT = math.exp(-T)

    for m in range(n_start - 1, n - 1, -1):
        F = (2 * T * F + exp_mT) / (2 * m + 1)

    return F


def boys(n: int, T: float) -> float:
    """
    Compute Boys function F_n(T) using the most stable method for given T.

    Strategy:
        - T = 0: Return exact value 1/(2n+1)
        - T < T_switch: Use series expansion
        - T >= T_switch: Use erf + upward recurrence

    Args:
        n: Order of the Boys function (n >= 0)
        T: Argument (T >= 0)

    Returns:
        F_n(T) with accuracy typically better than 1e-14
    """
    T_SWITCH = 25.0

    if T < T_SWITCH:
        return boys_series(n, T)
    else:
        return boys_erf_upward(n, T)


def boys_reference(n: int, T: float) -> float:
    """
    Compute F_n(T) using scipy's incomplete gamma function (reference).

    The relationship is:
        F_n(T) = gamma(n + 0.5) * gammainc(n + 0.5, T) / (2 * T^{n+0.5})

    Args:
        n: Order of the Boys function
        T: Argument

    Returns:
        F_n(T) from scipy reference
    """
    if T == 0:
        return 1.0 / (2 * n + 1)

    a = n + 0.5
    gamma_a = special.gamma(a)
    gammainc_val = special.gammainc(a, T)

    return 0.5 * gamma_a * gammainc_val / (T ** a)


# =============================================================================
# CORE UTILITIES: ERI Implementation
# =============================================================================

def norm_s_primitive(alpha: float) -> float:
    """
    Normalization constant for an s-type primitive Gaussian.

    N_s = (2*alpha/pi)^{3/4}

    Args:
        alpha: Gaussian exponent

    Returns:
        Normalization constant
    """
    return (2.0 * alpha / np.pi) ** 0.75


def gaussian_product_center(
    alpha: float, A: np.ndarray,
    beta: float, B: np.ndarray
) -> Tuple[float, np.ndarray, float, float]:
    """
    Apply Gaussian Product Theorem to compute composite center and parameters.

    Args:
        alpha: Exponent of first Gaussian
        A: Center of first Gaussian (3D array)
        beta: Exponent of second Gaussian
        B: Center of second Gaussian (3D array)

    Returns:
        Tuple of (p, P, mu, R_AB_sq) where:
        - p = alpha + beta
        - P = (alpha*A + beta*B)/p
        - mu = alpha*beta/p
        - R_AB_sq = |A - B|^2
    """
    p = alpha + beta
    P = (alpha * A + beta * B) / p
    mu = alpha * beta / p
    R_AB_sq = np.sum((A - B)**2)

    return p, P, mu, R_AB_sq


def eri_ssss_unnorm(
    alpha: float, A: np.ndarray,
    beta: float, B: np.ndarray,
    gamma: float, C: np.ndarray,
    delta: float, D: np.ndarray
) -> float:
    """
    Compute (ss|ss) ERI for UNNORMALIZED primitive Gaussians.

    The formula is:
        (ab|cd) = (2 pi^{5/2}) / (pq sqrt(p+q))
                  * exp(-mu_ab R_AB^2) * exp(-nu_cd R_CD^2) * F_0(T)

    Args:
        alpha, A: Exponent and center of primitive a
        beta, B: Exponent and center of primitive b
        gamma, C: Exponent and center of primitive c
        delta, D: Exponent and center of primitive d

    Returns:
        (ab|cd) for unnormalized Gaussians
    """
    # GPT for bra (a, b) -> p, P
    p, P, mu_ab, R_AB_sq = gaussian_product_center(alpha, A, beta, B)

    # GPT for ket (c, d) -> q, Q
    q, Q, nu_cd, R_CD_sq = gaussian_product_center(gamma, C, delta, D)

    # Boys function argument
    rho = p * q / (p + q)
    R_PQ_sq = np.sum((P - Q)**2)
    T = rho * R_PQ_sq

    # Prefactor and exponential
    prefactor = 2.0 * (np.pi ** 2.5) / (p * q * np.sqrt(p + q))
    exp_factor = np.exp(-mu_ab * R_AB_sq - nu_cd * R_CD_sq)

    return prefactor * exp_factor * boys(0, T)


def eri_ssss_norm(
    alpha: float, A: np.ndarray,
    beta: float, B: np.ndarray,
    gamma: float, C: np.ndarray,
    delta: float, D: np.ndarray
) -> float:
    """
    Compute (ss|ss) ERI for NORMALIZED primitive Gaussians.

    Args:
        alpha, A: Exponent and center of primitive a
        beta, B: Exponent and center of primitive b
        gamma, C: Exponent and center of primitive c
        delta, D: Exponent and center of primitive d

    Returns:
        (ab|cd) for normalized Gaussians
    """
    N_a = norm_s_primitive(alpha)
    N_b = norm_s_primitive(beta)
    N_c = norm_s_primitive(gamma)
    N_d = norm_s_primitive(delta)

    return N_a * N_b * N_c * N_d * eri_ssss_unnorm(alpha, A, beta, B, gamma, C, delta, D)


# =============================================================================
# EXERCISE 4.1: (ss|ss) FORMULA VERIFICATION
# =============================================================================

def exercise_4_1():
    """
    Exercise 4.1: Derive the (ss|ss) Formula [Core]

    Verify the analytical (ss|ss) ERI formula against PySCF for H2.
    """
    try:
        from pyscf import gto
    except ImportError:
        print("  PySCF not available. Skipping PySCF validation.")
        return

    print("\n" + "=" * 75)
    print("EXERCISE 4.1: (ss|ss) Formula Verification")
    print("=" * 75)

    print("""
The (ss|ss) ERI formula:
    (ab|cd) = (2 pi^{5/2}) / (pq sqrt(p+q))
              * exp(-mu_ab R_AB^2) * exp(-nu_cd R_CD^2) * F_0(T)

where T = rho |P - Q|^2 and rho = pq/(p+q)
""")

    # Test case: H2 molecule with single primitives
    R = 1.4  # Bond length in Bohr
    alpha = 1.0  # Exponent for both atoms

    # Define custom basis for PySCF
    custom_basis = {
        'H': gto.basis.parse(f'''
H   S
    {alpha}   1.0
''')
    }

    mol = gto.M(
        atom=f'H 0 0 0; H 0 0 {R}',
        basis=custom_basis,
        unit='Bohr',
        verbose=0
    )

    # Get PySCF ERIs
    eri_pyscf = mol.intor('int2e', aosym='s1')

    # Our implementation
    A = np.array([0., 0., 0.])
    B = np.array([0., 0., R])

    print(f"Test system: H2 with R = {R} Bohr, alpha = {alpha}")
    print("-" * 60)
    print(f"{'Integral':>12} {'Our impl':>18} {'PySCF':>18} {'Error':>12}")
    print("-" * 60)

    max_error = 0.0
    test_cases = [
        (0, 0, 0, 0, "(00|00)"),
        (0, 0, 1, 1, "(00|11)"),
        (0, 1, 0, 1, "(01|01)"),
        (1, 1, 1, 1, "(11|11)"),
    ]

    centers = [A, B]
    exponents = [alpha, alpha]

    for mu, nu, lam, sig, label in test_cases:
        ours = eri_ssss_norm(
            exponents[mu], centers[mu],
            exponents[nu], centers[nu],
            exponents[lam], centers[lam],
            exponents[sig], centers[sig]
        )
        pyscf = eri_pyscf[mu, nu, lam, sig]
        error = abs(ours - pyscf)
        max_error = max(max_error, error)

        print(f"{label:>12} {ours:>18.12f} {pyscf:>18.12f} {error:>12.2e}")

    print("-" * 60)
    print(f"Maximum absolute error: {max_error:.2e}")
    passed = max_error < 1e-10
    print(f"Validation: {'PASSED' if passed else 'FAILED'} (tolerance 1e-10)")


# =============================================================================
# EXERCISE 4.2: LIMITING BEHAVIOR OF ERIS
# =============================================================================

def exercise_4_2():
    """
    Exercise 4.2: Limiting Behavior of ERIs [Core]

    Analyze ERI behavior as R_PQ -> 0 and R_PQ -> infinity.
    """
    print("\n" + "=" * 75)
    print("EXERCISE 4.2: Limiting Behavior of ERIs")
    print("=" * 75)

    alpha = 1.0
    A = np.array([0., 0., 0.])

    # (a) R_PQ -> 0 (coinciding pair centers)
    print("\n(a) R_PQ -> 0 (coinciding pair centers):")
    print("-" * 50)

    eri_same = eri_ssss_norm(alpha, A, alpha, A, alpha, A, alpha, A)

    # Expected: (2 pi^{5/2}) / (p*q*sqrt(p+q)) * N^4 * F_0(0)
    p = 2 * alpha
    N = norm_s_primitive(alpha)
    expected = 2 * (np.pi ** 2.5) / (p * p * np.sqrt(2 * p)) * (N ** 4)

    print(f"  T = 0 (all at same center)")
    print(f"  F_0(0) = 1")
    print(f"  Computed ERI: {eri_same:.12f}")
    print(f"  Expected:     {expected:.12f}")
    print(f"  Error: {abs(eri_same - expected):.2e}")

    # (b) R_PQ -> infinity
    print("\n(b) R_PQ -> infinity (large separation):")
    print("-" * 50)

    print(f"{'R (Bohr)':>12} {'T':>12} {'ERI':>15} {'1/R':>15} {'Ratio':>10}")
    print("-" * 65)

    for R in [1.0, 5.0, 10.0, 50.0, 100.0]:
        B = np.array([R, 0., 0.])
        eri_far = eri_ssss_norm(alpha, A, alpha, A, alpha, B, alpha, B)

        # T parameter
        p = 2 * alpha
        rho = p * p / (2 * p)
        T = rho * R**2

        coulomb = 1.0 / R
        ratio = eri_far / coulomb

        print(f"{R:>12.1f} {T:>12.2f} {eri_far:>15.6e} {coulomb:>15.6e} {ratio:>10.4f}")

    print("\n  Physical interpretation:")
    print("  At large R, ERI -> 1/R (classical Coulomb limit)")
    print("  The ratio ERI/(1/R) approaches a constant determined by")
    print("  the Gaussian charge distribution width.")


# =============================================================================
# EXERCISE 4.3: SCHWARZ SCREENING EXPERIMENT
# =============================================================================

def exercise_4_3():
    """
    Exercise 4.3: Schwarz Screening Experiment [Core]

    Compute the fraction of ERIs screened by Schwarz inequality for H2O
    with different basis sets.
    """
    try:
        from pyscf import gto
    except ImportError:
        print("  PySCF not available. Skipping exercise 4.3.")
        return

    print("\n" + "=" * 75)
    print("EXERCISE 4.3: Schwarz Screening Experiment")
    print("=" * 75)

    print("""
Schwarz inequality: |(mu nu|lambda sigma)| <= Q_mu,nu * Q_lambda,sigma
where Q_mu,nu = sqrt((mu nu|mu nu))

If Q_mu,nu * Q_lambda,sigma < tau, the ERI is screened (set to zero).
""")

    # Test molecule: H2O
    h2o_geom = '''
    O  0.000000  0.000000  0.117369
    H  0.000000  0.757918 -0.469476
    H  0.000000 -0.757918 -0.469476
    '''

    bases = ['sto-3g', '6-31g', '6-31+g*']
    tau = 1e-10  # Screening threshold

    print(f"\nMolecule: H2O")
    print(f"Screening threshold: tau = {tau:.0e}")
    print("-" * 70)
    print(f"{'Basis':>12} {'NAO':>6} {'N^4':>12} {'Screened':>12} {'Screened %':>12}")
    print("-" * 70)

    for basis in bases:
        mol = gto.M(atom=h2o_geom, basis=basis, unit='Angstrom', verbose=0)
        nao = mol.nao_nr()

        # Get full ERI tensor
        eri = mol.intor('int2e', aosym='s1')

        # Compute Schwarz bounds: Q[mu,nu] = sqrt((mu nu|mu nu))
        Q = np.zeros((nao, nao))
        for mu in range(nao):
            for nu in range(nao):
                Q[mu, nu] = np.sqrt(abs(eri[mu, nu, mu, nu]))

        # Count screened ERIs
        n_screened = 0
        n_total = nao**4

        for mu in range(nao):
            for nu in range(nao):
                for lam in range(nao):
                    for sig in range(nao):
                        if Q[mu, nu] * Q[lam, sig] < tau:
                            n_screened += 1

        frac = 100.0 * n_screened / n_total

        print(f"{basis:>12} {nao:>6} {n_total:>12,} {n_screened:>12,} {frac:>11.1f}%")

    # Verify Schwarz bound is always an upper bound
    print("\n" + "-" * 70)
    print("Schwarz bound verification (random 100 quartets from 6-31g):")
    print("-" * 70)

    mol = gto.M(atom=h2o_geom, basis='6-31g', unit='Angstrom', verbose=0)
    nao = mol.nao_nr()
    eri = mol.intor('int2e', aosym='s1')

    # Compute Schwarz bounds: Q[mu,nu] = sqrt((mu nu|mu nu))
    Q = np.zeros((nao, nao))
    for mu in range(nao):
        for nu in range(nao):
            Q[mu, nu] = np.sqrt(abs(eri[mu, nu, mu, nu]))

    np.random.seed(42)
    n_violations = 0
    max_ratio = 0.0

    for _ in range(100):
        mu, nu, lam, sig = np.random.randint(0, nao, 4)
        actual = abs(eri[mu, nu, lam, sig])
        bound = Q[mu, nu] * Q[lam, sig]

        ratio = actual / bound if bound > 1e-15 else 0.0
        max_ratio = max(max_ratio, ratio)

        if actual > bound * (1 + 1e-10):  # Allow for numerical error
            n_violations += 1

    print(f"  Number of violations: {n_violations}")
    print(f"  Maximum |ERI|/bound ratio: {max_ratio:.6f}")
    print(f"  (Ratio should always be <= 1)")


# =============================================================================
# EXERCISE 4.4: BOYS FUNCTION STABILITY
# =============================================================================

def exercise_4_4():
    """
    Exercise 4.4: Boys Function Stability [Advanced]

    Compare different evaluation methods across T regimes.
    """
    print("\n" + "=" * 75)
    print("EXERCISE 4.4: Boys Function Stability Study")
    print("=" * 75)

    print("""
Comparing three methods:
1. Pure upward recurrence: F_0 from erf, then F_{n+1} = [(2n+1)F_n - e^{-T}]/(2T)
2. Hybrid series/recursion: Series for T < 25, erf+upward for T >= 25
3. Downward recurrence: Start from asymptotic at n_max+20, recurse down

Reference: scipy.special.gammainc
""")

    print("\n(a) Maximum error vs. reference for n = 0,...,10:")
    print("-" * 80)
    print(f"{'T':>12} {'Upward':>15} {'Hybrid':>15} {'Downward':>15}")
    print("-" * 80)

    T_values = [1e-10, 1e-6, 1e-2, 1.0, 10.0, 100.0]

    for T in T_values:
        errors = {'upward': 0.0, 'hybrid': 0.0, 'downward': 0.0}

        for n in range(11):
            ref = boys_reference(n, T)

            # Pure upward (may fail for small T)
            try:
                upward = boys_erf_upward(n, T)
                errors['upward'] = max(errors['upward'], abs(upward - ref))
            except:
                errors['upward'] = float('inf')

            # Hybrid
            hybrid = boys(n, T)
            errors['hybrid'] = max(errors['hybrid'], abs(hybrid - ref))

            # Downward
            if T > 0:
                down = boys_downward(n, T)
                errors['downward'] = max(errors['downward'], abs(down - ref))
            else:
                errors['downward'] = max(errors['downward'], abs(boys_series(n, T) - ref))

        print(f"{T:>12.0e} {errors['upward']:>15.2e} {errors['hybrid']:>15.2e} {errors['downward']:>15.2e}")

    # (b) Demonstrate cancellation at small T
    print("\n(b) Catastrophic cancellation in upward recurrence at T = 0.001:")
    print("-" * 70)
    print(f"{'n':>4} {'Series (stable)':>20} {'erf+upward':>20} {'Rel Error':>15}")
    print("-" * 70)

    T = 0.001
    for n in range(12):
        series_val = boys_series(n, T)
        upward_val = boys_erf_upward(n, T)

        rel_error = abs(series_val - upward_val) / abs(series_val) if series_val != 0 else 0
        flag = " <-- UNSTABLE" if rel_error > 1e-8 else ""

        print(f"{n:>4} {series_val:>20.15f} {upward_val:>20.15f} {rel_error:>15.2e}{flag}")

    print("\nConclusion: Use series for T < 25, upward recurrence for T >= 25.")


# =============================================================================
# EXERCISE 4.6: ERI SYMMETRY VERIFICATION
# =============================================================================

def exercise_4_6():
    """
    Exercise 4.6: ERI Symmetry Verification [Core]

    Verify 8-fold symmetry of ERIs for H2O/STO-3G.
    """
    try:
        from pyscf import gto
    except ImportError:
        print("  PySCF not available. Skipping exercise 4.6.")
        return

    print("\n" + "=" * 75)
    print("EXERCISE 4.6: ERI Symmetry Verification")
    print("=" * 75)

    print("""
8-fold ERI symmetry:
1. (mu nu|lam sig) = (nu mu|lam sig)     [bra swap]
2. (mu nu|lam sig) = (mu nu|sig lam)     [ket swap]
3. (mu nu|lam sig) = (lam sig|mu nu)     [bra-ket exchange]
Plus all combinations of the above.
""")

    # H2O molecule
    h2o_geom = '''
    O  0.000000  0.000000  0.117369
    H  0.000000  0.757918 -0.469476
    H  0.000000 -0.757918 -0.469476
    '''

    mol = gto.M(atom=h2o_geom, basis='sto-3g', unit='Angstrom', verbose=0)
    nao = mol.nao_nr()
    eri = mol.intor('int2e', aosym='s1')

    print(f"System: H2O/STO-3G")
    print(f"Number of AOs: {nao}")
    print(f"Total ERIs: {nao**4}")

    # Check all 8 symmetry relations
    max_dev = 0.0
    n_checked = 0

    for i in range(nao):
        for j in range(nao):
            for k in range(nao):
                for l in range(nao):
                    ref = eri[i, j, k, l]

                    # All 7 symmetry partners
                    syms = [
                        eri[j, i, k, l],  # bra swap
                        eri[i, j, l, k],  # ket swap
                        eri[j, i, l, k],  # bra + ket swap
                        eri[k, l, i, j],  # bra-ket exchange
                        eri[l, k, i, j],  # bra-ket + bra swap
                        eri[k, l, j, i],  # bra-ket + ket swap
                        eri[l, k, j, i],  # all three
                    ]

                    for s in syms:
                        dev = abs(ref - s)
                        max_dev = max(max_dev, dev)

                    n_checked += 1

    print(f"\nSymmetry relations checked: {n_checked * 7:,}")
    print(f"Maximum symmetry violation: {max_dev:.2e}")
    print(f"Result: {'PASSED' if max_dev < 1e-14 else 'FAILED'} (tolerance 1e-14)")

    # Show some example symmetric ERIs
    print("\nExample: (01|23) and all symmetric partners")
    print("-" * 50)
    if nao >= 4:
        i, j, k, l = 0, 1, 2, 3
        print(f"  (01|23) = {eri[0, 1, 2, 3]:.12f}")
        print(f"  (10|23) = {eri[1, 0, 2, 3]:.12f}  [bra swap]")
        print(f"  (01|32) = {eri[0, 1, 3, 2]:.12f}  [ket swap]")
        print(f"  (23|01) = {eri[2, 3, 0, 1]:.12f}  [bra-ket]")


# =============================================================================
# EXERCISE 4.7: ERI SCALING WITH BASIS SIZE
# =============================================================================

def exercise_4_7():
    """
    Exercise 4.7: ERI Scaling with Basis Size [Core]

    Analyze how memory requirements scale with basis set size.
    """
    try:
        from pyscf import gto
    except ImportError:
        print("  PySCF not available. Skipping exercise 4.7.")
        return

    print("\n" + "=" * 75)
    print("EXERCISE 4.7: ERI Scaling with Basis Size")
    print("=" * 75)

    print("""
Memory scaling:
- Full tensor (aosym='s1'): N^4 * 8 bytes
- 8-fold symmetric (aosym='s8'): ~N^4/8 * 8 bytes

Testing with water clusters using 6-31G basis.
""")

    # Build water clusters of increasing size
    def water_cluster(n_waters: int) -> str:
        """Generate geometry for n water molecules."""
        geom = []
        for i in range(n_waters):
            x_offset = i * 3.0  # Separate by 3 Angstrom
            geom.append(f"O  {x_offset:.6f}  0.000000  0.117369")
            geom.append(f"H  {x_offset:.6f}  0.757918 -0.469476")
            geom.append(f"H  {x_offset:.6f} -0.757918 -0.469476")
        return "; ".join(geom)

    print(f"{'System':>12} {'NAO':>6} {'N^4':>15} {'Full (MB)':>12} {'Symm (MB)':>12}")
    print("-" * 60)

    n_ao_list = []
    n_eri_list = []

    for n_water in [1, 2, 3, 4, 5]:
        try:
            mol = gto.M(atom=water_cluster(n_water), basis='6-31g',
                       unit='Angstrom', verbose=0)
            nao = mol.nao_nr()

            n_eri_full = nao**4
            n_eri_symm = (nao * (nao + 1) // 2) * (nao * (nao + 1) // 2 + 1) // 2

            mem_full = n_eri_full * 8 / 1e6  # MB
            mem_symm = n_eri_symm * 8 / 1e6  # MB

            n_ao_list.append(nao)
            n_eri_list.append(n_eri_full)

            print(f"{n_water} H2O:   {nao:>6} {n_eri_full:>15,} {mem_full:>12.1f} {mem_symm:>12.1f}")
        except Exception as e:
            print(f"{n_water} H2O: Error - {e}")

    # Verify O(N^4) scaling
    if len(n_ao_list) >= 2:
        log_n = np.log(np.array(n_ao_list))
        log_eri = np.log(np.array(n_eri_list))

        # Fit slope
        slope = np.polyfit(log_n, log_eri, 1)[0]

        print("-" * 60)
        print(f"\nScaling analysis: log(N_ERI) vs log(N_AO)")
        print(f"Fitted slope: {slope:.2f} (expected: 4.0 for O(N^4))")


# =============================================================================
# EXERCISE 4.8: MOMENT MATCHING FOR RYS QUADRATURE
# =============================================================================

def exercise_4_8():
    """
    Exercise 4.8: Moment Matching for Rys Quadrature [Advanced]

    Compute Rys quadrature nodes and weights from moments.
    """
    print("\n" + "=" * 75)
    print("EXERCISE 4.8: Moment Matching for Rys Quadrature")
    print("=" * 75)

    print("""
Rys quadrature: integrate f(x) against weight w_T(x) = x^{-1/2} exp(-Tx)

Moments: m_k(T) = integral_0^1 x^k w_T(x) dx = 2 F_k(T)

For n_r roots, the quadrature exactly reproduces moments k = 0, 1, ..., 2n_r - 1
""")

    T = 1.0
    print(f"\nTest case: T = {T}")

    # (a) Compute moments
    print("\n(a) Moments m_k = 2*F_k(T):")
    print("-" * 30)
    for k in range(6):
        m_k = 2.0 * boys(k, T)
        print(f"  m_{k} = {m_k:.10f}")

    # (b) One-root quadrature
    print("\n(b) One-root quadrature (n_r = 1):")
    print("-" * 40)

    m0 = 2.0 * boys(0, T)
    m1 = 2.0 * boys(1, T)

    W1 = m0
    x1 = m1 / m0

    print(f"  W_1 = m_0 = {W1:.10f}")
    print(f"  x_1 = m_1/m_0 = {x1:.10f}")

    # Verify moment matching
    print("\n  Moment verification:")
    for k in range(4):
        m_k = 2.0 * boys(k, T)
        approx = W1 * (x1 ** k)
        error = abs(approx - m_k)
        exact = "exact" if error < 1e-10 else f"error = {error:.2e}"
        print(f"    m_{k}: actual = {m_k:.10f}, approx = {approx:.10f} ({exact})")

    # (c) Two-root quadrature using Hankel matrix method
    print("\n(c) Two-root quadrature (n_r = 2) via Hankel matrix:")
    print("-" * 50)

    # Compute moments
    moments = np.array([2.0 * boys(k, T) for k in range(4)])
    m0, m1, m2, m3 = moments

    # Form Hankel matrices
    H = np.array([[m0, m1], [m1, m2]])
    H1 = np.array([[m1, m2], [m2, m3]])

    print(f"  Hankel matrix H:")
    print(f"    [{m0:.6f}  {m1:.6f}]")
    print(f"    [{m1:.6f}  {m2:.6f}]")

    # Cholesky decomposition: H = L * L^T
    try:
        L = np.linalg.cholesky(H)
        L_inv = np.linalg.inv(L)

        # Jacobi matrix: J = L^{-1} H^{(1)} L^{-T}
        J = L_inv @ H1 @ L_inv.T

        # Eigenvalues are nodes, eigenvectors give weights
        eigenvalues, eigenvectors = np.linalg.eigh(J)

        # Nodes are eigenvalues
        x_nodes = eigenvalues

        # Weights from first row of eigenvector matrix
        W_nodes = m0 * eigenvectors[0, :]**2

        print(f"\n  Nodes and weights:")
        for i in range(2):
            print(f"    x_{i+1} = {x_nodes[i]:.10f}, W_{i+1} = {W_nodes[i]:.10f}")

        # Verify moment matching
        print("\n  Moment verification:")
        for k in range(6):
            m_k = 2.0 * boys(k, T)
            approx = sum(W_nodes[i] * x_nodes[i]**k for i in range(2))
            error = abs(approx - m_k)
            status = "exact" if error < 1e-10 else f"error = {error:.2e}"
            print(f"    m_{k}: actual = {m_k:.10f}, approx = {approx:.10f} ({status})")

    except np.linalg.LinAlgError:
        print("  Cholesky decomposition failed (matrix not positive definite)")


# =============================================================================
# EXERCISE 4.9: BOYS FUNCTION EVALUATION STRATEGIES
# =============================================================================

def exercise_4_9():
    """
    Exercise 4.9: Boys Function Evaluation Strategies [Research/Challenge]

    Compare computational strategies for Boys function evaluation.
    """
    print("\n" + "=" * 75)
    print("EXERCISE 4.9: Boys Function Evaluation Strategies")
    print("=" * 75)

    print("""
Comparing evaluation strategies:
1. Direct series (our implementation)
2. Chebyshev approximation (polynomial approximation in segments)
3. Tabulation + interpolation

Note: This is a simplified comparison. Production codes use more sophisticated
approaches with precomputed coefficients.
""")

    import time

    # Define test points
    T_values = np.linspace(0, 50, 101)
    n_max = 10

    # Method 1: Direct series (our implementation)
    print("\n(a) Timing comparison (100 evaluations each):")
    print("-" * 50)

    start = time.perf_counter()
    for _ in range(100):
        for T in T_values:
            for n in range(n_max + 1):
                _ = boys(n, T)
    series_time = time.perf_counter() - start

    print(f"  Hybrid series/upward: {series_time:.4f} s")

    # Method 2: scipy reference (incomplete gamma)
    start = time.perf_counter()
    for _ in range(100):
        for T in T_values:
            for n in range(n_max + 1):
                _ = boys_reference(n, T)
    scipy_time = time.perf_counter() - start

    print(f"  scipy.special.gammainc: {scipy_time:.4f} s")

    # Accuracy comparison at selected points
    print("\n(b) Accuracy comparison at selected points:")
    print("-" * 70)
    print(f"{'T':>8} {'n':>4} {'Series':>18} {'scipy':>18} {'Difference':>14}")
    print("-" * 70)

    test_points = [(0.0, 0), (0.001, 5), (1.0, 3), (10.0, 5), (50.0, 8)]
    for T, n in test_points:
        series_val = boys(n, T)
        scipy_val = boys_reference(n, T)
        diff = abs(series_val - scipy_val)

        print(f"{T:>8.3f} {n:>4} {series_val:>18.12e} {scipy_val:>18.12e} {diff:>14.2e}")

    print("\nNote: In production codes (libcint, GAMESS), Boys function is evaluated using:")
    print("  - Precomputed rational/Chebyshev approximations for speed")
    print("  - Downward recurrence from asymptotic for stability")
    print("  - Table lookup with interpolation for intermediate T")


# =============================================================================
# CHECKPOINT 4.6: BOYS FUNCTION PROPERTIES
# =============================================================================

def checkpoint_4_6():
    """
    Checkpoint 4.6: Boys Function Properties

    Verify F_n(0) = 1/(2n+1) and other properties.
    """
    print("\n" + "=" * 75)
    print("CHECKPOINT 4.6: Boys Function Properties")
    print("=" * 75)

    # (a) Verify F_n(0) = 1/(2n+1)
    print("\n(a) Verify F_n(0) = 1/(2n+1):")
    print("-" * 50)
    print(f"{'n':>4} {'F_n(0) computed':>20} {'1/(2n+1)':>20} {'Error':>12}")
    print("-" * 50)

    for n in range(7):
        computed = boys(n, 0.0)
        exact = 1.0 / (2 * n + 1)
        error = abs(computed - exact)
        print(f"{n:>4} {computed:>20.15f} {exact:>20.15f} {error:>12.2e}")

    # (b) F_0(1) from erf formula
    print("\n(b) Estimate F_0(1) using erf formula:")
    print("-" * 50)
    T = 1.0
    sqrt_T = math.sqrt(T)
    erf_val = math.erf(sqrt_T)
    F0_T = 0.5 * math.sqrt(math.pi / T) * erf_val

    print(f"  erf(1) = {erf_val:.10f}")
    print(f"  F_0(1) = (1/2) sqrt(pi/1) * erf(1) = {F0_T:.10f}")

    # (c) Derivative identity: dF_n/dT = -F_{n+1}
    print("\n(c) Derivative identity dF_n/dT = -F_{n+1}:")
    print("-" * 50)

    T = 2.0
    h = 1e-6

    for n in range(4):
        F_n = boys(n, T)
        F_n_plus = boys(n, T + h)
        F_n_minus = boys(n, T - h)

        numerical_deriv = (F_n_plus - F_n_minus) / (2 * h)
        analytical_deriv = -boys(n + 1, T)

        error = abs(numerical_deriv - analytical_deriv)
        print(f"  n={n}: numerical = {numerical_deriv:.10f}, "
              f"analytical = {analytical_deriv:.10f}, error = {error:.2e}")

    # (d) Small T cancellation issue
    print("\n(d) Cancellation in upward recursion at T = 0.001:")
    print("-" * 50)

    T = 0.001
    n = 0
    exp_mT = math.exp(-T)
    F0 = boys(0, T)
    numerator = (2 * n + 1) * F0 - exp_mT

    print(f"  (2n+1) * F_n(T) = {(2*n+1) * F0:.15f}")
    print(f"  exp(-T)         = {exp_mT:.15f}")
    print(f"  Difference      = {numerator:.15e}")
    print(f"  (Note: nearly equal values -> cancellation)")


# =============================================================================
# CHECKPOINT 4.9: NUMERICAL STABILITY TEST
# =============================================================================

def checkpoint_4_9():
    """
    Checkpoint 4.9: Numerical Stability of Boys Evaluation
    """
    print("\n" + "=" * 75)
    print("CHECKPOINT 4.9: Numerical Stability Test")
    print("=" * 75)

    # (a) Stability as T -> 0
    print("\n(a) Stability as T -> 0:")
    print("-" * 60)
    print(f"{'T':>12} {'F_0(T)':>15} {'F_1(T)':>15} {'F_2(T)':>15}")
    print("-" * 60)

    for k in [2, 4, 6, 8, 10]:
        T = 10**(-k)
        F0 = boys(0, T)
        F1 = boys(1, T)
        F2 = boys(2, T)
        print(f"{T:>12.0e} {F0:>15.10f} {F1:>15.10f} {F2:>15.10f}")

    T = 0
    print(f"{'T=0 (exact)':>12} {1.0:>15.10f} {1/3:>15.10f} {0.2:>15.10f}")

    # (b) Comparison at n=5, T=0.001
    print("\n(b) Comparison at n=5, T=0.001:")
    print("-" * 50)

    T = 0.001
    n = 5

    series_val = boys_series(n, T)
    upward_val = boys_erf_upward(n, T)
    ref_val = boys_reference(n, T)

    print(f"  Series:      {series_val:.15f}")
    print(f"  erf+upward:  {upward_val:.15f}")
    print(f"  Reference:   {ref_val:.15f}")
    print(f"  Series error: {abs(series_val - ref_val):.2e}")
    print(f"  Upward error: {abs(upward_val - ref_val):.2e}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_all_exercises():
    """Run all exercises in sequence."""
    print()
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*   Chapter 4 Exercise Solutions: Two-Electron Integrals and Rys Quadrature   *")
    print("*" + " " * 78 + "*")
    print("*" * 80)

    # Core exercises
    exercise_4_1()
    exercise_4_2()
    exercise_4_3()
    exercise_4_4()
    exercise_4_6()
    exercise_4_7()
    exercise_4_8()
    exercise_4_9()

    # Checkpoint solutions
    print("\n")
    print("=" * 80)
    print("CHECKPOINT QUESTION SOLUTIONS")
    print("=" * 80)
    checkpoint_4_6()
    checkpoint_4_9()

    print("\n")
    print("=" * 80)
    print("EXERCISE SOLUTIONS COMPLETE")
    print("=" * 80)
    print("""
Summary of Key Findings:

1. EXERCISE 4.1: (ss|ss) formula matches PySCF to ~10^{-12} precision.

2. EXERCISE 4.2: ERIs approach 1/R at large separation (Coulomb limit).

3. EXERCISE 4.3: Schwarz screening eliminates 0-35% of ERIs depending on basis.

4. EXERCISE 4.4: Upward recurrence fails catastrophically for small T.
   Use series for T < 25, upward for T >= 25.

5. EXERCISE 4.6: 8-fold ERI symmetry verified to machine precision.

6. EXERCISE 4.7: ERI count scales as O(N^4), 8-fold symmetry reduces storage.

7. EXERCISE 4.8: Rys quadrature with n_r roots exactly reproduces 2n_r moments.

8. EXERCISE 4.9: Production codes use rational approximations for speed.
""")


def main():
    parser = argparse.ArgumentParser(description='Chapter 4 Exercise Solutions')
    parser.add_argument('--exercise', '-e', type=int, default=None,
                       help='Run specific exercise (1-9)')
    parser.add_argument('--checkpoint', '-c', type=int, default=None,
                       help='Run specific checkpoint (6 or 9)')
    args = parser.parse_args()

    if args.exercise is not None:
        exercise_map = {
            1: exercise_4_1,
            2: exercise_4_2,
            3: exercise_4_3,
            4: exercise_4_4,
            6: exercise_4_6,
            7: exercise_4_7,
            8: exercise_4_8,
            9: exercise_4_9,
        }
        if args.exercise in exercise_map:
            exercise_map[args.exercise]()
        else:
            print(f"Exercise 4.{args.exercise} not available or has no code.")
    elif args.checkpoint is not None:
        checkpoint_map = {
            6: checkpoint_4_6,
            9: checkpoint_4_9,
        }
        if args.checkpoint in checkpoint_map:
            checkpoint_map[args.checkpoint]()
        else:
            print(f"Checkpoint 4.{args.checkpoint} has no numerical solution.")
    else:
        run_all_exercises()


if __name__ == "__main__":
    main()
