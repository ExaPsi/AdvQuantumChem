#!/usr/bin/env python3
"""
Overlap Integral Decay and Screening

Explores how overlap integrals decay with distance, which is the basis
for integral screening in large-scale calculations.

The key result: S_ab ~ exp(-mu * R_AB^2)

where mu = alpha*beta/(alpha+beta) is the reduced exponent.

Supports Exercise 3.2 from Chapter 3.

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 3: One-Electron Integrals and Gaussian Product Theorem
"""

import numpy as np
import math


def overlap_ss_normalized(alpha: float, beta: float, R: float) -> float:
    """
    Normalized s-s overlap integral as function of separation.

    S_ab = N_a * N_b * (pi/p)^(3/2) * exp(-mu * R^2)

    Parameters
    ----------
    alpha, beta : float
        Gaussian exponents (Bohr^-2)
    R : float
        Separation between centers (Bohr)

    Returns
    -------
    float
        Overlap integral value
    """
    p = alpha + beta
    mu = alpha * beta / p

    N_a = (2 * alpha / math.pi) ** 0.75
    N_b = (2 * beta / math.pi) ** 0.75

    return N_a * N_b * (math.pi / p) ** 1.5 * math.exp(-mu * R**2)


def screening_radius(alpha: float, beta: float, threshold: float) -> float:
    """
    Compute distance at which overlap drops below threshold.

    |S_ab| < threshold  when  R > R_screen

    Parameters
    ----------
    alpha, beta : float
        Gaussian exponents
    threshold : float
        Screening threshold (e.g., 1e-8)

    Returns
    -------
    float
        Screening radius in Bohr
    """
    # At R = 0: S_max = N_a * N_b * (pi/p)^1.5
    p = alpha + beta
    mu = alpha * beta / p
    N_a = (2 * alpha / math.pi) ** 0.75
    N_b = (2 * beta / math.pi) ** 0.75
    S_max = N_a * N_b * (math.pi / p) ** 1.5

    # S = S_max * exp(-mu * R^2) < threshold
    # exp(-mu * R^2) < threshold / S_max
    # -mu * R^2 < log(threshold / S_max)
    # R^2 > -log(threshold / S_max) / mu
    # R > sqrt(-log(threshold / S_max) / mu)

    ratio = threshold / S_max
    if ratio >= 1:
        return 0.0  # Already below threshold at R = 0

    return math.sqrt(-math.log(ratio) / mu)


def main():
    print("=" * 70)
    print("Overlap Integral Decay and Screening")
    print("=" * 70)

    # Section 1: Basic decay behavior
    print("\n" + "-" * 50)
    print("1. OVERLAP DECAY WITH DISTANCE")
    print("-" * 50)

    alpha = beta = 0.5  # Bohr^-2
    mu = alpha * beta / (alpha + beta)

    print(f"\n   Parameters: alpha = beta = {alpha} Bohr^-2")
    print(f"   Reduced exponent: mu = {mu:.4f} Bohr^-2")

    distances = [0.5, 1.0, 2.0, 3.0, 5.0]

    print("\n   R (Bohr)    S_ab         log10|S|")
    print("   " + "-" * 40)

    for R in distances:
        S = overlap_ss_normalized(alpha, beta, R)
        log_S = math.log10(abs(S)) if S > 1e-300 else -300
        print(f"   {R:6.1f}      {S:12.6e}   {log_S:8.3f}")

    # Validation point from Exercise 3.2
    R_test = 3.0
    S_test = overlap_ss_normalized(0.5, 0.5, R_test)
    print(f"\n   Validation: At R = {R_test} Bohr, S = {S_test:.3f}")
    print(f"   Expected from Exercise 3.2: ~0.105")

    # Section 2: Screening thresholds
    print("\n" + "-" * 50)
    print("2. SCREENING RADII FOR DIFFERENT THRESHOLDS")
    print("-" * 50)

    print(f"\n   For alpha = beta = 0.5 Bohr^-2:")
    print("\n   Threshold    Screening Radius (Bohr)")
    print("   " + "-" * 40)

    for thresh in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
        R_scr = screening_radius(0.5, 0.5, thresh)
        print(f"   {thresh:.0e}         {R_scr:8.2f}")

    # Section 3: Effect of exponent on screening
    print("\n" + "-" * 50)
    print("3. EFFECT OF EXPONENT ON SCREENING")
    print("-" * 50)

    print("\n   Screening radius for |S| < 10^-8:")
    print("\n   alpha = beta    mu          R_screen (Bohr)")
    print("   " + "-" * 45)

    for exp in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        mu = exp * exp / (2 * exp)
        R_scr = screening_radius(exp, exp, 1e-8)
        print(f"   {exp:8.2f}        {mu:.4f}      {R_scr:8.2f}")

    print("""
   Observation:
   - Larger exponents (tighter orbitals) -> smaller screening radius
   - Diffuse orbitals (small alpha) have long-range overlap
   - This affects the sparsity of integral matrices
    """)

    # Section 4: Sparsity implications
    print("-" * 50)
    print("4. SPARSITY IMPLICATIONS FOR MOLECULES")
    print("-" * 50)

    print("""
   Consider a molecule with N atoms separated by ~2 Bohr on average:

   For STO-3G (typical alpha ~ 0.15-0.3):
   - Screening radius ~ 6-8 Bohr for 10^-8 threshold
   - Atoms > 3-4 bonds away have negligible overlap
   - Matrix becomes sparse in extended systems

   For diffuse functions (alpha ~ 0.01):
   - Screening radius ~ 20+ Bohr
   - Almost all pairs interact
   - Matrix is dense

   Practical implications:
   1. More diffuse basis -> denser integrals -> more expensive
   2. Screening accelerates calculations for large molecules
   3. Schwarz inequality extends screening to 4-center integrals
    """)

    # Section 5: Exponential decay verification
    print("-" * 50)
    print("5. EXPONENTIAL DECAY: log|S| vs R^2")
    print("-" * 50)

    print("\n   S = S_max * exp(-mu * R^2)")
    print("   log|S| = log|S_max| - mu * R^2")
    print("\n   This should be LINEAR in R^2:")

    alpha = beta = 0.5
    mu = alpha * beta / (alpha + beta)

    print(f"\n   R (Bohr)    R^2      log|S|      Predicted (slope = -mu)")
    print("   " + "-" * 55)

    S0 = overlap_ss_normalized(alpha, beta, 0)
    log_S0 = math.log(S0)

    for R in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        S = overlap_ss_normalized(alpha, beta, R)
        log_S = math.log(S) if S > 0 else -999
        predicted = log_S0 - mu * R**2
        print(f"   {R:6.1f}     {R**2:6.1f}    {log_S:8.4f}     {predicted:8.4f}")

    print(f"\n   Slope = -mu = -{mu:.4f}")

    # Section 6: Physical interpretation
    print("\n" + "-" * 50)
    print("6. PHYSICAL INTERPRETATION")
    print("-" * 50)

    print("""
   The exponential decay of overlap has deep physical meaning:

   1. LOCALIZATION: Gaussian orbitals are localized. Two orbitals
      far apart have exponentially small overlap because their
      probability densities don't intersect significantly.

   2. SHORT-RANGE NATURE: Unlike Coulomb interactions (1/r decay),
      overlap is SHORT-RANGE with exp(-mu*R^2) decay. This makes
      large-molecule calculations tractable.

   3. SCREENING PHILOSOPHY: Since tiny integrals don't affect
      results, we can skip computing them. This is "screening."

   4. CONNECTION TO SPARSITY: In a sparse matrix, most elements
      are negligible. For overlaps, this happens naturally due
      to spatial localization.

   5. IMPLICATIONS FOR ALGORITHMS:
      - Linear-scaling methods exploit this decay
      - Coulomb is harder (1/r decays slower)
      - Fast multipole and density fitting help with Coulomb
    """)

    print("\n" + "=" * 70)
    print("Overlap Screening Exploration Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
