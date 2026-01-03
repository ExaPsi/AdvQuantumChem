#!/usr/bin/env python3
"""
Gaussian Product Theorem Implementation and Verification

The GPT states that the product of two Gaussians centered at A and B
yields a new Gaussian at composite center P:

exp(-α|r-A|²) · exp(-β|r-B|²) = exp(-p|r-P|²) · exp(-μR²_AB)

where p = α+β, μ = αβ/(α+β), P = (αA + βB)/(α+β)

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix C: Gaussian Integral Formula Sheet
"""

import numpy as np


def gaussian_product_theorem(alpha: float, A: np.ndarray,
                              beta: float, B: np.ndarray) -> tuple:
    """
    Apply the Gaussian Product Theorem to two s-type Gaussians.

    Args:
        alpha, beta: Gaussian exponents
        A, B: Gaussian centers (3-vectors)

    Returns:
        p: Composite exponent (alpha + beta)
        P: Composite center (weighted average)
        mu: Reduced exponent (alpha*beta/(alpha+beta))
        K_AB: Pre-exponential factor exp(-mu*R_AB^2)
    """
    # Composite exponent
    p = alpha + beta

    # Reduced exponent (analogous to reduced mass)
    mu = alpha * beta / (alpha + beta)

    # Composite center (charge-weighted average)
    P = (alpha * A + beta * B) / (alpha + beta)

    # Pre-exponential factor from GPT
    R_AB_sq = np.sum((A - B)**2)
    K_AB = np.exp(-mu * R_AB_sq)

    return p, P, mu, K_AB


def main():
    print("=" * 60)
    print("Gaussian Product Theorem Verification")
    print("=" * 60)

    # Test parameters
    alpha, beta = 0.5, 0.8
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, 1.4])  # 1.4 Bohr ~ 0.74 Angstrom

    p, P, mu, K_AB = gaussian_product_theorem(alpha, A, beta, B)

    print(f"\nInput parameters:")
    print(f"  α = {alpha}, β = {beta}")
    print(f"  A = {A}")
    print(f"  B = {B}")
    print(f"  R_AB = {np.linalg.norm(A - B):.4f} Bohr")

    print(f"\nGPT parameters:")
    print(f"  p = α + β = {p:.4f}")
    print(f"  μ = αβ/(α+β) = {mu:.4f}")
    print(f"  P = (αA + βB)/p = {P}")
    print(f"  K_AB = exp(-μR²) = {K_AB:.6f}")

    # Numerical verification on 1D grid along z-axis
    z = np.linspace(-5, 5, 1000)

    # Left-hand side: product of two Gaussians
    lhs = np.exp(-alpha * z**2) * np.exp(-beta * (z - 1.4)**2)

    # Right-hand side: single Gaussian at P with prefactor
    rhs = K_AB * np.exp(-p * (z - P[2])**2)

    max_diff = np.max(np.abs(lhs - rhs))

    print(f"\nNumerical verification (1D grid):")
    print(f"  max|LHS - RHS| = {max_diff:.2e}")
    print(f"  (Should be ~machine precision, confirming GPT is exact)")


if __name__ == "__main__":
    main()
