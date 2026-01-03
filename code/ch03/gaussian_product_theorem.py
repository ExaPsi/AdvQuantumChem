#!/usr/bin/env python3
"""
Gaussian Product Theorem: Exploration and Visualization

The Gaussian Product Theorem (GPT) states that the product of two Gaussians
centered at different points is itself a Gaussian centered at a composite point:

    exp(-alpha|r-A|^2) * exp(-beta|r-B|^2) = exp(-p|r-P|^2) * exp(-mu*R_AB^2)

where:
    p = alpha + beta              (total exponent)
    mu = alpha*beta/(alpha+beta)  (reduced exponent)
    P = (alpha*A + beta*B)/p      (composite center)
    R_AB = |A - B|                (distance between centers)

This script explores:
1. GPT parameter relationships
2. How composite center P moves with exponent ratio
3. Verification of the theorem numerically

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 3: One-Electron Integrals and Gaussian Product Theorem
"""

import numpy as np


def gpt_parameters(alpha: float, A: np.ndarray,
                   beta: float, B: np.ndarray) -> dict:
    """
    Compute Gaussian Product Theorem parameters.

    Parameters
    ----------
    alpha : float
        Exponent of first Gaussian
    A : np.ndarray
        Center of first Gaussian (3D)
    beta : float
        Exponent of second Gaussian
    B : np.ndarray
        Center of second Gaussian (3D)

    Returns
    -------
    dict
        GPT parameters: p, mu, P, K, R_AB
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    p = alpha + beta
    mu = alpha * beta / p
    P = (alpha * A + beta * B) / p
    R_AB = np.linalg.norm(A - B)
    K = np.exp(-mu * R_AB**2)  # Prefactor

    return {
        "p": p,
        "mu": mu,
        "P": P,
        "K": K,
        "R_AB": R_AB
    }


def gaussian_3d(r: np.ndarray, alpha: float, center: np.ndarray) -> float:
    """Evaluate 3D Gaussian at point r."""
    diff = r - center
    return np.exp(-alpha * np.dot(diff, diff))


def verify_gpt(alpha: float, A: np.ndarray,
               beta: float, B: np.ndarray,
               test_points: np.ndarray) -> dict:
    """
    Verify GPT by comparing product of Gaussians with composite Gaussian.

    Parameters
    ----------
    alpha, A, beta, B : GPT input parameters
    test_points : np.ndarray
        Array of shape (N, 3) with test points

    Returns
    -------
    dict
        Verification results including max error
    """
    params = gpt_parameters(alpha, A, beta, B)

    errors = []
    for r in test_points:
        # Left-hand side: product of two Gaussians
        lhs = gaussian_3d(r, alpha, A) * gaussian_3d(r, beta, B)

        # Right-hand side: composite Gaussian with prefactor
        rhs = params["K"] * gaussian_3d(r, params["p"], params["P"])

        errors.append(abs(lhs - rhs))

    return {
        "max_error": max(errors),
        "mean_error": np.mean(errors),
        "verified": max(errors) < 1e-14
    }


def main():
    print("=" * 70)
    print("Gaussian Product Theorem: Exploration")
    print("=" * 70)

    # Define two Gaussian centers
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, 2.0])

    print(f"\nGaussian centers:")
    print(f"   A = {A}")
    print(f"   B = {B}")
    print(f"   |A-B| = {np.linalg.norm(A-B):.2f} Bohr")

    # Section 1: Basic GPT parameters
    print("\n" + "-" * 50)
    print("1. GPT PARAMETERS FOR alpha = beta = 1.0")
    print("-" * 50)

    alpha = beta = 1.0
    params = gpt_parameters(alpha, A, beta, B)

    print(f"\n   alpha = {alpha}, beta = {beta}")
    print(f"\n   Total exponent: p = alpha + beta = {params['p']}")
    print(f"   Reduced exponent: mu = alpha*beta/p = {params['mu']}")
    print(f"   Composite center: P = {params['P']}")
    print(f"   Prefactor: K = exp(-mu*R^2) = {params['K']:.6f}")

    print("\n   Note: When alpha = beta, P is the midpoint of A and B")

    # Section 2: Asymmetric exponents
    print("\n" + "-" * 50)
    print("2. COMPOSITE CENTER WITH ASYMMETRIC EXPONENTS")
    print("-" * 50)

    print("\n   How P moves as alpha/beta varies (beta = 1.0 fixed):")
    print("\n   alpha     P_z          Distance from A")
    print("   " + "-" * 40)

    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        params = gpt_parameters(alpha, A, 1.0, B)
        dist_from_A = np.linalg.norm(params["P"] - A)
        print(f"   {alpha:5.1f}     {params['P'][2]:8.4f}     {dist_from_A:.4f}")

    print("""
   Observation:
   - Larger alpha -> P moves toward A (the tighter Gaussian)
   - P = (alpha*A + beta*B)/(alpha+beta) is a weighted average
   - Tight Gaussians (large exponent) "anchor" the composite center
    """)

    # Section 3: Numerical verification
    print("-" * 50)
    print("3. NUMERICAL VERIFICATION OF GPT")
    print("-" * 50)

    # Generate random test points
    np.random.seed(42)
    test_points = np.random.randn(100, 3) * 2  # Points around origin

    for alpha, beta in [(1.0, 1.0), (0.5, 2.0), (0.3, 0.7)]:
        result = verify_gpt(alpha, A, beta, B, test_points)
        print(f"\n   alpha={alpha}, beta={beta}:")
        print(f"      Max error:  {result['max_error']:.2e}")
        print(f"      Mean error: {result['mean_error']:.2e}")
        print(f"      Verified:   {result['verified']}")

    # Section 4: Prefactor decay
    print("\n" + "-" * 50)
    print("4. PREFACTOR DECAY WITH DISTANCE")
    print("-" * 50)

    print("\n   K = exp(-mu * R_AB^2) decay for alpha = beta = 1.0:")
    print("\n   R_AB (Bohr)    mu*R^2      K = exp(-mu*R^2)")
    print("   " + "-" * 45)

    alpha = beta = 1.0
    mu = alpha * beta / (alpha + beta)

    for R in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        B_test = np.array([0.0, 0.0, R])
        params = gpt_parameters(alpha, A, beta, B_test)
        print(f"   {R:6.1f}         {mu * R**2:8.4f}    {params['K']:.6e}")

    print("""
   Observation:
   - K decays exponentially with R^2
   - This is why overlap integrals decay exponentially with distance
   - Basis of integral screening: if K is small, integral is negligible
    """)

    # Section 5: Connection to integrals
    print("-" * 50)
    print("5. CONNECTION TO OVERLAP INTEGRALS")
    print("-" * 50)

    print("""
   The overlap integral between two s-type Gaussians is:

       S_ab = (pi/p)^(3/2) * K
            = (pi/p)^(3/2) * exp(-mu * R_AB^2)

   The GPT provides:
   1. The prefactor K from the distance decay
   2. The composite center P for evaluating more complex integrals
   3. The total exponent p that determines the "width" of the product

   For normalized primitives:
       S_ab = N_a * N_b * (pi/p)^(3/2) * exp(-mu * R_AB^2)

   where N_a = (2*alpha/pi)^(3/4) is the normalization constant.
    """)

    # Section 6: Physical interpretation
    print("-" * 50)
    print("6. PHYSICAL INTERPRETATION")
    print("-" * 50)

    print("""
   The GPT has deep physical meaning:

   1. PRODUCT IS A GAUSSIAN: The product of two Gaussians is still a Gaussian.
      This is why Gaussians are computationally convenient - products remain
      in the same function space.

   2. COMPOSITE CENTER: P represents where the "overlap density" is concentrated.
      Tighter orbitals (larger exponent) pull P toward their center.

   3. TOTAL EXPONENT: p = alpha + beta. The product Gaussian is tighter
      than either parent - you need more basis functions in tight regions.

   4. EXPONENTIAL SCREENING: K = exp(-mu*R^2) shows that distant orbitals
      have exponentially small overlap. This enables screening in large
      molecule calculations.

   5. REDUCED EXPONENT: mu = alpha*beta/p is analogous to reduced mass
      in classical mechanics. It appears in the screening factor.
    """)

    print("\n" + "=" * 70)
    print("GPT Exploration Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
