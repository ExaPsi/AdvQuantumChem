#!/usr/bin/env python3
"""
Boys Function Validation via PySCF/libcint Comparison

This script validates our Boys function implementation by comparing
computed ERIs against PySCF, which uses libcint internally.

The key insight: if our Boys function is correct, then our (ss|ss) ERIs
will match PySCF exactly. Any discrepancy indicates a Boys function error.

Validation chain:
    Our boys(n, T) → Our ERI formula → Compare with PySCF ERI (libcint)

This approach is more pedagogically valuable than direct libcint interfacing
because it validates the complete integral pipeline.

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix D: Boys Functions
"""

import numpy as np
from pyscf import gto
from boys_function import boys


def norm_s(alpha: float) -> float:
    """Normalization constant for s-type Gaussian: (2α/π)^(3/4)."""
    return (2.0 * alpha / np.pi)**0.75


def eri_ssss(alpha: float, A: np.ndarray,
             beta: float, B: np.ndarray,
             gamma: float, C: np.ndarray,
             delta: float, D: np.ndarray) -> float:
    """
    Compute (ss|ss) ERI using our Boys function implementation.

    Formula (chemist's notation):
        (ab|cd) = N_a N_b N_c N_d × (2π^{5/2}) / (pq√(p+q))
                  × exp(-μ_ab R²_AB) × exp(-ν_cd R²_CD) × F_0(T)

    where:
        p = α + β,  q = γ + δ
        μ = αβ/p,   ν = γδ/q
        P = (αA + βB)/p,  Q = (γC + δD)/q
        ρ = pq/(p+q)
        T = ρ |P - Q|²

    This uses our boys(0, T) implementation which we want to validate.
    """
    # Normalization
    N_a, N_b, N_c, N_d = norm_s(alpha), norm_s(beta), norm_s(gamma), norm_s(delta)

    # GPT for bra pair (a, b)
    p = alpha + beta
    mu_ab = alpha * beta / p
    P = (alpha * A + beta * B) / p
    R_AB_sq = np.sum((A - B)**2)

    # GPT for ket pair (c, d)
    q = gamma + delta
    nu_cd = gamma * delta / q
    Q = (gamma * C + delta * D) / q
    R_CD_sq = np.sum((C - D)**2)

    # Combined exponent and Boys argument
    rho = p * q / (p + q)
    PQ = P - Q
    R_PQ_sq = np.sum(PQ**2)
    T = rho * R_PQ_sq

    # ERI formula
    prefactor = (2.0 * np.pi**2.5) / (p * q * np.sqrt(p + q))
    exp_factor = np.exp(-mu_ab * R_AB_sq) * np.exp(-nu_cd * R_CD_sq)
    F0 = boys(0, T)  # Our Boys function!

    return N_a * N_b * N_c * N_d * prefactor * exp_factor * F0


def get_pyscf_eri(alpha: float, A: np.ndarray,
                  beta: float, B: np.ndarray,
                  gamma: float, C: np.ndarray,
                  delta: float, D: np.ndarray) -> float:
    """
    Compute the same ERI using PySCF (which uses libcint internally).

    We construct a minimal molecule with 4 s-type Gaussians and extract
    the corresponding ERI element.
    """
    # Build molecule with 4 ghost atoms, each with a single s-primitive
    mol = gto.M(
        atom=f'''
        X1  {A[0]:.10f}  {A[1]:.10f}  {A[2]:.10f}
        X2  {B[0]:.10f}  {B[1]:.10f}  {B[2]:.10f}
        X3  {C[0]:.10f}  {C[1]:.10f}  {C[2]:.10f}
        X4  {D[0]:.10f}  {D[1]:.10f}  {D[2]:.10f}
        ''',
        basis={
            'X1': gto.basis.parse(f'''
                X1    S
                    {alpha:.10f}    1.0
            '''),
            'X2': gto.basis.parse(f'''
                X2    S
                    {beta:.10f}    1.0
            '''),
            'X3': gto.basis.parse(f'''
                X3    S
                    {gamma:.10f}    1.0
            '''),
            'X4': gto.basis.parse(f'''
                X4    S
                    {delta:.10f}    1.0
            '''),
        },
        unit='Bohr',
        verbose=0
    )

    # Get the (0,1|2,3) ERI element
    eri = mol.intor('int2e')
    return eri[0, 1, 2, 3]


def validate_single_eri(alpha, A, beta, B, gamma, C, delta, D, label=""):
    """Compare our ERI with PySCF's for a single set of parameters."""
    our_eri = eri_ssss(alpha, A, beta, B, gamma, C, delta, D)
    pyscf_eri = get_pyscf_eri(alpha, A, beta, B, gamma, C, delta, D)
    error = abs(our_eri - pyscf_eri)
    rel_error = error / abs(pyscf_eri) if abs(pyscf_eri) > 1e-20 else 0

    return {
        'label': label,
        'our_eri': our_eri,
        'pyscf_eri': pyscf_eri,
        'abs_error': error,
        'rel_error': rel_error
    }


def main():
    """Validate Boys function through ERI comparison with PySCF/libcint."""
    print("=" * 75)
    print("Boys Function Validation via PySCF/libcint ERI Comparison")
    print("=" * 75)
    print()
    print("Strategy: If our boys(n, T) is correct, our ERIs will match PySCF.")
    print("PySCF uses libcint internally for integral evaluation.")
    print()

    # Define test cases with varying T values
    # T = ρ|P-Q|² where ρ = pq/(p+q)
    test_cases = []

    # Case 1: Identical centers (T = 0)
    A = np.array([0.0, 0.0, 0.0])
    test_cases.append({
        'alpha': 1.0, 'A': A,
        'beta': 1.0, 'B': A,
        'gamma': 1.0, 'C': A,
        'delta': 1.0, 'D': A,
        'label': 'All same center (T≈0)'
    })

    # Case 2: Small separation (small T)
    B = np.array([0.1, 0.0, 0.0])
    test_cases.append({
        'alpha': 1.0, 'A': A,
        'beta': 1.0, 'B': B,
        'gamma': 1.0, 'C': A,
        'delta': 1.0, 'D': B,
        'label': 'Small separation (small T)'
    })

    # Case 3: Moderate separation (moderate T)
    B = np.array([1.0, 0.0, 0.0])
    C = np.array([0.5, 0.5, 0.0])
    D = np.array([1.5, 0.5, 0.0])
    test_cases.append({
        'alpha': 0.5, 'A': A,
        'beta': 0.8, 'B': B,
        'gamma': 0.6, 'C': C,
        'delta': 0.7, 'D': D,
        'label': 'Moderate separation (moderate T)'
    })

    # Case 4: Large separation (large T)
    B = np.array([5.0, 0.0, 0.0])
    C = np.array([2.5, 2.5, 0.0])
    D = np.array([7.5, 2.5, 0.0])
    test_cases.append({
        'alpha': 2.0, 'A': A,
        'beta': 2.0, 'B': B,
        'gamma': 2.0, 'C': C,
        'delta': 2.0, 'D': D,
        'label': 'Large separation (large T)'
    })

    # Case 5: Very diffuse functions (small exponents → small T)
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([1.0, 0.0, 0.0])
    C = np.array([0.0, 1.0, 0.0])
    D = np.array([1.0, 1.0, 0.0])
    test_cases.append({
        'alpha': 0.1, 'A': A,
        'beta': 0.1, 'B': B,
        'gamma': 0.1, 'C': C,
        'delta': 0.1, 'D': D,
        'label': 'Diffuse functions (α=0.1)'
    })

    # Case 6: Very tight functions (large exponents)
    test_cases.append({
        'alpha': 10.0, 'A': A,
        'beta': 10.0, 'B': B,
        'gamma': 10.0, 'C': C,
        'delta': 10.0, 'D': D,
        'label': 'Tight functions (α=10)'
    })

    # Case 7: Mixed exponents
    test_cases.append({
        'alpha': 0.2, 'A': np.array([0.0, 0.0, 0.0]),
        'beta': 5.0, 'B': np.array([1.5, 0.0, 0.0]),
        'gamma': 1.0, 'C': np.array([0.0, 1.5, 0.0]),
        'delta': 3.0, 'D': np.array([1.5, 1.5, 0.0]),
        'label': 'Mixed exponents (0.2, 5, 1, 3)'
    })

    # Run all tests
    print("-" * 75)
    print(f"{'Test Case':<35} {'Our ERI':>14} {'PySCF ERI':>14} {'Rel Error':>12}")
    print("-" * 75)

    max_rel_error = 0
    all_passed = True

    for tc in test_cases:
        result = validate_single_eri(
            tc['alpha'], tc['A'],
            tc['beta'], tc['B'],
            tc['gamma'], tc['C'],
            tc['delta'], tc['D'],
            tc['label']
        )

        status = "✓" if result['rel_error'] < 1e-10 else "✗"
        print(f"{result['label']:<35} {result['our_eri']:>14.8e} "
              f"{result['pyscf_eri']:>14.8e} {result['rel_error']:>12.2e} {status}")

        max_rel_error = max(max_rel_error, result['rel_error'])
        if result['rel_error'] >= 1e-10:
            all_passed = False

    print("-" * 75)
    print(f"Maximum relative error: {max_rel_error:.2e}")
    print()

    # Summary
    print("=" * 75)
    print("VALIDATION SUMMARY")
    print("=" * 75)
    if all_passed:
        print("✓ All ERI tests passed (relative error < 1e-10)")
        print()
        print("Conclusion: Our boys(n, T) implementation is consistent with")
        print("libcint's gamma_inc_like function used by PySCF.")
    else:
        print("✗ Some tests failed!")
        print("Review the Boys function implementation.")

    print()
    print("Technical note:")
    print("  - PySCF uses libcint's gamma_inc_like() for Boys function")
    print("  - libcint uses: series for small T, erf + upward for large T")
    print("  - Our implementation uses the same mathematical approach")
    print("=" * 75)

    # Additional: Show what T values were tested
    print()
    print("Boys function arguments (T values) tested:")
    print("-" * 50)
    for tc in test_cases:
        p = tc['alpha'] + tc['beta']
        q = tc['gamma'] + tc['delta']
        rho = p * q / (p + q)
        P = (tc['alpha'] * tc['A'] + tc['beta'] * tc['B']) / p
        Q = (tc['gamma'] * tc['C'] + tc['delta'] * tc['D']) / q
        T = rho * np.sum((P - Q)**2)
        print(f"  {tc['label']:<40} T = {T:.6f}")


if __name__ == "__main__":
    main()
