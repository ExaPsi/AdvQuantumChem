#!/usr/bin/env python3
"""
Lab 2A Solution: Eigenvalues of S and Conditioning vs Basis Set

This script explores how basis set choice affects the overlap matrix eigenvalue
spectrum and conditioning. We demonstrate the relationship between diffuse
functions and numerical near-linear dependence.

Learning objectives:
1. Compute and interpret the eigenvalue spectrum of the overlap matrix S
2. Understand condition number kappa(S) = s_max / s_min
3. Observe how diffuse functions decrease smallest eigenvalues
4. Connect near-linear dependence to numerical stability concerns

Test molecule: H2O (water)

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 2: Gaussian Basis Sets and Orthonormalization
"""

import numpy as np
from pyscf import gto

# =============================================================================
# Section 1: Core Functions for Overlap Matrix Analysis
# =============================================================================

def compute_overlap_spectrum(mol: gto.Mole) -> tuple[np.ndarray, float, float]:
    """
    Compute the eigenvalue spectrum of the AO overlap matrix.

    Args:
        mol: PySCF Mole object with defined geometry and basis

    Returns:
        eigenvalues: Sorted eigenvalues in descending order
        kappa: Condition number (s_max / s_min)
        kappa_eff: Effective condition number (ignoring eigenvalues < 1e-14)
    """
    # Compute overlap matrix S_μν = <χ_μ|χ_ν>
    S = mol.intor("int1e_ovlp")

    # S is symmetric positive (semi-)definite, so use eigvalsh for efficiency
    eigenvalues = np.linalg.eigvalsh(S)

    # Sort in descending order for clarity
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]

    # Standard condition number
    s_max = eigenvalues_sorted[0]
    s_min = eigenvalues_sorted[-1]
    kappa = s_max / s_min if s_min > 0 else np.inf

    # Effective condition number: ignore numerically zero eigenvalues
    threshold = 1e-14  # Near machine epsilon
    significant_eigs = eigenvalues_sorted[eigenvalues_sorted > threshold]
    if len(significant_eigs) > 0:
        kappa_eff = significant_eigs[0] / significant_eigs[-1]
    else:
        kappa_eff = np.inf

    return eigenvalues_sorted, kappa, kappa_eff


def analyze_basis_conditioning(atom: str, basis_sets: list[str],
                                unit: str = "Angstrom") -> dict:
    """
    Compare overlap matrix conditioning across multiple basis sets.

    Args:
        atom: Atom specification string for PySCF
        basis_sets: List of basis set names to compare
        unit: Unit for coordinates (default Angstrom)

    Returns:
        Dictionary with analysis results for each basis set
    """
    results = {}

    for basis in basis_sets:
        mol = gto.M(
            atom=atom,
            basis=basis,
            unit=unit,
            verbose=0
        )

        eigs, kappa, kappa_eff = compute_overlap_spectrum(mol)

        results[basis] = {
            "nao": mol.nao_nr(),
            "eigenvalues": eigs,
            "s_max": eigs[0],
            "s_min": eigs[-1],
            "kappa": kappa,
            "kappa_eff": kappa_eff,
            "n_small": np.sum(eigs < 1e-6),  # Count nearly dependent functions
        }

    return results


def print_conditioning_report(results: dict) -> None:
    """Print formatted report of conditioning analysis."""
    print("=" * 75)
    print("Overlap Matrix Conditioning Analysis")
    print("=" * 75)
    print()
    print(f"{'Basis':<16} {'NAO':>5} {'s_min':>12} {'s_max':>10} "
          f"{'kappa':>12} {'kappa_eff':>12}")
    print("-" * 75)

    for basis, data in results.items():
        print(f"{basis:<16} {data['nao']:>5} {data['s_min']:>12.3e} "
              f"{data['s_max']:>10.3f} {data['kappa']:>12.3e} "
              f"{data['kappa_eff']:>12.3e}")

    print("-" * 75)
    print()


def print_eigenvalue_details(results: dict, n_show: int = 5) -> None:
    """Print detailed eigenvalue information."""
    print("Eigenvalue Spectra (smallest and largest)")
    print("=" * 75)

    for basis, data in results.items():
        eigs = data["eigenvalues"]
        n_total = len(eigs)

        print(f"\n{basis} ({data['nao']} AOs):")
        print(f"  Largest {n_show}:  ", end="")
        print("  ".join([f"{e:.4f}" for e in eigs[:n_show]]))
        print(f"  Smallest {n_show}: ", end="")
        print("  ".join([f"{e:.2e}" for e in eigs[-n_show:]]))

        # Eigenvalue spread statistics
        log_spread = np.log10(eigs[0]) - np.log10(max(eigs[-1], 1e-16))
        print(f"  Log10 spread: {log_spread:.1f} orders of magnitude")


# =============================================================================
# Section 2: Physical Interpretation Functions
# =============================================================================

def explain_eigenvalue_meaning() -> None:
    """Explain the physical meaning of overlap eigenvalues."""
    explanation = """
Physical Interpretation of Overlap Eigenvalues
===============================================

The overlap matrix S has eigenvalues in [0, N] where N is the number of AOs.
Each eigenvalue measures the "effective independence" of a linear combination
of basis functions:

- s_i near 1: The corresponding eigenvector represents a well-defined,
  independent direction in function space.

- s_i near 0: The corresponding eigenvector represents a nearly redundant
  direction - a linear combination that integrates to nearly zero norm.

- s_i = 0 exactly: True linear dependence (rare for finite Gaussian bases).

WHY DIFFUSE FUNCTIONS CAUSE TROUBLE:
------------------------------------
Diffuse functions (small exponents alpha) extend far from the nucleus.
When multiple diffuse functions overlap in space, they become nearly
linearly dependent, causing small eigenvalues of S.

NUMERICAL CONSEQUENCES:
-----------------------
- Condition number kappa(S) = s_max / s_min measures sensitivity to
  rounding errors.
- If kappa(S) ~ 10^16 (near 1/epsilon), we lose all significant digits
  when solving systems involving S.
- The orthogonalizer X = S^(-1/2) amplifies eigenvalues by s_i^(-1/2),
  which becomes huge for small s_i.

SOLUTION: Eigenvalue thresholding (Chapter 2, Algorithm 2.2)
"""
    print(explanation)


# =============================================================================
# Section 3: Main Demonstration
# =============================================================================

def main():
    """Run the complete Lab 2A demonstration."""

    print()
    print("*" * 75)
    print("*" + " " * 73 + "*")
    print("*   Lab 2A: Eigenvalues of S and Conditioning vs Basis Set" + " " * 15 + "*")
    print("*" + " " * 73 + "*")
    print("*" * 75)
    print()

    # Water molecule geometry (experimental)
    # O-H bond length: 0.9572 Angstrom
    # H-O-H angle: 104.52 degrees
    h2o_geometry = """
    O    0.000000    0.000000    0.117369
    H    0.756950    0.000000   -0.469476
    H   -0.756950    0.000000   -0.469476
    """

    print("Test Molecule: H2O (Water)")
    print("-" * 40)
    print("Geometry (Angstrom):")
    print(h2o_geometry)

    # Basis sets to compare:
    # 1. STO-3G: Minimal basis (no diffuse functions)
    # 2. cc-pVDZ: Double-zeta with polarization (no diffuse)
    # 3. aug-cc-pVDZ: Same as cc-pVDZ plus diffuse functions
    # 4. aug-cc-pVTZ: Triple-zeta with augmentation (more functions)

    basis_sets = [
        "sto-3g",        # Minimal: 7 AOs for H2O
        "cc-pVDZ",       # Double-zeta: 24 AOs
        "aug-cc-pVDZ",   # Augmented DZ: 41 AOs
        "aug-cc-pVTZ",   # Augmented TZ: 92 AOs
    ]

    print("\nBasis Sets Being Compared:")
    print("-" * 40)
    print("1. STO-3G:       Minimal basis (pedagogical)")
    print("2. cc-pVDZ:      Correlation-consistent double-zeta")
    print("3. aug-cc-pVDZ:  + diffuse functions (for anions, properties)")
    print("4. aug-cc-pVTZ:  + more functions (higher angular momentum)")
    print()

    # Run analysis
    results = analyze_basis_conditioning(h2o_geometry, basis_sets)

    # Print summary report
    print_conditioning_report(results)

    # Print detailed eigenvalue information
    print_eigenvalue_details(results, n_show=5)

    # ==========================================================================
    # Section 4: Validation Against PySCF
    # ==========================================================================

    print()
    print("=" * 75)
    print("Validation: Comparing Our Eigenvalues with NumPy Consistency")
    print("=" * 75)

    # Take one basis set and verify properties
    test_basis = "aug-cc-pVDZ"
    mol = gto.M(atom=h2o_geometry, basis=test_basis, unit="Angstrom", verbose=0)
    S = mol.intor("int1e_ovlp")

    # Verify S is symmetric
    sym_error = np.linalg.norm(S - S.T)
    print(f"\nBasis: {test_basis}")
    print(f"  ||S - S^T|| = {sym_error:.2e} (should be ~0, verifying symmetry)")

    # Verify eigenvalue decomposition reconstructs S
    eigenvalues, U = np.linalg.eigh(S)
    S_reconstructed = U @ np.diag(eigenvalues) @ U.T
    recon_error = np.linalg.norm(S - S_reconstructed)
    print(f"  ||S - U diag(s) U^T|| = {recon_error:.2e} (eigendecomposition)")

    # Verify all eigenvalues are positive (S is positive definite)
    print(f"  All eigenvalues > 0: {np.all(eigenvalues > 0)}")
    print(f"  Min eigenvalue: {eigenvalues.min():.6e}")

    # Sum of eigenvalues = Trace(S) = number of AOs (for normalized basis)
    trace_S = np.trace(S)
    sum_eigs = np.sum(eigenvalues)
    print(f"  Tr(S) = {trace_S:.6f}, Sum(eigenvalues) = {sum_eigs:.6f}")

    # ==========================================================================
    # Section 5: What You Should Observe
    # ==========================================================================

    print()
    print("=" * 75)
    print("What You Should Observe")
    print("=" * 75)

    observations = """
1. BASIS SET SIZE PROGRESSION:
   STO-3G (7) < cc-pVDZ (24) < aug-cc-pVDZ (41) < aug-cc-pVTZ (92)
   More functions = more flexibility but also more potential overlap.

2. CONDITION NUMBER TREND:
   kappa(STO-3G) ~ 10^1 (excellent)
   kappa(cc-pVDZ) ~ 10^2 to 10^3 (good)
   kappa(aug-cc-pVDZ) ~ 10^5 to 10^7 (diffuse functions!)
   kappa(aug-cc-pVTZ) ~ 10^7 to 10^9 (even more diffuse overlap)

3. SMALLEST EIGENVALUE BEHAVIOR:
   - STO-3G: s_min ~ 0.1 to 0.5 (all functions well separated)
   - cc-pVDZ: s_min ~ 0.01 to 0.1 (some overlap, still safe)
   - aug-cc-*: s_min ~ 10^(-6) to 10^(-8) (near-linear dependence!)

4. PHYSICAL ORIGIN OF SMALL EIGENVALUES:
   Diffuse functions on adjacent atoms overlap significantly.
   For example, diffuse s-functions on O and H become nearly
   linearly dependent when r(O-H) is small relative to their extent.

5. NUMERICAL IMPLICATIONS:
   - kappa > 10^8: Need eigenvalue thresholding for stability
   - kappa > 10^12: Severe numerical problems expected
   - kappa ~ 10^16: Near total loss of precision
"""
    print(observations)

    # Physical explanation
    explain_eigenvalue_meaning()

    print()
    print("=" * 75)
    print("Lab 2A Complete")
    print("=" * 75)
    print()


if __name__ == "__main__":
    main()
