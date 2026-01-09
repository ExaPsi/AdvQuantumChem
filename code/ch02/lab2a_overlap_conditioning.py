#!/usr/bin/env python3
"""
Lab 2A: Eigenvalues of S and Conditioning vs Basis Set

This script accompanies Chapter 2 of the Advanced Quantum Chemistry lecture notes.
It demonstrates how to:
  1. Compute overlap matrix eigenvalue spectra for different basis sets
  2. Observe how diffuse functions affect conditioning
  3. Calculate effective condition numbers
  4. Understand the relationship between basis set size and linear dependence

Key concepts:
  - The overlap matrix S is positive definite for linearly independent basis functions
  - Eigenvalues of S measure the "size" of the basis in different directions
  - Small eigenvalues indicate near-linear dependence (ill-conditioning)
  - Diffuse functions (aug-) lead to larger overlap and smaller eigenvalues
  - Condition number kappa(S) = lambda_max / lambda_min measures numerical stability

Physical insight: Why is S != I (the identity)?
  The overlap matrix S_uv = <u|v> = integral of u*(r) v(r) d^3r differs from I because:
  1. Atom-centered: Each basis function is centered on an atom, not on a grid.
     Functions on the same atom overlap significantly.
  2. Extended: Gaussians extend to infinity (though decay rapidly).
     Functions on neighboring atoms can overlap.
  3. Redundancy: Large off-diagonal elements mean the basis is not efficiently
     spanning the function space - one function can be (almost) expressed as
     a linear combination of others.

Theoretical Background:
  - Section 2.5: The Overlap Matrix as a Metric
  - Section 2.10: Hands-on Python (Lab 2A)

Usage:
    python lab2a_overlap_conditioning.py
"""
import numpy as np
from pyscf import gto


def overlap_spectrum(atom: str, basis: str, unit: str = "Angstrom") -> tuple:
    """
    Compute eigenvalues of the overlap matrix S and the condition number.

    The overlap matrix S_uv = <u|v> measures how basis functions overlap.
    Its eigenvalues tell us about linear independence:
      - All eigenvalues positive: linearly independent basis
      - Small eigenvalues: near-linear dependence (problematic for numerics)

    Parameters
    ----------
    atom : str
        Atomic coordinates in format "Element x y z; Element x y z; ..."
    basis : str
        Basis set name (e.g., "sto-3g", "cc-pvdz", "aug-cc-pvdz")
    unit : str
        Unit for coordinates ("Angstrom" or "Bohr"). Default is "Angstrom".

    Returns
    -------
    nao : int
        Number of atomic orbitals (basis functions).
    eigenvalues : ndarray
        Eigenvalues of S sorted in descending order.
    condition_number : float
        Effective condition number (max eigenvalue / min eigenvalue above threshold).

    Notes
    -----
    The condition number kappa(S) determines how numerical errors are amplified
    when solving the generalized eigenvalue problem FC = SCe.
    """
    # Build the molecule
    mol = gto.M(atom=atom, basis=basis, unit=unit, verbose=0)

    # Compute overlap matrix
    S = mol.intor("int1e_ovlp")

    # Compute eigenvalues (eigvalsh for symmetric matrices)
    eigenvalues = np.linalg.eigvalsh(S)

    # Sort in descending order (largest first)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]

    # Compute effective condition number, ignoring eigenvalues below machine precision
    # This avoids division by near-zero eigenvalues that may be numerical noise
    tiny = 1e-14
    eigenvalues_kept = eigenvalues_sorted[eigenvalues_sorted > tiny]

    # Guard against edge case where all eigenvalues are below threshold
    # (This should never happen for a proper basis set, but defensive coding is good)
    if len(eigenvalues_kept) == 0:
        condition_number = np.inf
    else:
        condition_number = eigenvalues_kept[0] / eigenvalues_kept[-1]

    return mol.nao_nr(), eigenvalues_sorted, condition_number


def compare_basis_sets(atom: str, basis_list: list) -> dict:
    """
    Compare overlap matrix conditioning across multiple basis sets.

    This function helps visualize how basis set choice affects the numerical
    stability of quantum chemistry calculations.

    Parameters
    ----------
    atom : str
        Atomic coordinates in PySCF format.
    basis_list : list of str
        List of basis set names to compare.

    Returns
    -------
    results : dict
        Dictionary mapping basis name to (nao, eigenvalues, condition_number).
    """
    results = {}
    for basis in basis_list:
        try:
            nao, eigenvalues, cond = overlap_spectrum(atom, basis)
            results[basis] = {
                "nao": nao,
                "eigenvalues": eigenvalues,
                "condition_number": cond,
                "min_eigenvalue": eigenvalues[-1],
                "max_eigenvalue": eigenvalues[0],
            }
        except Exception as e:
            print(f"Warning: Could not compute for basis {basis}: {e}")
            results[basis] = None
    return results


def analyze_eigenvalue_distribution(eigenvalues: np.ndarray) -> dict:
    """
    Analyze the distribution of overlap matrix eigenvalues.

    This provides insight into the structure of the basis set:
      - Eigenvalues near 1: well-balanced, orthogonal-like functions
      - Large eigenvalues (>1): highly overlapping functions
      - Small eigenvalues (<1): spread-out, diffuse functions
      - Very small eigenvalues (<1e-6): near-linear dependence

    Parameters
    ----------
    eigenvalues : ndarray
        Eigenvalues of the overlap matrix (sorted descending).

    Returns
    -------
    analysis : dict
        Dictionary containing distribution statistics.
    """
    # Thresholds for categorization
    tiny = 1e-10
    small = 1e-6
    moderate = 1e-3

    analysis = {
        "total_count": len(eigenvalues),
        "max": eigenvalues[0],
        "min": eigenvalues[-1],
        "mean": np.mean(eigenvalues),
        "median": np.median(eigenvalues),
        "std": np.std(eigenvalues),
        # Count eigenvalues in different ranges
        "count_large": np.sum(eigenvalues > 1.5),
        "count_near_one": np.sum((eigenvalues >= 0.5) & (eigenvalues <= 1.5)),
        "count_small": np.sum((eigenvalues < 0.5) & (eigenvalues >= small)),
        "count_very_small": np.sum((eigenvalues < small) & (eigenvalues >= tiny)),
        "count_near_zero": np.sum(eigenvalues < tiny),
        # Effective dimension (how many eigenvalues are "significant")
        "effective_dimension": np.sum(eigenvalues > moderate),
    }

    return analysis


def print_eigenvalue_summary(basis: str, nao: int, eigenvalues: np.ndarray,
                              cond: float) -> None:
    """
    Print a formatted summary of overlap eigenvalue analysis.

    Parameters
    ----------
    basis : str
        Basis set name.
    nao : int
        Number of atomic orbitals.
    eigenvalues : ndarray
        Eigenvalues of S.
    cond : float
        Condition number.
    """
    print(f"\nBasis: {basis}")
    print(f"  Number of AOs:     {nao}")
    print(f"  Max eigenvalue:    {eigenvalues[0]:.6e}")
    print(f"  Min eigenvalue:    {eigenvalues[-1]:.6e}")
    print(f"  Condition number:  {cond:.6e}")

    # Classify conditioning
    if cond < 1e3:
        quality = "excellent (well-conditioned)"
    elif cond < 1e6:
        quality = "good"
    elif cond < 1e10:
        quality = "moderate (some linear dependence)"
    else:
        quality = "poor (significant linear dependence)"
    print(f"  Conditioning:      {quality}")


def print_comparison_table(results: dict) -> None:
    """
    Print a formatted comparison table of basis set conditioning.

    Parameters
    ----------
    results : dict
        Results from compare_basis_sets().
    """
    print("\n" + "=" * 80)
    print("Basis Set Comparison Summary")
    print("=" * 80)
    print(f"{'Basis':<16} {'NAO':>5} {'Min Eigenval':>14} {'Max Eigenval':>14} {'Cond Number':>14}")
    print("-" * 80)

    for basis, data in results.items():
        if data is not None:
            print(f"{basis:<16} {data['nao']:>5} {data['min_eigenvalue']:>14.3e} "
                  f"{data['max_eigenvalue']:>14.3e} {data['condition_number']:>14.3e}")
        else:
            print(f"{basis:<16} {'N/A':>5} {'N/A':>14} {'N/A':>14} {'N/A':>14}")

    print("-" * 80)


def demonstrate_diffuse_effect(results: dict) -> None:
    """
    Demonstrate the effect of diffuse functions on conditioning.

    Diffuse functions (indicated by 'aug-' prefix) extend further from the nucleus
    and tend to overlap more with each other, leading to smaller eigenvalues
    and worse conditioning.

    Parameters
    ----------
    results : dict
        Results from compare_basis_sets().
    """
    print("\n" + "=" * 80)
    print("Effect of Diffuse Functions on Conditioning")
    print("=" * 80)

    # Find pairs of standard and augmented basis sets
    pairs = [
        ("cc-pvdz", "aug-cc-pvdz"),
        ("cc-pvtz", "aug-cc-pvtz"),
    ]

    for standard, augmented in pairs:
        if standard in results and augmented in results:
            std_data = results[standard]
            aug_data = results[augmented]
            if std_data is not None and aug_data is not None:
                print(f"\n{standard} vs {augmented}:")
                print(f"  NAO increase:         {std_data['nao']} -> {aug_data['nao']} "
                      f"(+{aug_data['nao'] - std_data['nao']})")
                print(f"  Min eigenvalue:       {std_data['min_eigenvalue']:.3e} -> "
                      f"{aug_data['min_eigenvalue']:.3e}")

                # How much worse is conditioning?
                cond_ratio = aug_data['condition_number'] / std_data['condition_number']
                print(f"  Condition number:     {std_data['condition_number']:.3e} -> "
                      f"{aug_data['condition_number']:.3e}")
                print(f"  Conditioning worsened by factor: {cond_ratio:.1f}x")


def main():
    """Main function demonstrating Lab 2A concepts."""
    print("=" * 80)
    print("Lab 2A: Eigenvalues of S and Conditioning vs Basis Set")
    print("=" * 80)

    # Test molecule: water
    atom = "O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043"
    print(f"\nTest molecule: H2O")
    print(f"Geometry: {atom}")

    # List of basis sets to compare
    # Ordered roughly by size and quality
    basis_list = [
        "sto-3g",       # Minimal basis
        "6-31g",        # Split-valence
        "6-31g*",       # Split-valence + polarization on heavy atoms
        "cc-pvdz",      # Correlation-consistent double-zeta
        "aug-cc-pvdz",  # With diffuse functions
        "aug-cc-pvtz",  # Triple-zeta with diffuse
    ]

    print(f"\nBasis sets to compare: {basis_list}")

    # Part 1: Compare all basis sets
    print("\n" + "-" * 80)
    print("Part 1: Computing overlap eigenvalues for each basis set")
    print("-" * 80)

    results = compare_basis_sets(atom, basis_list)

    for basis in basis_list:
        if results[basis] is not None:
            print_eigenvalue_summary(
                basis,
                results[basis]["nao"],
                results[basis]["eigenvalues"],
                results[basis]["condition_number"]
            )

    # Part 2: Comparison table
    print_comparison_table(results)

    # Part 3: Analyze effect of diffuse functions
    demonstrate_diffuse_effect(results)

    # Part 4: Detailed eigenvalue distribution analysis
    print("\n" + "=" * 80)
    print("Part 4: Detailed Eigenvalue Distribution Analysis")
    print("=" * 80)

    for basis in ["cc-pvdz", "aug-cc-pvdz"]:
        if results[basis] is not None:
            analysis = analyze_eigenvalue_distribution(results[basis]["eigenvalues"])
            print(f"\n{basis}:")
            print(f"  Total functions:      {analysis['total_count']}")
            print(f"  Effective dimension:  {analysis['effective_dimension']} "
                  f"(eigenvalues > 1e-3)")
            print(f"  Large (>1.5):         {analysis['count_large']}")
            print(f"  Near one (0.5-1.5):   {analysis['count_near_one']}")
            print(f"  Small (1e-6 to 0.5):  {analysis['count_small']}")
            print(f"  Very small (<1e-6):   {analysis['count_very_small']}")
            print(f"  Near zero (<1e-10):   {analysis['count_near_zero']}")

    # Part 5: What you should observe
    print("\n" + "=" * 80)
    print("What You Should Observe")
    print("=" * 80)
    print("""
1. SCALING: The number of AOs increases with basis set quality:
   - STO-3G (minimal):     ~7 AOs for H2O
   - cc-pVDZ (double-zeta): ~24 AOs for H2O
   - aug-cc-pVTZ:          ~92 AOs for H2O

2. CONDITIONING: The condition number kappa(S) worsens (increases) with:
   - More diffuse functions (aug- prefix)
   - Larger basis sets with similar exponents
   - Functions centered on nearby atoms

3. DIFFUSE FUNCTION EFFECT: Adding 'aug-' typically:
   - Increases NAO by ~30-50%
   - Decreases minimum eigenvalue by 1-3 orders of magnitude
   - Worsens condition number by 10-1000x

4. NUMERICAL IMPLICATIONS:
   - kappa(S) < 10^6:  Generally safe for double precision
   - kappa(S) > 10^8:  May see numerical instabilities
   - kappa(S) > 10^12: Eigenvalue thresholding essential

5. WHY THIS MATTERS: When solving FC = SCe:
   - We need X such that X^T S X = I
   - X = S^(-1/2) amplifies errors by sqrt(kappa(S))
   - Large condition numbers require careful numerics
""")

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print("All computations completed successfully.")
    print("See Chapter 2, Section 2.10 for theoretical background.")
    print("Proceed to Lab 2B to build orthogonalizers that handle ill-conditioning.")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
