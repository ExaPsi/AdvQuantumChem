#!/usr/bin/env python3
"""
lab6b_diis.py - Pulay DIIS Implementation and Difficult SCF Cases (Lab 6B)

This module implements Pulay's Direct Inversion in the Iterative Subspace (DIIS)
for accelerating SCF convergence, and demonstrates its effectiveness on
difficult cases like stretched H2.

DIIS Concept:
    The SCF error vector is defined as R = FPS - SPF (the commutator).
    At convergence, R = 0 because F and P share the same eigenvectors.

    DIIS constructs an improved Fock matrix as a linear combination:
        F_DIIS = sum_i c_i F_i

    The coefficients c_i minimize ||sum_i c_i r_i||^2 subject to sum_i c_i = 1.

    This leads to a linear system involving the B matrix:
        B_ij = r_i . r_j  (dot product of error vectors)

DIIS typically reduces SCF iterations by a factor of 2-5 compared to simple
fixed-point iteration.

References:
    - Chapter 6, Section 6: SCF convergence and DIIS
    - Algorithm 6.2: Pulay DIIS for RHF
    - Listings 6.3-6.4: DIIS implementation and testing
    - P. Pulay, Chem. Phys. Lett. 73, 393 (1980)
    - P. Pulay, J. Comp. Chem. 3, 556 (1982)

Course: 2302638 Advanced Quantum Chemistry
"""

import numpy as np
from typing import List, Optional
from lab6a_rhf_scf import symm_orth, build_jk, rhf_scf


class PulayDIIS:
    """
    Pulay DIIS accelerator for SCF convergence.

    DIIS stores a history of Fock matrices and error vectors, then
    constructs an improved Fock matrix by minimizing the residual
    in the subspace spanned by previous iterations.

    Attributes
    ----------
    max_vec : int
        Maximum number of vectors to store
    F_list : List[np.ndarray]
        History of Fock matrices
    R_list : List[np.ndarray]
        History of flattened error vectors
    """

    def __init__(self, max_vec: int = 8):
        """
        Initialize DIIS accelerator.

        Parameters
        ----------
        max_vec : int
            Maximum number of Fock/error vector pairs to store
        """
        self.max_vec = max_vec
        self.F_list: List[np.ndarray] = []
        self.R_list: List[np.ndarray] = []

    def update(self, F: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Update DIIS with new Fock matrix and error vector.

        The DIIS equations are:
            [B  -1] [c ]   [0 ]
            [-1  0] [λ ] = [-1]

        where B_ij = r_i . r_j

        Parameters
        ----------
        F : np.ndarray
            Current Fock matrix
        R : np.ndarray
            Current SCF residual (FPS - SPF)

        Returns
        -------
        F_new : np.ndarray
            Extrapolated Fock matrix, or original F if DIIS fails
        """
        # Store copies
        self.F_list.append(F.copy())
        self.R_list.append(R.reshape(-1).copy())

        # Truncate to max_vec
        if len(self.F_list) > self.max_vec:
            self.F_list.pop(0)
            self.R_list.pop(0)

        m = len(self.F_list)

        # Need at least 2 vectors for extrapolation
        if m < 2:
            return F

        # Build B matrix (including Lagrange multiplier row/column)
        # B = [r_i . r_j  | -1]
        #     [-1         |  0]
        B = np.empty((m + 1, m + 1), dtype=float)
        B[-1, :] = -1.0
        B[:, -1] = -1.0
        B[-1, -1] = 0.0

        for i in range(m):
            for j in range(m):
                B[i, j] = np.dot(self.R_list[i], self.R_list[j])

        # RHS: [0, 0, ..., 0, -1]
        rhs = np.zeros(m + 1, dtype=float)
        rhs[-1] = -1.0

        # Solve the linear system
        try:
            sol = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            # Singular B matrix -> skip DIIS this step
            return F

        # Extract coefficients (exclude Lagrange multiplier)
        c = sol[:m]

        # Construct extrapolated Fock matrix
        F_new = np.zeros_like(F)
        for ci, Fi in zip(c, self.F_list):
            F_new += ci * Fi

        return F_new

    def reset(self):
        """Clear the DIIS history."""
        self.F_list.clear()
        self.R_list.clear()


# =============================================================================
# DIIS analysis and testing
# =============================================================================

def analyze_diis_convergence():
    """
    Compare SCF convergence with and without DIIS for H2O/STO-3G.
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        print("PySCF not installed. Cannot run analysis.")
        return

    print("=" * 70)
    print("DIIS Convergence Analysis: H2O/STO-3G")
    print("=" * 70)

    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    S = mol.intor("int1e_ovlp")
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e", aosym="s1")
    Enuc = mol.energy_nuc()

    print("\n--- SCF without DIIS ---")
    try:
        E1, *_ = rhf_scf(S, h, eri, mol.nelectron, Enuc=Enuc, diis=None, max_cycle=30, verbose=True)
    except RuntimeError as e:
        print(f"  {e}")
        E1 = None

    print("\n--- SCF with DIIS ---")
    diis = PulayDIIS(max_vec=8)
    try:
        E2, *_ = rhf_scf(S, h, eri, mol.nelectron, Enuc=Enuc, diis=diis, max_cycle=30, verbose=True)
    except RuntimeError as e:
        print(f"  {e}")
        E2 = None

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    if E1 is not None:
        print(f"  E (no DIIS): {E1:.10f}")
    else:
        print("  E (no DIIS): Did not converge")
    if E2 is not None:
        print(f"  E (DIIS):    {E2:.10f}")
    else:
        print("  E (DIIS):    Did not converge")


def test_stretched_h2():
    """
    Test DIIS on stretched H2 - a challenging SCF case.

    At large bond lengths, H2 exhibits:
    - Small HOMO-LUMO gap
    - Slow SCF convergence
    - Possible oscillations without DIIS
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        print("PySCF not installed.")
        return

    print("\n" + "=" * 70)
    print("Lab 6B: Stretched H2 Test (DIIS vs no DIIS)")
    print("=" * 70)

    R = 2.5  # Angstrom, deliberately stretched
    print(f"\nH2 bond length: {R} Angstrom (stretched)")
    print("This is a difficult SCF case due to small HOMO-LUMO gap.")

    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {R}",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    S = mol.intor("int1e_ovlp")
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e", aosym="s1")
    Enuc = mol.energy_nuc()

    print("\n--- SCF without DIIS ---")
    try:
        E1, eps1, *_ = rhf_scf(S, h, eri, mol.nelectron, Enuc=Enuc, diis=None, max_cycle=50, verbose=True)
        gap1 = eps1[1] - eps1[0]  # LUMO - HOMO
        print(f"  HOMO-LUMO gap: {gap1:.6f} Hartree")
    except RuntimeError as e:
        print(f"  {e}")
        E1, gap1 = None, None

    print("\n--- SCF with DIIS ---")
    diis = PulayDIIS(max_vec=8)
    try:
        E2, eps2, *_ = rhf_scf(S, h, eri, mol.nelectron, Enuc=Enuc, diis=diis, max_cycle=50, verbose=True)
        gap2 = eps2[1] - eps2[0]
        print(f"  HOMO-LUMO gap: {gap2:.6f} Hartree")
    except RuntimeError as e:
        print(f"  {e}")
        E2, gap2 = None, None

    # PySCF reference
    mf = scf.RHF(mol)
    mf.verbose = 0
    E_ref = mf.kernel()

    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    if E1 is not None:
        print(f"  E (no DIIS): {E1:.10f}")
    else:
        print("  E (no DIIS): Did not converge")
    if E2 is not None:
        print(f"  E (DIIS):    {E2:.10f}")
    else:
        print("  E (DIIS):    Did not converge")
    print(f"  E (PySCF):   {E_ref:.10f}")

    if E2 is not None:
        diff = E2 - E_ref
        print(f"\n  DIIS vs PySCF difference: {diff:+.2e} Hartree")


def test_stretched_n2():
    """
    Test DIIS on stretched N2 - another challenging case.
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        print("PySCF not installed.")
        return

    print("\n" + "=" * 70)
    print("Stretched N2 Test (DIIS vs no DIIS)")
    print("=" * 70)

    R = 2.5  # Angstrom, stretched from equilibrium ~1.1 A
    print(f"\nN2 bond length: {R} Angstrom (stretched)")

    mol = gto.M(
        atom=f"N 0 0 0; N 0 0 {R}",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    S = mol.intor("int1e_ovlp")
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e", aosym="s1")
    Enuc = mol.energy_nuc()

    print("\n--- SCF without DIIS ---")
    try:
        E1, eps1, *_ = rhf_scf(S, h, eri, mol.nelectron, Enuc=Enuc, diis=None, max_cycle=50, verbose=True)
    except RuntimeError as e:
        print(f"  {e}")
        E1 = None

    print("\n--- SCF with DIIS ---")
    diis = PulayDIIS(max_vec=8)
    try:
        E2, eps2, *_ = rhf_scf(S, h, eri, mol.nelectron, Enuc=Enuc, diis=diis, max_cycle=50, verbose=True)
    except RuntimeError as e:
        print(f"  {e}")
        E2 = None

    # PySCF reference
    mf = scf.RHF(mol)
    mf.verbose = 0
    E_ref = mf.kernel()

    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    if E1 is not None:
        print(f"  E (no DIIS): {E1:.10f}")
    else:
        print("  E (no DIIS): Did not converge")
    if E2 is not None:
        print(f"  E (DIIS):    {E2:.10f}")
    else:
        print("  E (DIIS):    Did not converge")
    print(f"  E (PySCF):   {E_ref:.10f}")


def demonstrate_diis_theory():
    """
    Print the DIIS theory explanation.
    """
    print("\n" + "=" * 70)
    print("DIIS Theory: Direct Inversion in the Iterative Subspace")
    print("=" * 70)

    print("""
The SCF residual is defined as:
    R = FPS - SPF   (commutator in AO basis)

At convergence, F and P share the same eigenvectors, so:
    FPS = SPF  =>  R = 0

DIIS Goal:
    Find coefficients c_i such that the residual of the extrapolated
    Fock matrix is minimized:
        minimize ||sum_i c_i r_i||^2  subject to  sum_i c_i = 1

This is a constrained least-squares problem with Lagrangian:
    L = sum_{ij} c_i c_j (r_i . r_j) - lambda(sum_i c_i - 1)

Setting derivatives to zero gives the linear system:
    [B  -1] [c ]   [0 ]
    [-1  0] [lambda] = [-1]

where B_ij = r_i . r_j

The extrapolated Fock matrix is:
    F_DIIS = sum_i c_i F_i

DIIS typically reduces iteration count by 2-5x compared to simple
fixed-point iteration, especially for systems with small HOMO-LUMO gaps.

Common enhancements:
    1. Damping before DIIS starts (first few iterations)
    2. Level shifting for very small gaps
    3. EDIIS for initial iterations, then switch to DIIS
""")


# =============================================================================
# Checkpoint questions
# =============================================================================

def checkpoint_diis_behavior():
    """
    Checkpoint: Understanding DIIS behavior.
    """
    print("\n" + "=" * 70)
    print("Checkpoint: DIIS Behavior for Difficult Cases")
    print("=" * 70)

    print("""
For stretched H2, does DIIS always help?

Answer: DIIS significantly accelerates convergence for most cases, but
for very difficult cases (small HOMO-LUMO gap, near-degeneracy), DIIS
alone may not be sufficient. Consider:

1. Adding initial damping (first 2-3 cycles with alpha = 0.5):
   P_new = alpha * P_new + (1 - alpha) * P_old

2. Level shifting (add constant to virtual orbital energies):
   epsilon_a -> epsilon_a + sigma  for a in virtual

3. Starting DIIS only after a few damped iterations

Why does this combination work?
- Damping prevents large oscillations in early iterations
- Once the density is closer to convergence, DIIS takes over
- DIIS excels at accelerating final convergence
- The combination is more robust than either approach alone
""")


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Lab 6B: Pulay DIIS and Difficult SCF Cases")
    print("Course: 2302638 Advanced Quantum Chemistry")
    print("=" * 70)

    # Theory explanation
    demonstrate_diis_theory()

    # Test on easy case first
    analyze_diis_convergence()

    # Test on difficult cases
    test_stretched_h2()
    test_stretched_n2()

    # Checkpoint question
    checkpoint_diis_behavior()

    print("\n" + "=" * 70)
    print("Lab 6B Complete")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. DIIS minimizes ||sum_i c_i r_i||^2 subject to sum_i c_i = 1")
    print("  2. The error vector R = FPS - SPF vanishes at convergence")
    print("  3. DIIS typically reduces iterations by 2-5x")
    print("  4. For difficult cases, combine damping + DIIS")
    print("=" * 70)
