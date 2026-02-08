#!/usr/bin/env python3
"""
Lab 6B Solution: Pulay DIIS Acceleration and Difficult SCF Cases

This script implements Algorithm 6.2 from Chapter 6: Pulay's Direct Inversion
in the Iterative Subspace (DIIS) for accelerating SCF convergence.

Learning Objectives:
1. Understand why SCF can converge slowly or oscillate
2. Build SCF error vectors: e = FPS - SPF
3. Store Fock/error history and solve the DIIS linear system
4. Extrapolate a better Fock matrix: F_DIIS = sum_i c_i F_i
5. Compare convergence with and without DIIS on difficult cases

Test Systems:
- H2O / STO-3G (easy case, for baseline)
- Stretched H2 (R = 2.5 A) - difficult due to near-degeneracy

Physical Insight:
-----------------
The SCF iteration is a fixed-point problem: P -> F(P) -> C -> P_new.
Near convergence, consecutive Fock matrices lie in an "error subspace."
DIIS finds the optimal linear combination of recent Fock matrices that
minimizes the error, effectively jumping closer to the solution.

For stretched bonds, the HOMO-LUMO gap becomes small, causing:
- Slow convergence (many iterations)
- Oscillations between nearly-degenerate solutions
- Potential divergence without damping/DIIS

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 6: Hartree-Fock Self-Consistent Field as an Integral-Driven Algorithm
"""

import numpy as np
from pyscf import gto, scf
from typing import Tuple, Optional, List


# =============================================================================
# Section 1: DIIS Implementation
# =============================================================================

class PulayDIIS:
    """
    Pulay's Direct Inversion in the Iterative Subspace (DIIS) for SCF acceleration.

    DIIS constructs an improved Fock matrix as a linear combination of previous
    Fock matrices, with coefficients chosen to minimize the norm of the
    corresponding linear combination of error vectors.

    The error vector for RHF is the SCF commutator residual:
        e_i = vec(F_i P_i S - S P_i F_i)

    DIIS solves the constrained minimization:
        minimize ||sum_i c_i e_i||^2  subject to  sum_i c_i = 1

    This leads to the augmented linear system:
        [B  -1] [c]   [0]
        [-1  0] [L] = [-1]

    where B_ij = e_i^T e_j (error overlap matrix).

    Reference:
        Pulay, P. "Convergence acceleration of iterative sequences.
        The case of SCF iteration." Chem. Phys. Lett. 73, 393-398 (1980).
    """

    def __init__(self, max_vectors: int = 8, start_iter: int = 2):
        """
        Initialize DIIS extrapolator.

        Args:
            max_vectors: Maximum number of Fock/error vectors to store
            start_iter: Minimum number of stored vectors before DIIS kicks in
        """
        self.max_vectors = max_vectors
        self.start_iter = start_iter
        self.fock_list: List[np.ndarray] = []
        self.error_list: List[np.ndarray] = []

    def reset(self):
        """Clear the DIIS history."""
        self.fock_list = []
        self.error_list = []

    def update(self, F: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Add new Fock/error pair and return extrapolated Fock matrix.

        Args:
            F: Current Fock matrix (nao x nao)
            R: SCF residual R = FPS - SPF (nao x nao)

        Returns:
            F_diis: Extrapolated Fock matrix, or F if not enough history
        """
        # Flatten error matrix to vector
        error_vec = R.ravel().copy()

        # Store copies (important!)
        self.fock_list.append(F.copy())
        self.error_list.append(error_vec)

        # Trim history if exceeds max_vectors
        if len(self.fock_list) > self.max_vectors:
            self.fock_list.pop(0)
            self.error_list.pop(0)

        # Need at least start_iter vectors for DIIS
        m = len(self.fock_list)
        if m < self.start_iter:
            return F

        # Build error overlap matrix B
        # B_ij = e_i^T e_j
        B = np.zeros((m + 1, m + 1))
        for i in range(m):
            for j in range(m):
                B[i, j] = np.dot(self.error_list[i], self.error_list[j])

        # Augmented system with constraint sum_i c_i = 1
        # Last row/column enforce the constraint via Lagrange multiplier
        B[-1, :m] = -1.0
        B[:m, -1] = -1.0
        B[-1, -1] = 0.0

        # Right-hand side
        rhs = np.zeros(m + 1)
        rhs[-1] = -1.0

        # Solve the linear system
        try:
            solution = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            # Singular matrix - skip DIIS this iteration
            return F

        # Extract coefficients (exclude Lagrange multiplier)
        coeffs = solution[:m]

        # Verify constraint (for debugging)
        # assert abs(np.sum(coeffs) - 1.0) < 1e-10, "DIIS constraint violated"

        # Build extrapolated Fock matrix
        F_diis = np.zeros_like(F)
        for ci, Fi in zip(coeffs, self.fock_list):
            F_diis += ci * Fi

        return F_diis

    def get_coefficients(self) -> Optional[np.ndarray]:
        """Return the last computed DIIS coefficients (for debugging)."""
        return getattr(self, '_last_coeffs', None)


# =============================================================================
# Section 2: Core SCF Functions (from Lab 6A)
# =============================================================================

def symmetric_orthogonalizer(S: np.ndarray, threshold: float = 1e-10) -> np.ndarray:
    """Build symmetric orthogonalizer X = S^(-1/2)."""
    eigenvalues, U = np.linalg.eigh(S)
    keep = eigenvalues > threshold
    s_inv_sqrt = 1.0 / np.sqrt(eigenvalues[keep])
    X = U[:, keep] @ np.diag(s_inv_sqrt)  # rectangular (N x N')
    return X


def build_jk_matrices(P: np.ndarray, eri: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build Coulomb (J) and Exchange (K) matrices from density and ERIs."""
    J = np.einsum('uvls,ls->uv', eri, P, optimize=True)
    K = np.einsum('ulvs,ls->uv', eri, P, optimize=True)
    return J, K


def build_fock_matrix(h: np.ndarray, J: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Build RHF Fock matrix F = h + J - 0.5*K."""
    return h + J - 0.5 * K


def solve_roothaan_hall(F: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve Roothaan-Hall equations via orthogonalization."""
    F_prime = X.T @ F @ X
    eps, C_prime = np.linalg.eigh(F_prime)
    C = X @ C_prime
    return eps, C


def build_density_matrix(C: np.ndarray, n_occ: int) -> np.ndarray:
    """Build RHF density matrix P = 2 * C_occ @ C_occ.T."""
    C_occ = C[:, :n_occ]
    return 2.0 * C_occ @ C_occ.T


def compute_scf_energy(P: np.ndarray, h: np.ndarray, F: np.ndarray,
                       E_nuc: float) -> Tuple[float, float]:
    """Compute electronic and total SCF energies."""
    E_elec = 0.5 * np.einsum('uv,uv->', P, h + F, optimize=True)
    E_total = E_elec + E_nuc
    return E_elec, E_total


def compute_scf_residual(F: np.ndarray, P: np.ndarray,
                         S: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute SCF residual R = FPS - SPF."""
    R = F @ P @ S - S @ P @ F
    return R, np.linalg.norm(R)


def compute_density_rmsd(P_new: np.ndarray, P_old: np.ndarray) -> float:
    """Compute RMSD between density matrices."""
    diff = P_new - P_old
    return np.sqrt(np.sum(diff**2) / diff.size)


# =============================================================================
# Section 3: SCF Driver with Optional DIIS
# =============================================================================

def rhf_scf_with_diis(mol: gto.Mole,
                      use_diis: bool = True,
                      max_iter: int = 100,
                      conv_E: float = 1e-8,
                      conv_rmsd: float = 1e-8,
                      diis_max_vec: int = 8,
                      diis_start: int = 2,
                      damping: float = 0.0,
                      verbose: bool = True) -> dict:
    """
    Run RHF SCF with optional DIIS acceleration.

    Args:
        mol: PySCF Mole object
        use_diis: Enable DIIS acceleration
        max_iter: Maximum SCF iterations
        conv_E: Energy convergence threshold
        conv_rmsd: Density RMSD convergence threshold
        diis_max_vec: Maximum vectors in DIIS history
        diis_start: Minimum iterations before DIIS starts
        damping: Density damping factor (0 = no damping, 0.5 = 50% damping)
        verbose: Print iteration details

    Returns:
        Dictionary with SCF results
    """
    # Extract integrals
    S = mol.intor("int1e_ovlp")
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e", aosym="s1")
    E_nuc = mol.energy_nuc()

    nao = mol.nao_nr()
    n_elec = mol.nelectron
    n_occ = n_elec // 2

    # Build orthogonalizer
    X = symmetric_orthogonalizer(S)

    # Initial guess from core Hamiltonian
    eps, C = solve_roothaan_hall(h, X)
    P = build_density_matrix(C, n_occ)

    # Initialize DIIS
    diis = PulayDIIS(max_vectors=diis_max_vec, start_iter=diis_start) if use_diis else None

    if verbose:
        diis_str = "DIIS" if use_diis else "No DIIS"
        damp_str = f", damping={damping:.2f}" if damping > 0 else ""
        print(f"\nRHF SCF ({diis_str}{damp_str})")
        print("-" * 70)
        print(f"{'Iter':>4}  {'E_total (Hartree)':>20}  {'dE':>12}  {'RMSD(P)':>12}  {'|R|':>12}")
        print("-" * 70)

    # SCF loop
    history = []
    converged = False
    E_old = 0.0

    for iteration in range(1, max_iter + 1):
        # Build Fock matrix
        J, K = build_jk_matrices(P, eri)
        F = build_fock_matrix(h, J, K)

        # Compute residual (before DIIS)
        R, norm_R = compute_scf_residual(F, P, S)

        # Apply DIIS extrapolation
        if diis is not None:
            F = diis.update(F, R)

        # Solve Roothaan-Hall
        eps, C = solve_roothaan_hall(F, X)

        # Build new density
        P_new = build_density_matrix(C, n_occ)

        # Apply damping if requested
        if damping > 0:
            P_new = (1 - damping) * P_new + damping * P

        # Compute energy
        J_new, K_new = build_jk_matrices(P_new, eri)
        F_new = build_fock_matrix(h, J_new, K_new)
        E_elec, E_total = compute_scf_energy(P_new, h, F_new, E_nuc)

        # Convergence metrics
        dE = E_total - E_old
        rmsd_P = compute_density_rmsd(P_new, P)

        history.append({'E': E_total, 'dE': dE, 'rmsd_P': rmsd_P, 'norm_R': norm_R})

        if verbose:
            print(f"{iteration:4d}  {E_total:+20.12f}  {dE:+12.3e}  {rmsd_P:12.3e}  {norm_R:12.3e}")

        # Check convergence
        if iteration > 1 and abs(dE) < conv_E and rmsd_P < conv_rmsd:
            converged = True
            break

        P = P_new
        E_old = E_total

    if verbose:
        print("-" * 70)
        if converged:
            print(f"Converged in {iteration} iterations.")
        else:
            print(f"NOT converged after {iteration} iterations!")
        print(f"Final energy: {E_total:.10f} Hartree")

    return {
        'E_total': E_total,
        'E_elec': E_elec,
        'E_nuc': E_nuc,
        'C': C,
        'eps': eps,
        'P': P_new,
        'F': F_new,
        'converged': converged,
        'n_iter': iteration,
        'history': history
    }


# =============================================================================
# Section 4: Test Cases
# =============================================================================

def test_easy_case():
    """Test on H2O/STO-3G (should converge easily)."""
    print("\n" + "=" * 70)
    print("Test Case 1: H2O / STO-3G (Easy Case)")
    print("=" * 70)

    mol = gto.M(
        atom="""
            O   0.000000   0.000000   0.117369
            H   0.756950   0.000000  -0.469476
            H  -0.756950   0.000000  -0.469476
        """,
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    print("\n--- Without DIIS ---")
    result_no_diis = rhf_scf_with_diis(mol, use_diis=False, verbose=True)

    print("\n--- With DIIS ---")
    result_with_diis = rhf_scf_with_diis(mol, use_diis=True, verbose=True)

    # PySCF reference
    mf = scf.RHF(mol)
    mf.verbose = 0
    E_ref = mf.kernel()

    print("\n" + "-" * 50)
    print("Comparison:")
    print(f"  Without DIIS: {result_no_diis['n_iter']:3d} iterations, E = {result_no_diis['E_total']:.10f}")
    print(f"  With DIIS:    {result_with_diis['n_iter']:3d} iterations, E = {result_with_diis['E_total']:.10f}")
    print(f"  PySCF:        E = {E_ref:.10f}")
    print(f"  Error (DIIS): {abs(result_with_diis['E_total'] - E_ref):.2e} Hartree")

    return result_no_diis, result_with_diis


def test_stretched_h2(R: float = 2.5):
    """Test on stretched H2 (difficult convergence)."""
    print("\n" + "=" * 70)
    print(f"Test Case 2: Stretched H2 (R = {R} Angstrom) - Difficult Case")
    print("=" * 70)

    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {R}",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    # Compute HOMO-LUMO gap to show why this is difficult
    mf_ref = scf.RHF(mol)
    mf_ref.verbose = 0
    mf_ref.kernel()
    gap = mf_ref.mo_energy[1] - mf_ref.mo_energy[0]
    print(f"\nHOMO-LUMO gap: {gap:.6f} Hartree = {gap * 27.2114:.4f} eV")
    print("(Small gap = difficult SCF convergence)")

    print("\n--- Without DIIS ---")
    result_no_diis = rhf_scf_with_diis(mol, use_diis=False, max_iter=100, verbose=True)

    print("\n--- With DIIS ---")
    result_with_diis = rhf_scf_with_diis(mol, use_diis=True, max_iter=100, verbose=True)

    E_ref = mf_ref.e_tot

    print("\n" + "-" * 50)
    print("Comparison:")
    print(f"  Without DIIS: {result_no_diis['n_iter']:3d} iterations, "
          f"E = {result_no_diis['E_total']:.10f}, converged = {result_no_diis['converged']}")
    print(f"  With DIIS:    {result_with_diis['n_iter']:3d} iterations, "
          f"E = {result_with_diis['E_total']:.10f}, converged = {result_with_diis['converged']}")
    print(f"  PySCF:        E = {E_ref:.10f}")

    return result_no_diis, result_with_diis


def test_very_stretched_h2(R: float = 4.0):
    """Test on very stretched H2 (extremely difficult)."""
    print("\n" + "=" * 70)
    print(f"Test Case 3: Very Stretched H2 (R = {R} Angstrom) - Very Difficult")
    print("=" * 70)

    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {R}",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    # Show HOMO-LUMO gap
    mf_ref = scf.RHF(mol)
    mf_ref.verbose = 0
    mf_ref.kernel()
    gap = mf_ref.mo_energy[1] - mf_ref.mo_energy[0]
    print(f"\nHOMO-LUMO gap: {gap:.6f} Hartree = {gap * 27.2114:.4f} eV")

    print("\n--- Without DIIS (with damping=0.5) ---")
    result_no_diis = rhf_scf_with_diis(mol, use_diis=False, damping=0.5,
                                        max_iter=100, verbose=True)

    print("\n--- With DIIS ---")
    result_with_diis = rhf_scf_with_diis(mol, use_diis=True, max_iter=100, verbose=True)

    print("\n--- With DIIS + Damping (hybrid approach) ---")
    result_hybrid = rhf_scf_with_diis(mol, use_diis=True, damping=0.3,
                                       max_iter=100, verbose=True)

    E_ref = mf_ref.e_tot

    print("\n" + "-" * 50)
    print("Comparison:")
    print(f"  Damping only: {result_no_diis['n_iter']:3d} iter, converged={result_no_diis['converged']}")
    print(f"  DIIS only:    {result_with_diis['n_iter']:3d} iter, converged={result_with_diis['converged']}")
    print(f"  DIIS+Damping: {result_hybrid['n_iter']:3d} iter, converged={result_hybrid['converged']}")
    print(f"  PySCF ref:    E = {E_ref:.10f}")

    return result_no_diis, result_with_diis, result_hybrid


def study_convergence_vs_bond_length():
    """Study how SCF difficulty varies with H2 bond length."""
    print("\n" + "=" * 70)
    print("Convergence Study: H2 SCF vs Bond Length")
    print("=" * 70)

    distances = [0.74, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    print("\n" + "-" * 80)
    print(f"{'R (Ang)':>8}  {'Gap (eV)':>10}  {'Iter (no DIIS)':>15}  "
          f"{'Iter (DIIS)':>12}  {'E (Hartree)':>14}")
    print("-" * 80)

    for R in distances:
        mol = gto.M(
            atom=f"H 0 0 0; H 0 0 {R}",
            basis="sto-3g",
            unit="Angstrom",
            verbose=0
        )

        # Without DIIS
        result_no_diis = rhf_scf_with_diis(mol, use_diis=False, max_iter=100, verbose=False)
        iter_no_diis = result_no_diis['n_iter'] if result_no_diis['converged'] else ">100"

        # With DIIS
        result_with_diis = rhf_scf_with_diis(mol, use_diis=True, max_iter=100, verbose=False)
        iter_with_diis = result_with_diis['n_iter'] if result_with_diis['converged'] else ">100"

        # Reference for gap
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.kernel()
        gap_ev = (mf.mo_energy[1] - mf.mo_energy[0]) * 27.2114

        E_final = result_with_diis['E_total'] if result_with_diis['converged'] else result_no_diis['E_total']

        print(f"{R:8.2f}  {gap_ev:10.4f}  {str(iter_no_diis):>15}  "
              f"{str(iter_with_diis):>12}  {E_final:14.8f}")

    print("-" * 80)


# =============================================================================
# Section 5: Main Demonstration
# =============================================================================

def main():
    """Run the complete Lab 6B demonstration."""

    print()
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*   Lab 6B: Pulay DIIS and Difficult SCF Cases                     *")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()

    # Run test cases
    test_easy_case()
    test_stretched_h2(R=2.5)
    test_very_stretched_h2(R=4.0)
    study_convergence_vs_bond_length()

    # =========================================================================
    # What You Should Observe
    # =========================================================================
    print("\n" + "=" * 70)
    print("What You Should Observe")
    print("=" * 70)

    observations = """
1. EASY CASE (H2O at equilibrium):
   - Both methods converge, but DIIS typically uses fewer iterations
   - The improvement is modest (maybe 15 -> 10 iterations)
   - DIIS shows characteristic "jumps" in energy as it extrapolates

2. STRETCHED H2 (R ~ 2.5 A):
   - Without DIIS: May converge slowly or oscillate
   - With DIIS: Usually converges faster and more reliably
   - The HOMO-LUMO gap is small (~0.1-0.2 eV), causing difficulties

3. VERY STRETCHED H2 (R ~ 4.0 A):
   - Without acceleration: Often fails to converge or oscillates
   - DIIS alone may not be enough - damping can help stabilize
   - The hybrid approach (DIIS + damping) is often most robust

4. CONVERGENCE vs BOND LENGTH:
   - At equilibrium (R=0.74 A): Large gap, easy convergence
   - As R increases: Gap decreases, convergence becomes harder
   - At large R: RHF struggles because it cannot describe bond breaking

5. PHYSICAL ORIGIN OF DIFFICULTY:
   - Small HOMO-LUMO gap = near-degeneracy
   - Near-degeneracy causes oscillation between different solutions
   - Stretched bonds require correlation (beyond HF) for proper description

6. DIIS MECHANICS:
   - DIIS needs a few iterations to build history (typically 2-3)
   - The extrapolation can overshoot, causing temporary energy increases
   - Singular B matrix can occur when errors become linearly dependent

7. PRACTICAL RECOMMENDATIONS:
   - Always use DIIS (or equivalent) in production codes
   - For difficult cases, combine with damping or level shifting
   - If SCF fails, try: different initial guess, larger DIIS space, damping
"""
    print(observations)

    print("\n" + "=" * 70)
    print("Lab 6B Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
