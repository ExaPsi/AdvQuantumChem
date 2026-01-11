#!/usr/bin/env python3
"""
Lab 3A Solution: One-Electron Integral Sanity Checks

This solution script demonstrates systematic verification of one-electron
integrals extracted from PySCF. For H2O in the STO-3G basis, we verify:

1. SYMMETRY: All one-electron operators are Hermitian (real symmetric)
   - Overlap S: <mu|nu> = <nu|mu>
   - Kinetic T: <mu|-1/2 nabla^2|nu> = <nu|-1/2 nabla^2|mu>
   - Nuclear V: <mu|V_nuc|nu> = <nu|V_nuc|mu>

2. SIGN CONVENTIONS:
   - Kinetic T: Always positive (kinetic energy is positive)
   - Nuclear V: Always negative (electron-nucleus attraction)

3. MATRIX STRUCTURE:
   - Diagonal dominance patterns
   - Off-diagonal decay with spatial separation

4. PHYSICAL CONSISTENCY:
   - Electron count: Tr[P*S] = N_electrons
   - Core-valence separation visible in matrix structure

Physical Insight:
-----------------
The one-electron integrals encode the physics of a single electron in the
field of the nuclei. The overlap matrix S is NOT the identity because
atomic orbitals on different centers overlap in space. The kinetic matrix
T measures how "curved" the wavefunction is (high curvature = high kinetic
energy). The nuclear attraction V is always negative because electrons are
attracted to nuclei.

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 3: One-Electron Integrals and Gaussian Product Theorem
"""

import numpy as np
from pyscf import gto, scf


def print_section(title: str, width: int = 60) -> None:
    """Print a formatted section header."""
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def check_symmetry(M: np.ndarray, name: str, tol: float = 1e-12) -> bool:
    """
    Check if a matrix is symmetric (Hermitian for real matrices).

    For real one-electron operators, symmetry follows from the fact that
    the operator is self-adjoint and the basis functions are real:
        <mu|O|nu> = <nu|O|mu>*  and for real functions, * is identity.

    Parameters
    ----------
    M : np.ndarray
        Matrix to check
    name : str
        Name of the matrix for printing
    tol : float
        Tolerance for symmetry check

    Returns
    -------
    bool
        True if matrix is symmetric within tolerance
    """
    is_symmetric = np.allclose(M, M.T, atol=tol)
    antisym_norm = np.linalg.norm(M - M.T, 'fro')
    print(f"   {name}: symmetric = {is_symmetric}, ||M - M^T||_F = {antisym_norm:.2e}")
    return is_symmetric


def analyze_diagonal_pattern(M: np.ndarray, name: str, ao_labels: list) -> None:
    """
    Analyze diagonal elements to identify core vs valence structure.

    For one-electron integrals:
    - Core orbitals (1s on O): Large kinetic/nuclear attraction
    - Valence orbitals (2s, 2p on O; 1s on H): Smaller magnitudes

    Parameters
    ----------
    M : np.ndarray
        Matrix to analyze
    name : str
        Name of the matrix
    ao_labels : list
        Labels for each AO (from mol.ao_labels())
    """
    diag = np.diag(M)
    print(f"\n   {name} diagonal elements:")
    for i, (val, label) in enumerate(zip(diag, ao_labels)):
        # Truncate label for clean output
        short_label = label.replace(' ', '')[:12]
        print(f"      [{i:2d}] {short_label:12s} : {val:12.6f}")


def analyze_off_diagonal_decay(M: np.ndarray, name: str) -> None:
    """
    Analyze off-diagonal decay pattern.

    Off-diagonal elements of one-electron integrals decay with the
    spatial separation of the basis function centers. This is because:
    - Overlap decays as exp(-mu * R^2)
    - Kinetic is proportional to overlap for s-type
    - Nuclear attraction involves Boys function which also decays

    Parameters
    ----------
    M : np.ndarray
        Matrix to analyze
    name : str
        Name of the matrix
    """
    n = M.shape[0]
    diag_vals = np.diag(M)

    # Compute average absolute off-diagonal
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag_vals = np.abs(M[off_diag_mask])

    avg_diag = np.mean(np.abs(diag_vals))
    avg_off_diag = np.mean(off_diag_vals)
    max_off_diag = np.max(off_diag_vals)

    print(f"\n   {name} structure:")
    print(f"      Average |diagonal|:     {avg_diag:12.6f}")
    print(f"      Average |off-diagonal|: {avg_off_diag:12.6f}")
    print(f"      Max |off-diagonal|:     {max_off_diag:12.6f}")
    print(f"      Off-diag / Diag ratio:  {avg_off_diag/avg_diag:.3f}")


def main():
    print("=" * 70)
    print("Lab 3A Solution: One-Electron Integral Sanity Checks")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Build the H2O molecule
    # =========================================================================
    # Water geometry: O at origin, H atoms in xz plane
    # Bond length ~ 0.96 A, bond angle ~ 104.5 degrees
    mol = gto.M(
        atom="""
            O   0.000000   0.000000   0.000000
            H   0.758600   0.000000   0.504300
            H  -0.758600   0.000000   0.504300
        """,
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    nao = mol.nao_nr()
    ao_labels = mol.ao_labels()

    print("\n" + "=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)
    print(f"\nMolecule: H2O (water)")
    print(f"Basis set: STO-3G (minimal basis)")
    print(f"Number of AOs: {nao}")
    print(f"Number of electrons: {mol.nelectron}")
    print(f"\nAO labels:")
    for i, label in enumerate(ao_labels):
        print(f"   [{i:2d}] {label}")

    # =========================================================================
    # STEP 2: Extract one-electron integrals
    # =========================================================================
    # These are the fundamental one-electron operators in quantum chemistry:
    #
    # S_uv = <chi_u | chi_v>              : Overlap (metric tensor)
    # T_uv = <chi_u | -1/2 nabla^2 | chi_v> : Kinetic energy
    # V_uv = <chi_u | sum_A -Z_A/r_A | chi_v> : Nuclear attraction
    # h_uv = T_uv + V_uv                  : Core Hamiltonian
    #
    print_section("1. EXTRACTING ONE-ELECTRON INTEGRALS")

    S = mol.intor("int1e_ovlp")   # Overlap matrix
    T = mol.intor("int1e_kin")    # Kinetic energy matrix
    V = mol.intor("int1e_nuc")    # Nuclear attraction matrix
    h = T + V                      # Core Hamiltonian

    print(f"\n   Matrix shapes: S{S.shape}, T{T.shape}, V{V.shape}")

    # =========================================================================
    # STEP 3: Verify symmetry (Hermitian property)
    # =========================================================================
    # All one-electron operators in quantum chemistry are Hermitian:
    #   O_uv = <u|O|v> = <v|O|u>* = O_vu*
    # For real basis functions and real operators, this means O = O^T
    #
    print_section("2. SYMMETRY VERIFICATION (Hermitian Property)")

    print("\n   For real basis functions, Hermitian means symmetric: M = M^T")
    print("   Physical origin: Operators like T and V are self-adjoint.\n")

    all_symmetric = True
    all_symmetric &= check_symmetry(S, "Overlap S")
    all_symmetric &= check_symmetry(T, "Kinetic T")
    all_symmetric &= check_symmetry(V, "Nuclear V")
    all_symmetric &= check_symmetry(h, "Core Ham h")

    print(f"\n   All matrices symmetric: {all_symmetric}")

    # =========================================================================
    # STEP 4: Sign convention checks
    # =========================================================================
    # Physical expectations:
    # - Kinetic energy is always positive: T >= 0
    #   (Uncertainty principle: localized electron has high kinetic energy)
    # - Nuclear attraction is always negative: V < 0
    #   (Electrons are attracted to nuclei: -Z_A/r_A < 0)
    #
    print_section("3. SIGN CONVENTION VERIFICATION")

    print("\n   Physical expectations:")
    print("   - Kinetic T: Diagonal elements should be POSITIVE")
    print("     (Kinetic energy = curvature of wavefunction >= 0)")
    print("   - Nuclear V: Diagonal elements should be NEGATIVE")
    print("     (Electron-nucleus attraction: V = -Z/r < 0)")

    T_diag = np.diag(T)
    V_diag = np.diag(V)

    T_positive = np.all(T_diag > 0)
    V_negative = np.all(V_diag < 0)

    print(f"\n   Kinetic diagonal range: [{T_diag.min():.6f}, {T_diag.max():.6f}]")
    print(f"   All T_ii > 0: {T_positive}")

    print(f"\n   Nuclear diagonal range: [{V_diag.min():.6f}, {V_diag.max():.6f}]")
    print(f"   All V_ii < 0: {V_negative}")

    if T_positive and V_negative:
        print("\n   PASS: Sign conventions are correct")
    else:
        print("\n   FAIL: Check sign conventions!")

    # =========================================================================
    # STEP 5: Matrix norms and relative magnitudes
    # =========================================================================
    # Matrix norms give a sense of overall magnitude:
    # - Frobenius norm: ||M||_F = sqrt(sum_ij |M_ij|^2)
    # - For bound systems, |V| > |T| typically (virial theorem at equilibrium)
    #
    print_section("4. MATRIX NORMS (Frobenius)")

    norm_S = np.linalg.norm(S, 'fro')
    norm_T = np.linalg.norm(T, 'fro')
    norm_V = np.linalg.norm(V, 'fro')
    norm_h = np.linalg.norm(h, 'fro')

    print(f"\n   ||S||_F = {norm_S:12.6f}")
    print(f"   ||T||_F = {norm_T:12.6f}")
    print(f"   ||V||_F = {norm_V:12.6f}")
    print(f"   ||h||_F = {norm_h:12.6f}")

    print("\n   Ratios:")
    print(f"   ||V|| / ||T|| = {norm_V/norm_T:.3f}")
    print(f"   (Ratio > 1 indicates nuclear attraction dominates)")
    print(f"   ||h|| / ||T|| = {norm_h/norm_T:.3f}")
    print(f"   (Should be < 1 since V and T have opposite signs)")

    # =========================================================================
    # STEP 6: Diagonal element analysis (core vs valence)
    # =========================================================================
    print_section("5. DIAGONAL ELEMENT ANALYSIS")

    print("\n   The diagonal elements reveal core vs valence structure:")
    print("   - Core 1s on O: Large |T| and |V| (tightly bound)")
    print("   - Valence 2s, 2p on O: Medium values")
    print("   - H 1s: Smaller values (less tightly bound)")

    analyze_diagonal_pattern(T, "Kinetic T", ao_labels)
    analyze_diagonal_pattern(V, "Nuclear V", ao_labels)

    # =========================================================================
    # STEP 7: Off-diagonal structure
    # =========================================================================
    print_section("6. OFF-DIAGONAL STRUCTURE")

    print("\n   Off-diagonal elements measure coupling between AOs:")
    print("   - Large off-diagonal: Strong interaction (spatial overlap)")
    print("   - Small off-diagonal: Weak interaction (distant or orthogonal)")

    analyze_off_diagonal_decay(S, "Overlap S")
    analyze_off_diagonal_decay(T, "Kinetic T")
    analyze_off_diagonal_decay(V, "Nuclear V")

    # =========================================================================
    # STEP 8: Eigenvalue analysis of overlap matrix
    # =========================================================================
    # The overlap matrix S must be positive definite for a valid basis:
    #   - All eigenvalues > 0
    #   - Condition number = max(eig) / min(eig) measures linear dependence
    #   - Large condition number = near-linear dependence = numerical trouble
    #
    print_section("7. OVERLAP MATRIX EIGENVALUE ANALYSIS")

    eig_S = np.linalg.eigvalsh(S)
    eig_S_sorted = np.sort(eig_S)

    print(f"\n   Overlap eigenvalues (sorted):")
    for i, ev in enumerate(eig_S_sorted):
        print(f"      lambda_{i} = {ev:.10f}")

    cond_S = eig_S_sorted[-1] / eig_S_sorted[0]
    print(f"\n   Condition number: kappa(S) = {cond_S:.2f}")
    print(f"   All eigenvalues positive: {np.all(eig_S > 0)}")

    print("\n   Interpretation:")
    print("   - Condition number < 100: Well-conditioned (safe)")
    print("   - 100 < kappa < 10000: Moderate conditioning (caution)")
    print("   - kappa > 10000: Ill-conditioned (may need threshold)")

    if cond_S < 100:
        print(f"\n   STO-3G basis is well-conditioned (kappa = {cond_S:.1f})")

    # =========================================================================
    # STEP 9: Electron count verification
    # =========================================================================
    # The trace of P*S gives the number of electrons:
    #   Tr[P*S] = sum_uv P_uv S_vu = sum_uv P_uv S_uv = N_elec
    #
    # This is a fundamental consistency check for the density matrix.
    #
    print_section("8. ELECTRON COUNT VERIFICATION")

    print("\n   Running RHF to obtain density matrix P...")
    mf = scf.RHF(mol)
    mf.verbose = 0
    E_hf = mf.kernel()
    P = mf.make_rdm1()

    print(f"   RHF energy: {E_hf:.10f} Hartree")

    # Compute electron count: Tr[P*S]
    n_elec_computed = np.trace(P @ S)
    n_elec_expected = mol.nelectron

    print(f"\n   Expected electrons: {n_elec_expected}")
    print(f"   Tr[P*S] = {n_elec_computed:.10f}")
    print(f"   Difference: {abs(n_elec_computed - n_elec_expected):.2e}")

    if np.isclose(n_elec_computed, n_elec_expected, atol=1e-10):
        print("\n   PASS: Electron count is correct")
    else:
        print("\n   FAIL: Electron count mismatch!")

    # =========================================================================
    # STEP 10: Physical interpretation summary
    # =========================================================================
    print_section("9. PHYSICAL INTERPRETATION SUMMARY")

    print("""
   OVERLAP MATRIX S:
   -----------------
   - S_uu = 1: Each AO is normalized
   - S_uv measures spatial overlap between AOs u and v
   - Large |S_uv|: Orbitals share significant space
   - S is the metric tensor for the non-orthogonal AO basis

   KINETIC MATRIX T:
   -----------------
   - T_uu > 0: Kinetic energy is always positive
   - Large T: Rapid spatial variation (tightly bound orbitals)
   - Small T: Slowly varying (diffuse orbitals)
   - Core 1s orbitals have largest kinetic energy

   NUCLEAR ATTRACTION V:
   ---------------------
   - V_uu < 0: Electrons are attracted to nuclei
   - Large |V|: Electron density close to nuclei
   - Core orbitals have strongest nuclear attraction
   - V contains contributions from ALL nuclei

   CORE HAMILTONIAN h = T + V:
   ---------------------------
   - Describes one electron in the nuclear potential
   - Eigenvalues approximate orbital energies (crude)
   - Used as initial guess for SCF iterations

   KEY RELATIONSHIPS:
   ------------------
   - Virial theorem at equilibrium: 2<T> + <V> = 0
   - This means |V|/|T| ~ 2 for bound systems
   - Electron count: Tr[P*S] = N_electrons
    """)

    print("=" * 70)
    print("Lab 3A Solution Complete")
    print("=" * 70)

    # Return key results for testing
    return {
        'S': S,
        'T': T,
        'V': V,
        'h': h,
        'P': P,
        'E_hf': E_hf,
        'nao': nao,
        'n_elec': n_elec_computed
    }


if __name__ == "__main__":
    results = main()
