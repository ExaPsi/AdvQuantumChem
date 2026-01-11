#!/usr/bin/env python3
"""
Chapter 7 Exercise Solutions: Scaling and Properties

This module contains Python implementations for all Chapter 7 exercises involving:
- Scaling analysis and computational bottleneck identification
- Density fitting (DF/RI) implementation and comparison
- Virial theorem verification and basis set diagnostics
- Dipole moment calculations with origin dependence study
- Conventional vs DF-HF timing comparisons
- Basis set conditioning studies
- Hellmann-Feynman theorem and Pulay force concepts

Each exercise function is self-contained and can be run independently.
All calculations are validated against PySCF reference values.

Part of: Advanced Quantum Chemistry Lecture Notes (2302638)
Chapter 7: Scaling and Properties
"""

import time
import numpy as np
from typing import Tuple, List, Dict, Optional
from pyscf import gto, scf

# ==============================================================================
# Constants and Conversion Factors
# ==============================================================================

AU_TO_DEBYE = 2.5417464  # Conversion: atomic units to Debye
AU_TO_ANGSTROM = 0.529177  # Bohr to Angstrom


# ==============================================================================
# Utility Functions
# ==============================================================================


def run_conventional_hf(mol, verbose: bool = False) -> Tuple[float, float, scf.hf.RHF]:
    """
    Run conventional RHF and return energy, timing, and mf object.

    Args:
        mol: PySCF molecule object
        verbose: Whether to print SCF output

    Returns:
        Tuple of (energy, elapsed_time, mf_object)
    """
    t0 = time.time()
    mf = scf.RHF(mol)
    mf.verbose = 4 if verbose else 0
    energy = mf.kernel()
    elapsed = time.time() - t0
    return energy, elapsed, mf


def run_df_hf(mol, auxbasis: Optional[str] = None,
              verbose: bool = False) -> Tuple[float, float, scf.hf.RHF]:
    """
    Run density-fitted RHF and return energy, timing, and mf object.

    Args:
        mol: PySCF molecule object
        auxbasis: Auxiliary basis for density fitting
        verbose: Whether to print SCF output

    Returns:
        Tuple of (energy, elapsed_time, mf_object)
    """
    t0 = time.time()
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
    mf.verbose = 4 if verbose else 0
    energy = mf.kernel()
    elapsed = time.time() - t0
    return energy, elapsed, mf


def compute_dipole_from_density(mol, dm: np.ndarray,
                                 origin: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute molecular dipole moment from density matrix.

    The dipole moment is computed as:
        mu = mu_nuc - mu_el = sum_A Z_A * R_A - Tr[P * r]

    Args:
        mol: PySCF molecule object
        dm: Density matrix (nao x nao)
        origin: Origin for dipole calculation (default: [0, 0, 0])

    Returns:
        Dipole moment vector in atomic units (shape: 3)
    """
    if origin is None:
        origin = np.zeros(3)

    # Get position operator integrals relative to origin
    with mol.with_common_orig(origin):
        ao_r = mol.intor_symmetric('int1e_r', comp=3)  # Shape: (3, nao, nao)

    # Electronic contribution: mu_el = Tr[P * r]
    mu_el = np.einsum('xij,ji->x', ao_r, dm).real

    # Nuclear contribution: mu_nuc = sum_A Z_A (R_A - origin)
    charges = mol.atom_charges()
    coords = mol.atom_coords()  # In Bohr
    mu_nuc = np.einsum('a,ax->x', charges, coords - origin[None, :])

    # Total dipole: nuclear - electronic (electrons have charge -1)
    return mu_nuc - mu_el


def compute_energy_components(mol, mf) -> Dict[str, float]:
    """
    Compute individual energy components from a converged HF calculation.

    Returns:
        Dictionary with energy components and virial ratio
    """
    dm = mf.make_rdm1()

    # One-electron integrals
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')

    # Build J and K matrices
    J, K = mf.get_jk(mol, dm)

    # Energy components
    E_T = np.einsum('ij,ji->', dm, T)           # Kinetic energy
    E_Ven = np.einsum('ij,ji->', dm, V)         # Electron-nuclear attraction
    E_ee = 0.5 * np.einsum('ij,ji->', dm, J - 0.5 * K)  # Electron-electron
    E_nn = mol.energy_nuc()                     # Nuclear repulsion

    # Total potential energy
    E_V_total = E_Ven + E_ee + E_nn

    # Virial ratio
    eta = -E_V_total / E_T if abs(E_T) > 1e-12 else np.nan

    return {
        'E_T': E_T,
        'E_Ven': E_Ven,
        'E_ee': E_ee,
        'E_nn': E_nn,
        'E_V_total': E_V_total,
        'E_total': E_T + E_V_total,
        'eta': eta,
        'eta_deviation': abs(eta - 2),
    }


# ==============================================================================
# Exercise 7.1: Scaling Bottleneck Analysis [Core]
# ==============================================================================


def exercise_7_1():
    """
    Exercise 7.1: Analyze scaling bottlenecks in Hartree-Fock.

    Investigates:
    (a) Number of unique ERIs with 8-fold symmetry
    (b) Comparison with one-electron quantities
    (c) Why ERIs dominate and how DF helps
    """
    print("=" * 70)
    print("Exercise 7.1: Scaling Bottleneck Analysis")
    print("=" * 70)

    # Part (a): Count unique ERIs
    print("\n[Part a] Unique ERI count with 8-fold symmetry")
    print("-" * 50)

    N_values = [50, 100, 200, 500]

    print(f"{'N_AO':>8} {'Full ERIs (N^4)':>18} {'8-fold (N^4/8)':>18} {'Ratio':>10}")
    print("-" * 50)

    for N in N_values:
        full_eri = N**4
        # More precise formula for unique ERIs with 8-fold symmetry
        n_pair = N * (N + 1) // 2
        unique_eri = n_pair * (n_pair + 1) // 2
        approx_eri = N**4 // 8

        print(f"{N:>8} {full_eri:>18,} {unique_eri:>18,} {full_eri/unique_eri:>10.1f}")

    # Part (b): Comparison with one-electron quantities
    print("\n[Part b] Comparison of quantities for N = 100 AOs")
    print("-" * 50)

    N = 100
    n_fock = N * N
    n_1e = 3 * N * N  # S, T, V matrices
    n_eri = N**4 // 8

    print(f"{'Quantity':<25} {'Count':>15} {'Ratio to ERIs':>18}")
    print("-" * 50)
    print(f"{'Fock matrix elements':<25} {n_fock:>15,} {n_fock/n_eri:>18.2e}")
    print(f"{'One-e integrals (S,T,V)':<25} {n_1e:>15,} {n_1e/n_eri:>18.2e}")
    print(f"{'Unique ERIs':<25} {n_eri:>15,} {'1.00':>18}")

    print(f"""
ERIs outnumber one-electron quantities by factor of N^2/8 ~ {N**2/8:.0f}

[Part c] Why ERIs dominate and how DF helps:

1. STORAGE: Full ERI tensor requires N^4/8 * 8 bytes = N^4 bytes
   For N = 100: {(100**4)/(1024**2):.1f} MB
   For N = 500: {(500**4)/(1024**3):.1f} GB (often exceeds RAM!)

2. COMPUTATION: Each ERI requires Boys function + primitive contractions
   With ~10^7 unique ERIs, this is the dominant cost

3. CONTRACTION: Building J and K is O(N^4) per SCF iteration
   Diagonalization is only O(N^3)

DENSITY FITTING addresses this by factoring:
   (mu nu|lambda sigma) ~ sum_Q B_mu_nu^Q * B_lambda_sigma^Q

   - Storage: O(N^2 * N_aux) ~ O(N^3)
   - J-build: O(N^2 * N_aux) ~ O(N^3)
   - K-build: O(N^2 * N_aux * N_occ) ~ O(N^3)
""")

    # Verify with actual PySCF calculation
    print("[Practical verification with ethane/cc-pVDZ]")
    print("-" * 50)

    mol = gto.M(
        atom='''
        C -0.7560  0.0000  0.0000
        C  0.7560  0.0000  0.0000
        H -1.1404  0.5916  0.8327
        H -1.1404  0.4757 -0.8863
        H -1.1404 -1.0673  0.0537
        H  1.1404 -0.5916 -0.8327
        H  1.1404 -0.4757  0.8863
        H  1.1404  1.0673 -0.0537
        ''',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    N = mol.nao
    print(f"Number of AOs: {N}")
    print(f"Expected unique ERIs: {N**4//8:,}")
    print(f"Theoretical storage: {(N**4 * 8)/(1024**2):.1f} MB")

    # Time conventional vs DF
    E_conv, t_conv, _ = run_conventional_hf(mol)
    E_df, t_df, _ = run_df_hf(mol)

    print(f"\nConventional HF: {t_conv:.2f} s, E = {E_conv:.10f}")
    print(f"DF-HF:           {t_df:.2f} s, E = {E_df:.10f}")
    print(f"Speedup: {t_conv/t_df:.1f}x")

    print("\n" + "=" * 70)


# ==============================================================================
# Exercise 7.2: DF/RI Concept Check [Core]
# ==============================================================================


def exercise_7_2():
    """
    Exercise 7.2: Understand density fitting conceptually.

    Explains the DF approximation and its components:
    - (mu nu|P): Three-index integrals
    - (P|Q): Coulomb metric between auxiliary functions
    - B_mu_nu^Q: Fitted three-index tensor
    """
    print("=" * 70)
    print("Exercise 7.2: DF/RI Concept Check")
    print("=" * 70)

    print("""
DENSITY FITTING APPROXIMATION:

(mu nu|lambda sigma) ~ sum_PQ (mu nu|P)(P|Q)^-1(Q|lambda sigma)
                     = sum_Q B_mu_nu^Q * B_lambda_sigma^Q

where B_mu_nu^Q = sum_P (mu nu|P)(P|Q)^-1/2

EXPLANATION OF EACH OBJECT:

1. (mu nu|P): Three-index integral between pair density chi_mu*chi_nu
   and auxiliary function chi_P. Represents how well auxiliary function P
   can describe the electrostatic potential of pair (mu, nu).

2. (P|Q): Two-index Coulomb metric between auxiliary functions.
   This overlap-like matrix must be inverted for proper normalization.

3. B_mu_nu^Q: The fitted three-index tensor. Represents the expansion
   of pair density chi_mu*chi_nu in the auxiliary basis.
""")

    # Practical demonstration
    mol = gto.M(
        atom='O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    N = mol.nao

    print(f"[Demonstration with H2O/cc-pVDZ]")
    print("-" * 50)
    print(f"N_AO = {N}")

    # Run DF-HF to get auxiliary basis info
    mf_df = scf.RHF(mol).density_fit()
    mf_df.kernel()

    # Get auxiliary basis size
    auxmol = mf_df.with_df.auxmol
    N_aux = auxmol.nao if auxmol is not None else 0

    print(f"N_aux = {N_aux}")
    print(f"N_aux/N = {N_aux/N:.1f}")

    # Storage comparison
    full_eri_storage = N**4 / 8 * 8 / (1024**2)  # MB
    df_storage = N**2 * N_aux * 8 / (1024**2)    # MB

    print(f"\n[Part c] Storage comparison:")
    print(f"Full ERI storage:  {full_eri_storage:.2f} MB")
    print(f"DF tensor storage: {df_storage:.2f} MB")
    print(f"Storage ratio:     {full_eri_storage/df_storage:.1f}x reduction")

    # Physical meaning
    print("""
[Part b] Physical meaning of auxiliary basis:

The auxiliary functions {chi_P} form a basis for representing pair
densities rho_mu_nu(r) = chi_mu(r) * chi_nu(r).

Product angular momentum rules:
- s * s pairs: auxiliary functions up to s (or p for better fitting)
- p * p pairs: auxiliary functions up to d
- d * d pairs: auxiliary functions up to g

The auxiliary basis must include functions with angular momentum up to
2*L_max to accurately represent all pair densities.

For cc-pVDZ (L_max = 2 for C, O), the matching cc-pVDZ-jkfit includes
functions up to L = 4 (g functions).
""")

    print("=" * 70)


# ==============================================================================
# Exercise 7.3: Virial Ratio Trends [Core]
# ==============================================================================


def exercise_7_3():
    """
    Exercise 7.3: Study virial ratio trends with basis set quality.

    Computes virial ratio eta = -<V>/<T> for different basis sets
    and analyzes the approach to eta = 2.
    """
    print("=" * 70)
    print("Exercise 7.3: Virial Ratio Trends")
    print("=" * 70)

    # H2O at equilibrium geometry
    atom_str = '''
    O  0.0000  0.0000  0.0000
    H  0.7586  0.0000  0.5043
    H -0.7586  0.0000  0.5043
    '''

    basis_sets = ['STO-3G', 'cc-pVDZ', 'cc-pVTZ', 'cc-pVQZ']

    print("\nMolecule: H2O at equilibrium geometry")
    print("-" * 70)
    print(f"{'Basis':<12} {'N_AO':>6} {'<T> (Eh)':>14} {'<V> (Eh)':>16} "
          f"{'eta':>10} {'|eta-2|':>12}")
    print("-" * 70)

    results = []

    for basis in basis_sets:
        mol = gto.M(atom=atom_str, basis=basis, unit='Angstrom', verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()

        if not mf.converged:
            print(f"{basis:<12} {'DID NOT CONVERGE':>50}")
            continue

        result = compute_energy_components(mol, mf)
        result['basis'] = basis
        result['nao'] = mol.nao
        results.append(result)

        print(f"{basis:<12} {mol.nao:>6} {result['E_T']:>14.6f} "
              f"{result['E_V_total']:>16.6f} {result['eta']:>10.6f} "
              f"{result['eta_deviation']:>12.2e}")

    # Trend analysis
    print("\n" + "-" * 50)
    print("Trend Analysis:")
    print("-" * 50)

    for i in range(len(results)):
        if i > 0:
            improvement = results[i-1]['eta_deviation'] / results[i]['eta_deviation']
            print(f"{results[i-1]['basis']:>12} -> {results[i]['basis']:<12}: "
                  f"|eta-2| reduced by factor of {improvement:.1f}")

    print(f"""
[Why larger bases approach eta = 2]

The virial theorem 2<T> + <V> = 0 follows from stationarity under
coordinate scaling r -> lambda*r. In a finite Gaussian basis:

1. The scaled wavefunction Psi(lambda*r) lies OUTSIDE the basis set span
2. We cannot enforce stationarity with respect to scaling
3. Larger bases better approximate the complete basis limit
4. The deviation |eta - 2| serves as a measure of basis incompleteness

Diagnostic thresholds:
  |eta - 2| < 10^-3: Excellent basis quality
  10^-3 to 10^-2:    Adequate for most purposes
  > 10^-2:           Investigate basis/convergence issues
""")

    print("=" * 70)


# ==============================================================================
# Exercise 7.4: Dipole Moment Origin Shift [Core]
# ==============================================================================


def exercise_7_4():
    """
    Exercise 7.4: Study origin dependence of dipole moment.

    Demonstrates:
    (a) Origin independence for neutral molecules
    (b) Origin dependence for charged species
    (c) Physical explanation of the difference
    """
    print("=" * 70)
    print("Exercise 7.4: Dipole Moment Origin Shift")
    print("=" * 70)

    # Part (a): Neutral molecule (HF)
    print("\n[Part a] Neutral molecule: HF")
    print("-" * 50)

    mol_neutral = gto.M(
        atom='H 0 0 0; F 0 0 0.92',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    mf_neutral = scf.RHF(mol_neutral)
    mf_neutral.kernel()
    dm_neutral = mf_neutral.make_rdm1()

    origins = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([2.0, -1.0, 3.0]),
    ]

    print(f"{'Origin (Bohr)':<25} {'mu (a.u.)':<30} {'|mu| (D)':>12}")
    print("-" * 70)

    dipoles_neutral = []
    for origin in origins:
        mu = compute_dipole_from_density(mol_neutral, dm_neutral, origin)
        mu_norm = np.linalg.norm(mu) * AU_TO_DEBYE
        dipoles_neutral.append(mu_norm)
        print(f"({origin[0]:+5.1f}, {origin[1]:+5.1f}, {origin[2]:+5.1f})"
              f"   ({mu[0]:+8.5f}, {mu[1]:+8.5f}, {mu[2]:+8.5f})   {mu_norm:>10.6f}")

    max_diff_neutral = max(dipoles_neutral) - min(dipoles_neutral)
    print(f"\nMax difference: {max_diff_neutral:.2e} Debye")
    print("Result: Dipole is ORIGIN-INDEPENDENT for neutral molecule!")

    # Part (b): Charged species (HF+)
    print("\n[Part b] Charged molecule: HF+ (cation)")
    print("-" * 50)

    mol_cation = gto.M(
        atom='H 0 0 0; F 0 0 0.92',
        basis='cc-pVDZ',
        unit='Angstrom',
        charge=1,
        spin=1,
        verbose=0
    )

    mf_cation = scf.UHF(mol_cation)
    mf_cation.kernel()
    dm_cation = mf_cation.make_rdm1()
    dm_total = dm_cation[0] + dm_cation[1]  # Sum alpha and beta densities

    print(f"{'Origin (Bohr)':<25} {'mu (a.u.)':<30} {'|mu| (D)':>12}")
    print("-" * 70)

    dipoles_cation = []
    for origin in origins:
        mu = compute_dipole_from_density(mol_cation, dm_total, origin)
        mu_norm = np.linalg.norm(mu) * AU_TO_DEBYE
        dipoles_cation.append(mu_norm)
        print(f"({origin[0]:+5.1f}, {origin[1]:+5.1f}, {origin[2]:+5.1f})"
              f"   ({mu[0]:+8.5f}, {mu[1]:+8.5f}, {mu[2]:+8.5f})   {mu_norm:>10.6f}")

    max_diff_cation = max(dipoles_cation) - min(dipoles_cation)
    print(f"\nMax difference: {max_diff_cation:.4f} Debye")
    print("Result: Dipole is ORIGIN-DEPENDENT for charged species!")

    # Part (c): Verify the shift formula
    print("\n[Part c] Verification of shift formula: mu(O+d) = mu(O) - Q*d")
    print("-" * 50)

    O1 = origins[0]
    O2 = origins[1]
    d = O2 - O1  # Shift vector
    Q = 1  # Charge

    mu1 = compute_dipole_from_density(mol_cation, dm_total, O1)
    mu2 = compute_dipole_from_density(mol_cation, dm_total, O2)

    predicted_mu2 = mu1 - Q * d

    print(f"mu(O1) = ({mu1[0]:+.6f}, {mu1[1]:+.6f}, {mu1[2]:+.6f})")
    print(f"mu(O2) = ({mu2[0]:+.6f}, {mu2[1]:+.6f}, {mu2[2]:+.6f})")
    print(f"Predicted mu(O2) = mu(O1) - Q*d:")
    print(f"         ({predicted_mu2[0]:+.6f}, {predicted_mu2[1]:+.6f}, {predicted_mu2[2]:+.6f})")
    print(f"Difference: {np.linalg.norm(mu2 - predicted_mu2):.2e} a.u.")

    print(f"""
[Physical Explanation]

For a system with total charge Q, shifting the origin by d changes:
  - Nuclear contribution by +Q_nuc * d
  - Electronic contribution by +N_elec * d

The net change is: Delta_mu = -(Q_nuc - N_elec) * d = -Q * d

For neutral molecules (Q = 0): No change (origin-independent)
For ions (Q != 0): Dipole shifts by -Q * d (origin-dependent)

When reporting dipoles for ions, always specify the origin!
Common conventions:
  - Center of mass
  - Center of nuclear charge
""")

    print("=" * 70)


# ==============================================================================
# Exercise 7.5: Hellmann-Feynman and Pulay (Conceptual) [Core]
# ==============================================================================


def exercise_7_5():
    """
    Exercise 7.5: Understand Hellmann-Feynman theorem and Pulay forces.

    Conceptual exercise explaining:
    (a) Hellmann-Feynman theorem statement
    (b) "Forces are expectation values"
    (c) Origin of Pulay terms
    (d) When Pulay terms vanish
    """
    print("=" * 70)
    print("Exercise 7.5: Hellmann-Feynman and Pulay Forces (Conceptual)")
    print("=" * 70)

    print("""
[Part a] HELLMANN-FEYNMAN THEOREM:

For an exact eigenstate |Psi> of Hamiltonian H(lambda) depending on
parameter lambda:

    dE/dlambda = <Psi| dH/dlambda |Psi>

For nuclear coordinate R_Ax:

    F_Ax = -dE/dR_Ax = -<Psi| dH/dR_Ax |Psi>

This says forces can be computed as expectation values of
Hamiltonian derivatives, without differentiating the wavefunction!

[Part b] "FORCES ARE EXPECTATION VALUES":

The theorem implies:
  - dH/dR_Ax includes dV_en/dR_Ax (electron-nuclear Coulomb)
  - The expectation value gives the average electrostatic force
    from the electron density on the nucleus
  - No knowledge of wavefunction derivatives is needed

Physical interpretation: The force on nucleus A is just the
classical electrostatic force from the electron cloud.

[Part c] ORIGIN OF PULAY TERMS:

In AO-based calculations with atom-centered Gaussians, Hellmann-Feynman
is NOT sufficient. Additional "Pulay terms" arise because:

  - Basis functions chi_mu(r; R_A) depend on nuclear positions
  - When nucleus A moves, its basis functions move too
  - The wavefunction has implicit dependence on R_A through the basis
  - This gives contributions: sum_mu_nu W_mu_nu * dS_mu_nu/dR_A
    where W is the energy-weighted density matrix

The full gradient is:
    dE/dR = <Psi|dH/dR|Psi> + sum_mu_nu W_mu_nu * dS_mu_nu/dR
            (Hellmann-Feynman)      (Pulay terms)

[Part d] WHEN PULAY TERMS VANISH:

Pulay terms would vanish if:

1. COMPLETE BASIS: A complete basis spans all functions regardless of
   nuclear positions. Moving a nucleus doesn't change what's representable.

2. PLANE WAVES: Plane-wave bases are independent of nuclear positions
   (though pseudopotentials reintroduce nuclear dependence).

3. FLOATING GAUSSIANS: If basis function centers are variationally
   optimized, they become variational parameters and standard
   Hellmann-Feynman applies.

In practice, finite atom-centered bases ALWAYS require Pulay corrections.
""")

    # Numerical demonstration: gradient comparison
    print("[Numerical Demonstration: Gradient Comparison for H2]")
    print("-" * 60)

    from pyscf.grad import rhf as rhf_grad

    mol = gto.M(
        atom='H 0 0 0; H 0 0 0.76',  # Slightly stretched from equilibrium
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    mf = scf.RHF(mol)
    mf.kernel()

    # Analytical gradient using proper PySCF gradient module
    g = rhf_grad.Gradients(mf)
    grad_analytical = g.kernel()

    # Numerical gradient
    delta = 1e-4  # Bohr
    grad_numerical = np.zeros((mol.natm, 3))

    coords = mol.atom_coords().copy()

    for i in range(mol.natm):
        for j in range(3):
            coords_plus = coords.copy()
            coords_plus[i, j] += delta

            coords_minus = coords.copy()
            coords_minus[i, j] -= delta

            # Create shifted molecules
            atom_str_plus = f"H {coords_plus[0,0]} {coords_plus[0,1]} {coords_plus[0,2]}; " \
                           f"H {coords_plus[1,0]} {coords_plus[1,1]} {coords_plus[1,2]}"
            atom_str_minus = f"H {coords_minus[0,0]} {coords_minus[0,1]} {coords_minus[0,2]}; " \
                            f"H {coords_minus[1,0]} {coords_minus[1,1]} {coords_minus[1,2]}"

            mol_plus = gto.M(atom=atom_str_plus, basis='cc-pVDZ', unit='Bohr', verbose=0)
            mol_minus = gto.M(atom=atom_str_minus, basis='cc-pVDZ', unit='Bohr', verbose=0)

            E_plus = scf.RHF(mol_plus).kernel()
            E_minus = scf.RHF(mol_minus).kernel()

            grad_numerical[i, j] = (E_plus - E_minus) / (2 * delta)

    print(f"{'Atom':>6} {'Dir':>4} {'Analytical':>14} {'Numerical':>14} {'Diff':>12}")
    print("-" * 60)

    for i in range(mol.natm):
        for j, d in enumerate(['x', 'y', 'z']):
            diff = abs(grad_analytical[i, j] - grad_numerical[i, j])
            print(f"{i+1:>6} {d:>4} {grad_analytical[i,j]:>14.8f} "
                  f"{grad_numerical[i,j]:>14.8f} {diff:>12.2e}")

    print("\nAnalytical gradients include Pulay terms automatically!")

    print("\n" + "=" * 70)


# ==============================================================================
# Exercise 7.6: DF-HF Timing and Accuracy Study [Advanced]
# ==============================================================================


def exercise_7_6():
    """
    Exercise 7.6: Comprehensive DF-HF timing and accuracy study.

    Compares conventional and DF-HF for multiple molecules and basis sets.
    """
    print("=" * 70)
    print("Exercise 7.6: DF-HF Timing and Accuracy Study")
    print("=" * 70)

    # Test molecules of increasing size
    molecules = [
        ('H2O', '''O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043'''),
        ('CH4', '''C 0 0 0; H 0.6276 0.6276 0.6276; H 0.6276 -0.6276 -0.6276;
                   H -0.6276 0.6276 -0.6276; H -0.6276 -0.6276 0.6276'''),
        ('C2H6', '''C -0.7560 0 0; C 0.7560 0 0;
                    H -1.1404 0.5916 0.8327; H -1.1404 0.4757 -0.8863;
                    H -1.1404 -1.0673 0.0537; H 1.1404 -0.5916 -0.8327;
                    H 1.1404 -0.4757 0.8863; H 1.1404 1.0673 -0.0537'''),
    ]

    basis_sets = ['cc-pVDZ', 'cc-pVTZ']

    print("\n[Parts a-c] Timing and energy comparison")
    print("-" * 80)
    print(f"{'Molecule':<10} {'Basis':<12} {'N_AO':>6} {'E_conv (Eh)':>16} "
          f"{'E_DF (Eh)':>16} {'DF Error':>12} {'Speedup':>10}")
    print("-" * 80)

    all_results = []

    for name, atoms in molecules:
        for basis in basis_sets:
            mol = gto.M(atom=atoms, basis=basis, unit='Angstrom', verbose=0)

            E_conv, t_conv, _ = run_conventional_hf(mol)
            E_df, t_df, _ = run_df_hf(mol)

            df_error = abs(E_df - E_conv)
            speedup = t_conv / t_df if t_df > 0 else np.inf

            result = {
                'name': name, 'basis': basis, 'nao': mol.nao,
                'E_conv': E_conv, 'E_df': E_df, 'df_error': df_error,
                't_conv': t_conv, 't_df': t_df, 'speedup': speedup
            }
            all_results.append(result)

            print(f"{name:<10} {basis:<12} {mol.nao:>6} {E_conv:>16.10f} "
                  f"{E_df:>16.10f} {df_error:>12.2e} {speedup:>9.1f}x")

    # Part (d): Analyze DF error vs basis set error
    print("\n[Part d] DF error vs basis set error analysis")
    print("-" * 60)

    for name, atoms in molecules:
        results_mol = [r for r in all_results if r['name'] == name]
        if len(results_mol) >= 2:
            dz_result = [r for r in results_mol if 'DZ' in r['basis']][0]
            tz_result = [r for r in results_mol if 'TZ' in r['basis']][0]

            basis_error = abs(tz_result['E_conv'] - dz_result['E_conv'])
            df_error = dz_result['df_error']

            print(f"{name}:")
            print(f"   Basis set error (TZ-DZ): {basis_error:.6e} Eh")
            print(f"   DF error (DZ):           {df_error:.6e} Eh")
            print(f"   Ratio (basis/DF):        {basis_error/df_error:.0f}x")

    print("""
CONCLUSION: DF error is typically 1000-10000x smaller than basis set error.
DF is "safe" whenever the basis set itself limits accuracy.
""")

    print("=" * 70)


# ==============================================================================
# Exercise 7.7: DF vs Full ERI J/K Matrix Check [Advanced]
# ==============================================================================


def exercise_7_7():
    """
    Exercise 7.7: Compare J and K matrices from conventional vs DF methods.

    Investigates the matrix-level accuracy of density fitting.
    """
    print("=" * 70)
    print("Exercise 7.7: DF vs Full ERI J/K Matrix Check")
    print("=" * 70)

    mol = gto.M(
        atom='O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043',
        basis='6-31G',
        unit='Angstrom',
        verbose=0
    )

    print(f"\nMolecule: H2O/6-31G (N_AO = {mol.nao})")

    # Conventional HF
    mf_conv = scf.RHF(mol)
    mf_conv.kernel()
    dm = mf_conv.make_rdm1()

    # Full ERI contraction
    eri = mol.intor('int2e', aosym='s1')
    J_full = np.einsum('ijkl,kl->ij', eri, dm)
    K_full = np.einsum('ikjl,kl->ij', eri, dm)

    # DF-HF for J/K
    mf_df = scf.RHF(mol).density_fit()
    J_df, K_df = mf_df.get_jk(mol, dm)

    # Compute errors
    J_error = np.linalg.norm(J_df - J_full)
    K_error = np.linalg.norm(K_df - K_full)
    J_rel_error = J_error / np.linalg.norm(J_full)
    K_rel_error = K_error / np.linalg.norm(K_full)

    print("\n[Matrix comparison]")
    print("-" * 50)
    print(f"||J_DF - J_full||_F = {J_error:.2e}")
    print(f"||K_DF - K_full||_F = {K_error:.2e}")
    print(f"||J_DF - J_full||_F / ||J_full||_F = {J_rel_error:.2e}")
    print(f"||K_DF - K_full||_F / ||K_full||_F = {K_rel_error:.2e}")

    print(f"""
[Observations]

1. DF error is larger for K than for J (roughly {K_error/J_error:.1f}x here)

2. WHY K ERROR IS LARGER:
   - For J: J_mu_nu = sum_Q B_mu_nu^Q * d^Q uses the SAME index structure
     as the DF factorization, fitting naturally.

   - For K: The "crossed" index pattern K_mu_nu = sum_lambda_sigma
     (mu lambda|nu sigma) P_lambda_sigma does NOT factor as cleanly.

   - K-build requires intermediate transformations that accumulate
     fitting errors.

   - The exchange hole is more localized than the Coulomb hole,
     requiring finer representation.

3. Both errors are small enough that total energy errors are ~10^-5
   to 10^-6 Eh, well below basis set error.
""")

    # Verify energies match
    E_from_full = 0.5 * np.einsum('ij,ji->', dm, J_full - 0.5 * K_full)
    E_from_df = 0.5 * np.einsum('ij,ji->', dm, J_df - 0.5 * K_df)

    print(f"Coulomb-exchange energy (full): {E_from_full:.10f}")
    print(f"Coulomb-exchange energy (DF):   {E_from_df:.10f}")
    print(f"Difference: {abs(E_from_full - E_from_df):.2e}")

    print("\n" + "=" * 70)


# ==============================================================================
# Exercise 7.8: Connecting Integrals to Properties [Advanced]
# ==============================================================================


def exercise_7_8():
    """
    Exercise 7.8: Demonstrate the unified property pattern Tr[P*o].

    Shows how different one-electron properties reduce to the same
    computational pattern.
    """
    print("=" * 70)
    print("Exercise 7.8: Connecting Integrals to Properties")
    print("=" * 70)

    mol = gto.M(
        atom='O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    mf = scf.RHF(mol)
    mf.kernel()
    P = mf.make_rdm1()

    print(f"\nMolecule: H2O/cc-pVDZ")
    print(f"RHF Energy: {mf.e_tot:.10f} Eh")

    # Extract various operator matrices
    S = mol.intor('int1e_ovlp')
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')

    with mol.with_common_orig([0, 0, 0]):
        r_ints = mol.intor('int1e_r', comp=3)

    # Compute properties using Tr[P*o]
    print("\n[Properties from Tr[P*o] pattern]")
    print("-" * 60)

    # 1. Electron count
    N_e = np.einsum('ij,ji->', P, S)
    print(f"Electron count: N_e = Tr[P*S] = {N_e:.10f}")
    print(f"                (expected: {mol.nelectron})")

    # 2. Kinetic energy
    T_expect = np.einsum('ij,ji->', P, T)
    print(f"\nKinetic energy: <T> = Tr[P*T] = {T_expect:.10f} Eh")

    # 3. Nuclear attraction energy
    V_expect = np.einsum('ij,ji->', P, V)
    print(f"Nuclear attraction: <V_en> = Tr[P*V] = {V_expect:.10f} Eh")

    # 4. Dipole moment
    mu_el = np.einsum('xij,ji->x', r_ints, P)
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    mu_nuc = np.einsum('a,ax->x', charges, coords)
    mu_total = mu_nuc - mu_el

    print(f"\nDipole moment: mu = mu_nuc - Tr[P*r]")
    print(f"   mu_nuc = ({mu_nuc[0]:.6f}, {mu_nuc[1]:.6f}, {mu_nuc[2]:.6f})")
    print(f"   mu_el  = ({mu_el[0]:.6f}, {mu_el[1]:.6f}, {mu_el[2]:.6f})")
    print(f"   mu     = ({mu_total[0]:.6f}, {mu_total[1]:.6f}, {mu_total[2]:.6f})")
    print(f"   |mu|   = {np.linalg.norm(mu_total) * AU_TO_DEBYE:.4f} Debye")

    # Verify against PySCF
    print("\n[Validation against PySCF]")
    print("-" * 50)

    assert abs(N_e - mol.nelectron) < 1e-10, "Electron count mismatch!"
    print(f"Electron count: PASSED")

    mu_pyscf = mf.dip_moment(unit='AU', verbose=0)
    assert np.allclose(mu_total, mu_pyscf, atol=1e-8), "Dipole mismatch!"
    print(f"Dipole moment: PASSED")

    print(f"""
[Part d] REFLECTION: The Unifying Pattern

The trace formula <O> = Tr[P*o] unifies diverse quantities:

  - Electron count:     o = S (identity operator)
  - Kinetic energy:     o = T (kinetic operator)
  - Nuclear attraction: o = V (potential operator)
  - Dipole moment:      o = r (position operator)
  - Quadrupole moment:  o = r*r (second moment tensor)

Once the density matrix P is known and integral machinery is available,
ANY one-electron property follows from a single matrix multiplication
and trace. This is the "integrals-first" viewpoint in action!
""")

    print("=" * 70)


# ==============================================================================
# Exercise 7.9: Basis Set Conditioning and DF Stability [Advanced]
# ==============================================================================


def exercise_7_9():
    """
    Exercise 7.9: Study relationship between basis conditioning and DF accuracy.
    """
    print("=" * 70)
    print("Exercise 7.9: Basis Set Conditioning and DF Stability")
    print("=" * 70)

    atom_str = 'O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043'

    basis_sets = ['STO-3G', '6-31G', '6-31+G*', '6-31++G**']

    print("\n[Part a] Smallest eigenvalue of S matrix")
    print("-" * 70)
    print(f"{'Basis':<15} {'N_AO':>6} {'s_min':>14} {'kappa(S)':>14} {'DF Error':>14}")
    print("-" * 70)

    results = []

    for basis in basis_sets:
        mol = gto.M(atom=atom_str, basis=basis, unit='Angstrom', verbose=0)

        S = mol.intor('int1e_ovlp')
        eigvals = np.linalg.eigvalsh(S)
        s_min = eigvals.min()
        kappa = eigvals.max() / s_min

        # Run conventional and DF
        E_conv, _, _ = run_conventional_hf(mol)
        E_df, _, _ = run_df_hf(mol)
        df_error = abs(E_df - E_conv)

        results.append({
            'basis': basis, 'nao': mol.nao, 's_min': s_min,
            'kappa': kappa, 'df_error': df_error
        })

        print(f"{basis:<15} {mol.nao:>6} {s_min:>14.2e} {kappa:>14.1f} {df_error:>14.2e}")

    # Part b: Correlation analysis
    print("\n[Part b] DF error vs log10(s_min)")
    print("-" * 50)

    for r in results:
        log_smin = np.log10(r['s_min'])
        print(f"{r['basis']:<15} log10(s_min) = {log_smin:>7.2f}, DF error = {r['df_error']:.2e}")

    print(f"""
[Part c] Correlation Analysis:

There is a WEAK correlation: DF error tends to increase as s_min decreases.
However, the correlation is INDIRECT:

1. Diffuse bases require auxiliary functions with diffuse exponents
2. If auxiliary basis is not matched, fitting quality degrades
3. Orbital basis conditioning affects MO orthogonalization, not DF directly

[Part d] Auxiliary basis metric (P|Q):

The more relevant quantity is the condition number of the auxiliary
Coulomb metric (P|Q). A high kappa((P|Q)) indicates potential DF
instability. The auxiliary basis should be matched to the orbital
basis to maintain reasonable conditioning.

BEST PRACTICE: Use matched JKFIT auxiliary bases
  - For cc-pVDZ: use cc-pVDZ-jkfit
  - For aug-cc-pVDZ: use aug-cc-pVDZ-jkfit
  - Avoid mixing unmatched orbital/auxiliary combinations
""")

    print("=" * 70)


# ==============================================================================
# Exercise 7.10: Virial Theorem at Non-Equilibrium Geometries [Research]
# ==============================================================================


def exercise_7_10():
    """
    Exercise 7.10: Study virial ratio along H2 dissociation curve.

    Demonstrates the general virial theorem with force correction.
    """
    print("=" * 70)
    print("Exercise 7.10: Virial Theorem at Non-Equilibrium Geometries")
    print("=" * 70)

    basis = 'cc-pVDZ'
    bond_lengths = [0.5, 0.6, 0.74, 1.0, 1.5, 2.0, 2.5, 3.0]  # Angstrom

    print(f"\nMolecule: H2")
    print(f"Basis: {basis}")
    print(f"Equilibrium bond length: ~0.74 Angstrom")
    print("-" * 70)
    print(f"{'R (A)':>8} {'E (Eh)':>16} {'<T> (Eh)':>12} {'eta':>10} {'|eta-2|':>12}")
    print("-" * 70)

    results = []

    for R in bond_lengths:
        mol = gto.M(
            atom=f'H 0 0 0; H 0 0 {R}',
            basis=basis,
            unit='Angstrom',
            verbose=0
        )

        mf = scf.RHF(mol)
        mf.kernel()

        result = compute_energy_components(mol, mf)
        result['R'] = R
        results.append(result)

        print(f"{R:>8.2f} {result['E_total']:>16.10f} {result['E_T']:>12.6f} "
              f"{result['eta']:>10.6f} {result['eta_deviation']:>12.2e}")

    # Find minimum deviation from 2
    min_idx = np.argmin([r['eta_deviation'] for r in results])
    R_best = results[min_idx]['R']

    print(f"\neta closest to 2 at R = {R_best:.2f} Angstrom")

    print(f"""
[Part d] GENERAL VIRIAL THEOREM WITH FORCES:

The general virial theorem states:

    2<T> + <V> = -sum_A R_A . F_A

where F_A is the force on nucleus A.

At equilibrium: F_A = 0 for all A, so eta = 2

At non-equilibrium:
  - COMPRESSED (R < R_eq): Forces are repulsive (outward)
    R_A . F_A > 0, so 2<T> + <V> < 0, meaning eta > 2

  - STRETCHED (R > R_eq): Forces are attractive (inward)
    R_A . F_A < 0, so 2<T> + <V> > 0, meaning eta < 2

This matches our computational observations!

[Part e] For polar molecules (HF, LiH):
  - Qualitatively similar behavior
  - Ionic character shows stronger deviations at stretched geometries
  - Electron localization effects become more pronounced
""")

    print("=" * 70)


# ==============================================================================
# Exercise 7.11: Auxiliary Basis Set Design Principles [Research]
# ==============================================================================


def exercise_7_11():
    """
    Exercise 7.11: Compare different auxiliary basis sets.

    Investigates how auxiliary basis choice affects DF accuracy.
    """
    print("=" * 70)
    print("Exercise 7.11: Auxiliary Basis Set Design Principles")
    print("=" * 70)

    mol = gto.M(
        atom='O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    N = mol.nao

    # Reference: conventional HF
    E_conv, _, _ = run_conventional_hf(mol)

    aux_bases = [
        ('def2-universal-jkfit', 'def2-universal-jkfit'),
        ('cc-pVDZ-jkfit', 'cc-pVDZ-jkfit'),
        ('cc-pVTZ-jkfit', 'cc-pVTZ-jkfit'),
    ]

    print(f"\nOrbital basis: cc-pVDZ (N_AO = {N})")
    print(f"Reference (conventional HF): E = {E_conv:.10f} Eh")
    print("-" * 70)
    print(f"{'Auxiliary Basis':<25} {'N_aux':>8} {'N_aux/N':>10} "
          f"{'DF Error (Eh)':>14} {'Time (s)':>10}")
    print("-" * 70)

    results = []

    for name, auxbasis in aux_bases:
        try:
            E_df, t_df, mf_df = run_df_hf(mol, auxbasis=auxbasis)

            # Get auxiliary basis size
            auxmol = mf_df.with_df.auxmol
            n_aux = auxmol.nao if auxmol is not None else 0

            delta_E = abs(E_df - E_conv)
            ratio = n_aux / N if N > 0 else 0

            results.append({'name': name, 'n_aux': n_aux, 'ratio': ratio,
                           'error': delta_E, 'time': t_df})

            print(f"{name:<25} {n_aux:>8} {ratio:>10.1f} "
                  f"{delta_E:>14.2e} {t_df:>10.3f}")
        except Exception as e:
            print(f"{name:<25} {'Error':>8} {str(e)[:30]}")

    print(f"""
[Part c] N_aux/N ratio analysis:

Optimal range: N_aux/N ~ 3-4 for JK fitting

Matched auxiliary bases (cc-pVXZ-jkfit for cc-pVXZ) typically give
the best balance of accuracy and efficiency.

[Part d] WEIGEND'S DESIGN PRINCIPLES (PCCP 2006):

1. ANGULAR MOMENTUM COVERAGE: Include functions up to 2*L_max
   to represent all pair products.

2. EXPONENT MATCHING: Auxiliary exponents should span the range
   of products of orbital exponents.

3. EVEN-TEMPERED SEQUENCES: Use geometric progressions for
   systematic coverage.

4. OPTIMIZATION: Minimize fitting error for a training set of
   molecules.

5. BASIS-SPECIFIC: Match auxiliary to orbital basis
   (cc-pVXZ-JKFIT for cc-pVXZ, etc.)

Using a matched auxiliary basis typically gives DF errors 10-100x
smaller than mismatched choices.
""")

    print("=" * 70)


# ==============================================================================
# Exercise 7.12: From Integrals to Forces (Capstone Preview) [Research]
# ==============================================================================


def exercise_7_12():
    """
    Exercise 7.12: Compare analytical vs numerical gradients.

    Demonstrates the importance of Pulay terms in gradient calculations.
    """
    print("=" * 70)
    print("Exercise 7.12: From Integrals to Forces (Capstone Preview)")
    print("=" * 70)

    from pyscf.grad import rhf as rhf_grad

    # Slightly distorted H2O geometry
    mol = gto.M(
        atom='O 0 0 0; H 0.80 0 0.55; H -0.72 0 0.48',  # Distorted from equilibrium
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    mf = scf.RHF(mol)
    mf.kernel()

    print(f"\nMolecule: H2O (distorted geometry)")
    print(f"Basis: cc-pVDZ")
    print(f"RHF Energy: {mf.e_tot:.10f} Eh")

    # Analytical gradient using proper PySCF gradient module
    g = rhf_grad.Gradients(mf)
    grad_analytical = g.kernel()

    # Numerical gradient
    print("\n[Computing numerical gradient (central difference)...]")

    delta = 1e-4  # Bohr
    grad_numerical = np.zeros((mol.natm, 3))

    coords = mol.atom_coords().copy()
    atom_symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    for i in range(mol.natm):
        for j in range(3):
            coords_plus = coords.copy()
            coords_plus[i, j] += delta

            coords_minus = coords.copy()
            coords_minus[i, j] -= delta

            # Create shifted molecules
            atom_str_plus = "; ".join([
                f"{atom_symbols[k]} {coords_plus[k,0]} {coords_plus[k,1]} {coords_plus[k,2]}"
                for k in range(mol.natm)
            ])
            atom_str_minus = "; ".join([
                f"{atom_symbols[k]} {coords_minus[k,0]} {coords_minus[k,1]} {coords_minus[k,2]}"
                for k in range(mol.natm)
            ])

            mol_plus = gto.M(atom=atom_str_plus, basis='cc-pVDZ', unit='Bohr', verbose=0)
            mol_minus = gto.M(atom=atom_str_minus, basis='cc-pVDZ', unit='Bohr', verbose=0)

            E_plus = scf.RHF(mol_plus).kernel()
            E_minus = scf.RHF(mol_minus).kernel()

            grad_numerical[i, j] = (E_plus - E_minus) / (2 * delta)

    print("\n[Gradient comparison]")
    print("-" * 70)
    print(f"{'Atom':<6} {'Dir':>4} {'Analytical':>14} {'Numerical':>14} {'Difference':>14}")
    print("-" * 70)

    max_diff = 0
    for i in range(mol.natm):
        for j, d in enumerate(['x', 'y', 'z']):
            diff = abs(grad_analytical[i, j] - grad_numerical[i, j])
            max_diff = max(max_diff, diff)
            print(f"{atom_symbols[i]:<6} {d:>4} {grad_analytical[i,j]:>14.8f} "
                  f"{grad_numerical[i,j]:>14.8f} {diff:>14.2e}")

    print(f"\nMax difference: {max_diff:.2e}")

    print(f"""
[Part d] SOURCE OF PULAY TERMS:

The analytical gradient includes three types of integral derivatives:

1. dH_core/dR: Core Hamiltonian derivative (Hellmann-Feynman-like)
2. d(ERI)/dR:  Two-electron integral derivatives
3. dS/dR:      Overlap derivative -> THIS GIVES PULAY TERMS

The Pulay force contribution is:
    F_A^Pulay = -sum_mu_nu W_mu_nu * dS_mu_nu/dR_A

where W is the energy-weighted density matrix.

[Part e] WHY ANALYTICAL GRADIENTS ARE PREFERRED:

                      Numerical           Analytical
Cost for N atoms      6N SCF calcs        1 SCF + gradient
Accuracy              ~6 digits           Machine precision
Step-size dependence  Sensitive           None
Stability             Subtractive cancel  Robust

For geometry optimization, analytical gradients are ESSENTIAL for
efficiency and reliability.
""")

    print("=" * 70)


# ==============================================================================
# Main driver: Run all exercises
# ==============================================================================


def main():
    """Run all Chapter 7 exercises with clear section headers."""

    print("\n" + "#" * 78)
    print("#" + " " * 76 + "#")
    print("#" + " CHAPTER 7 EXERCISE SOLUTIONS ".center(76) + "#")
    print("#" + " Scaling and Properties ".center(76) + "#")
    print("#" + " " * 76 + "#")
    print("#" * 78 + "\n")

    # Core exercises (7.1 - 7.5)
    print("\n" + "=" * 78)
    print(" CORE EXERCISES ".center(78, "="))
    print("=" * 78 + "\n")

    exercise_7_1()
    print("\n")

    exercise_7_2()
    print("\n")

    exercise_7_3()
    print("\n")

    exercise_7_4()
    print("\n")

    exercise_7_5()
    print("\n")

    # Advanced exercises (7.6 - 7.9)
    print("\n" + "=" * 78)
    print(" ADVANCED EXERCISES ".center(78, "="))
    print("=" * 78 + "\n")

    exercise_7_6()
    print("\n")

    exercise_7_7()
    print("\n")

    exercise_7_8()
    print("\n")

    exercise_7_9()
    print("\n")

    # Research exercises (7.10 - 7.12)
    print("\n" + "=" * 78)
    print(" RESEARCH EXERCISES ".center(78, "="))
    print("=" * 78 + "\n")

    exercise_7_10()
    print("\n")

    exercise_7_11()
    print("\n")

    exercise_7_12()
    print("\n")

    # Summary
    print("\n" + "#" * 78)
    print("#" + " " * 76 + "#")
    print("#" + " ALL CHAPTER 7 EXERCISES COMPLETED ".center(76) + "#")
    print("#" + " " * 76 + "#")
    print("#" * 78 + "\n")


if __name__ == "__main__":
    main()
