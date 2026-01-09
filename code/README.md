# Companion Code for Advanced Quantum Chemistry

Standalone Python implementations accompanying the lecture notes for **2302638 Advanced Quantum Chemistry**, Chulalongkorn University.

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py
```

## Directory Structure

```
code/
├── ch01/              # Chapter 1: Electron-Integral View
├── ch02/              # Chapter 2: Basis Sets and Orthonormalization
├── ch03/              # Chapter 3: One-Electron Integrals and GPT
├── ch04/              # Chapter 4: Two-Electron Integrals and Boys Function
├── ch05/              # Chapter 5: Rys Quadrature in Practice
├── ch06/              # Chapter 6: Hartree-Fock SCF
├── ch07/              # Chapter 7: Scaling, DF, and Properties
├── appendix_b/        # Atomic Units and Conversions
├── appendix_c/        # Gaussian Integral Formulas
├── appendix_d/        # Boys Function Reference
├── appendix_e/        # Rys Quadrature Reference
├── appendix_f/        # PySCF/libcint Interface
├── requirements.txt
└── verify_setup.py
```

---

## Chapter 1: Electron-Integral View of Quantum Chemistry

| File | Description |
|------|-------------|
| `lab1a_integrals.py` | AO integral extraction, symmetry verification, RHF energy validation |

**Topics covered:**
- Building molecules in PySCF
- Extracting one-electron integrals (S, T, V)
- Extracting two-electron integrals (ERIs)
- Verifying 8-fold ERI symmetry
- Reconstructing HF energy from integrals

---

## Chapter 2: Gaussian Basis Sets and Orthonormalization

| File | Description |
|------|-------------|
| `lab2a_overlap_conditioning.py` | Eigenvalues of S and conditioning vs basis set |
| `lab2b_orthogonalizers.py` | Build X such that X^T S X = I (canonical, symmetric, Cholesky, Gram-Schmidt) |
| `lab2c_gen_eigenproblem.py` | Solve FC = SCε via orthogonalization |

**Topics covered:**
- Overlap matrix eigenvalue spectrum
- Near-linear dependence detection
- Symmetric (Löwdin) orthogonalization
- Canonical orthogonalization with thresholding
- Cholesky orthogonalization (Section 2.8)
- Gram-Schmidt under S-metric (pedagogical)
- Generalized eigenvalue problem transformation

---

## Chapter 3: One-Electron Integrals and Gaussian Product Theorem

| File | Description |
|------|-------------|
| `lab3a_integral_sanity.py` | One-electron integral sanity checks |
| `lab3b_analytic_vs_pyscf.py` | Compare analytic formulas vs PySCF |
| `lab3c_dipole_moment.py` | Dipole integrals and dipole moment from density |
| `gaussian_product_theorem.py` | GPT exploration and visualization |
| `boys_function_exploration.py` | Boys function introduction |
| `overlap_screening.py` | Overlap decay and screening |

**Topics covered:**
- Gaussian Product Theorem (GPT)
- Analytic overlap and kinetic integrals
- Nuclear attraction integrals preview
- Dipole moment calculation

---

## Chapter 4: Two-Electron Integrals and Rys Quadrature Foundations

| File | Description |
|------|-------------|
| `boys_function.py` | Boys function evaluator with multiple strategies |
| `boys_quadrature_comparison.py` | Numerical quadrature approaches for Boys function |
| `ssss_eri.py` | Primitive (ss\|ss) ERI implementation |
| `eri_symmetry.py` | ERI 8-fold symmetry verification |
| `eri_scaling.py` | ERI storage and computation scaling |
| `schwarz_screening.py` | Schwarz inequality for ERI screening |

**Topics covered:**
- Boys function F_n(T) evaluation
- Series expansion and recurrence relations
- (ss|ss) closed-form ERI
- 8-fold permutation symmetry
- Schwarz screening bounds

---

## Chapter 5: Rys Quadrature in Practice

| File | Description |
|------|-------------|
| `boys_moments.py` | Boys function moments m_n(T) = 2F_n(T) |
| `rys_quadrature.py` | Rys quadrature nodes and weights (Algorithm 5.1) |
| `moment_matching.py` | Verification of Rys quadrature moment matching |
| `psss_eri.py` | (p_ξ s\|ss) ERI via derivative identity |
| `jk_build.py` | Coulomb (J) and Exchange (K) matrix construction |
| `hf_energy_validation.py` | Complete HF energy from integrals |

**Topics covered:**
- Hankel matrix construction from moments
- Golub-Welsch algorithm
- Higher angular momentum ERIs
- J/K matrix building from ERIs

---

## Chapter 6: Hartree-Fock SCF from Integrals

| File | Description |
|------|-------------|
| `lab6a_rhf_scf.py` | Minimal RHF SCF from AO integrals |
| `lab6b_diis.py` | Pulay DIIS implementation |
| `lab6c_jk_comparison.py` | In-core vs direct J/K building |

**Topics covered:**
- Complete RHF SCF algorithm
- Convergence criteria (energy, density, DIIS error)
- DIIS acceleration
- In-core vs direct SCF strategies

---

## Chapter 7: Scaling, Density Fitting, and Properties

| File | Description |
|------|-------------|
| `lab7a_df_hf_comparison.py` | Conventional HF vs Density-Fitted HF |
| `lab7b_dipole_calculation.py` | Dipole moment from density matrix |
| `lab7c_virial_diagnostic.py` | Virial theorem diagnostic |
| `scaling_analysis.py` | Scaling analysis for HF calculations |
| `basis_conditioning.py` | Basis set conditioning and numerical stability |

**Topics covered:**
- O(N⁴) ERI scaling vs O(N³) DF scaling
- Three-index tensor factorization
- One-electron property calculation
- Virial ratio diagnostics

---

## Appendix B: Atomic Units and Conversions

| File | Description |
|------|-------------|
| `geometry_units.py` | Unit specification in PySCF |
| `unit_conversions.py` | CODATA conversion factors |
| `unit_error_detection.py` | Common unit mistakes and sanity checks |
| `dipole_calculation.py` | Dipole moment with unit conversion |

---

## Appendix C: Gaussian Integral Formula Sheet

| File | Description |
|------|-------------|
| `gaussian_product_theorem.py` | GPT implementation and verification |
| `overlap_integral.py` | s-s overlap integral |
| `kinetic_integral.py` | s-s kinetic energy integral |
| `boys_function.py` | Boys function with series/recurrence |
| `nuclear_attraction.py` | Nuclear attraction integral |
| `two_electron_integral.py` | (ss\|ss) ERI implementation |
| `hf_energy_validation.py` | Complete HF energy from integrals |

---

## Appendix D: Boys Function Reference

| File | Description |
|------|-------------|
| `boys_function.py` | Core implementation with hybrid evaluation |
| `boys_validation.py` | Validation against scipy.special.gammainc |
| `boys_stability.py` | Stability issues and solutions |
| `boys_vs_pyscf.py` | Comparison with PySCF's implementation |

---

## Appendix E: Rys Quadrature Reference

| File | Description |
|------|-------------|
| `rys_quadrature.py` | Core Rys quadrature implementation |
| `rys_eri_example.py` | Complete (ss\|ss) integral via Rys |
| `rys_validation.py` | Validation against reference values |

---

## Appendix F: PySCF/libcint Interface

| File | Description |
|------|-------------|
| `molecule_construction.py` | Molecule construction methods in PySCF |
| `integral_inventory.py` | Comprehensive integral extraction |
| `eri_symmetry_demo.py` | ERI symmetry demonstration |
| `shell_access_demo.py` | Shell-by-shell integral access |
| `jk_build_comparison.py` | Manual vs PySCF J/K construction |
| `scf_from_scratch.py` | Minimal SCF using only PySCF integrals |
| `diis_demo.py` | DIIS acceleration demonstration |
| `dipole_calculation.py` | Dipole from density matrix |
| `density_fitting_timing.py` | DF vs conventional HF timing |
| `debug_utilities.py` | Validation functions for HF implementations |

---

## Key Formulas

### Gaussian Product Theorem
```
exp(-α|r-A|²) × exp(-β|r-B|²) = exp(-p|r-P|²) × exp(-μR²_AB)

where: p = α+β, μ = αβ/(α+β), P = (αA+βB)/(α+β)
```

### Boys Function
```
F_n(T) = ∫₀¹ t^(2n) exp(-Tt²) dt

Special cases:
  F_n(0) = 1/(2n+1)
  F_0(T) = (1/2)√(π/T) erf(√T)
```

### Hartree-Fock Energy
```
E_HF = ½Tr[P(h+F)] + E_nuc

where: P = density matrix, h = T + V, F = h + J - ½K
```

---

## Running the Code

```bash
# Run individual scripts
python ch01/lab1a_integrals.py
python ch06/lab6a_rhf_scf.py

# Run all scripts in a chapter
for f in ch05/*.py; do python "$f"; done

# Run all appendix scripts
for f in appendix_c/*.py; do python "$f"; done
```

All implementations are validated against PySCF reference calculations.

---

## References

### Primary Textbooks

1. **Szabo, A.; Ostlund, N. S.** *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory*. Dover Publications: Mineola, NY, 1996. ISBN: 978-0-486-69186-2.
   - Chapters 3-4: Hartree-Fock theory and basis sets

2. **Helgaker, T.; Jorgensen, P.; Olsen, J.** *Molecular Electronic-Structure Theory*. Wiley: Chichester, 2000.
   [DOI: 10.1002/9781119019572](https://doi.org/10.1002/9781119019572)
   - Comprehensive reference for integrals, SCF, and post-HF methods

### Key Papers

3. **Dupuis, M.; Rys, J.; King, H. F.** Evaluation of molecular integrals over Gaussian basis functions. *J. Chem. Phys.* **1976**, *65*, 111-116.
   [DOI: 10.1063/1.432807](https://doi.org/10.1063/1.432807)
   - Original Rys quadrature paper for ERI evaluation

4. **Rys, J.; Dupuis, M.; King, H. F.** Computation of electron repulsion integrals using the Rys quadrature method. *J. Comput. Chem.* **1983**, *4*, 154-157.
   [DOI: 10.1002/jcc.540040206](https://doi.org/10.1002/jcc.540040206)
   - Follow-up computational procedure for Rys quadrature

5. **Pulay, P.** Convergence acceleration of iterative sequences. The case of SCF iteration. *Chem. Phys. Lett.* **1980**, *73*, 393-398.
   [DOI: 10.1016/0009-2614(80)80396-4](https://doi.org/10.1016/0009-2614(80)80396-4)
   - Original DIIS (Direct Inversion in the Iterative Subspace) paper

6. **Pulay, P.** Improved SCF convergence acceleration. *J. Comput. Chem.* **1982**, *3*, 556-560.
   [DOI: 10.1002/jcc.540030413](https://doi.org/10.1002/jcc.540030413)
   - DIIS improvements for SCF

### Software Citations

7. **Sun, Q.** Libcint: An efficient general integral library for Gaussian basis functions. *J. Comput. Chem.* **2015**, *36*, 1664-1671.
   [DOI: 10.1002/jcc.23981](https://doi.org/10.1002/jcc.23981)
   - Integral library underlying PySCF

8. **Sun, Q.; Berkelbach, T. C.; Blunt, N. S.; Booth, G. H.; Guo, S.; Li, Z.; Liu, J.; McClain, J.; Sayfutyarova, E. R.; Sharma, S.; Wouters, S.; Chan, G. K.-L.** PySCF: the Python-based simulations of chemistry framework. *WIREs Comput. Mol. Sci.* **2018**, *8*, e1340.
   [DOI: 10.1002/wcms.1340](https://doi.org/10.1002/wcms.1340)
   - Primary PySCF citation

9. **Sun, Q.; Zhang, X.; Banerjee, S.; Bao, P.; Barbry, M.; Blunt, N. S.; Bogdanov, N. A.; Booth, G. H.; Chen, J.; Cui, Z.-H.; et al.** Recent developments in the PySCF program package. *J. Chem. Phys.* **2020**, *153*, 024109.
   [DOI: 10.1063/5.0006074](https://doi.org/10.1063/5.0006074)
   - Recent PySCF developments

### Constants and Units

10. **Tiesinga, E.; Mohr, P. J.; Newell, D. B.; Taylor, B. N.** CODATA recommended values of the fundamental physical constants: 2018. *Rev. Mod. Phys.* **2021**, *93*, 025010.
    [DOI: 10.1103/RevModPhys.93.025010](https://doi.org/10.1103/RevModPhys.93.025010)
    - Fundamental constants used for unit conversions

---

## License

MIT License - See [LICENSE](../LICENSE) for details.
