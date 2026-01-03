#!/usr/bin/env python3
"""
Boys Function Stability Analysis

Demonstrates why different evaluation strategies are needed for different T regimes:

1. CATASTROPHIC CANCELLATION in upward recurrence for small T:
   The upward recurrence F_{n+1} = [(2n+1)*F_n - exp(-T)] / (2T) involves
   subtracting two nearly equal numbers when T is small, since:
   - F_n(0) = 1/(2n+1), so (2n+1)*F_n(0) = 1
   - exp(-0) = 1
   The numerator is nearly 0/0 for small T!

2. STABILITY of downward recurrence:
   The downward recurrence F_n = [2T*F_{n+1} + exp(-T)] / (2n+1) adds
   positive quantities, making it unconditionally stable.

3. SERIES vs RECURRENCE comparison:
   The Taylor series converges quickly for small T and avoids the
   cancellation problem entirely.

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix D: Boys Functions
"""

import math
import numpy as np
from scipy.special import gamma as gamma_func
from scipy.special import gammainc

# Import our implementations
from boys_function import boys, _boys_series, _boys_erf_upward


def boys_scipy_reference(n: int, T: float) -> float:
    """Reference implementation using scipy (high accuracy)."""
    if T < 1e-15:
        return 1.0 / (2 * n + 1)
    a = n + 0.5
    return 0.5 * (T**(-a)) * gamma_func(a) * gammainc(a, T)


def boys_upward_only(n: int, T: float) -> float:
    """
    Compute F_n(T) using ONLY upward recurrence from F_0.

    This is UNSTABLE for small T! Included to demonstrate the problem.
    """
    if T == 0:
        return 1.0 / (2 * n + 1)

    # F_0 from erf
    sqrt_T = math.sqrt(T)
    F = 0.5 * math.sqrt(math.pi / T) * math.erf(sqrt_T)

    if n == 0:
        return F

    # Upward recurrence (unstable for small T)
    exp_mT = math.exp(-T)
    for m in range(n):
        F = ((2 * m + 1) * F - exp_mT) / (2 * T)
    return F


def boys_downward_only(n: int, T: float, n_extra: int = 30) -> float:
    """
    Compute F_n(T) using ONLY downward recurrence from high n.

    Starts from asymptotic approximation at n_start = n + n_extra.
    """
    if T == 0:
        return 1.0 / (2 * n + 1)

    n_start = n + n_extra

    # For small T, asymptotic formula can overflow. Use series instead.
    # The threshold depends on n_start since we compute T**(n_start + 0.5)
    # T^40.5 = 0 for T < ~1e-8 in float64
    T_threshold = max(1e-6, 10**(-300.0 / (n_start + 0.5)))
    if T < T_threshold:
        val = 0.0
        term = 1.0
        for k in range(50):
            val += term / (2 * n + 2 * k + 1)
            term *= -T / (k + 1)
            if abs(term) < 1e-16:
                break
        return val

    # Asymptotic starting value for large n
    double_fact = 1.0
    for k in range(1, 2 * n_start, 2):
        double_fact *= k
    F = double_fact / (2**(n_start + 1) * T**(n_start + 0.5)) * math.sqrt(math.pi)

    # Downward recurrence
    exp_mT = math.exp(-T)
    for m in range(n_start - 1, n - 1, -1):
        F = (2 * T * F + exp_mT) / (2 * m + 1)
    return F


def demonstrate_cancellation():
    """Show catastrophic cancellation in upward recurrence for small T."""
    print("\n" + "=" * 75)
    print("Demonstration 1: Catastrophic Cancellation in Upward Recurrence")
    print("=" * 75)

    print("\nFor the upward recurrence: F_{n+1} = [(2n+1)*F_n - exp(-T)] / (2T)")
    print("When T -> 0:")
    print("  - F_n(0) = 1/(2n+1), so (2n+1)*F_n(0) = 1")
    print("  - exp(-0) = 1")
    print("  - Numerator = 1 - 1 = 0 (subtraction of equal numbers!)")
    print("  - Denominator = 2*0 = 0")
    print("  - Result: 0/0 catastrophe!")

    print("\n" + "-" * 75)
    print("Showing the cancellation magnitude:")
    print("-" * 75)

    n = 2
    print(f"\nFor n = {n}, computing F_{n+1}(T) via upward from F_{n}(T):")
    print(f"{'T':>12}  {'(2n+1)*F_n':>18}  {'exp(-T)':>18}  {'Numerator':>18}")
    print("-" * 75)

    for T in [1.0, 0.1, 0.01, 1e-3, 1e-5, 1e-8, 1e-10, 1e-12]:
        F_n = boys_scipy_reference(n, T)
        term1 = (2 * n + 1) * F_n
        term2 = math.exp(-T)
        numerator = term1 - term2
        print(f"{T:12.2e}  {term1:18.14f}  {term2:18.14f}  {numerator:18.14e}")

    print("\nNote: As T decreases, the numerator becomes the difference of")
    print("two nearly identical numbers, losing significant digits!")


def compare_upward_downward():
    """Compare accuracy of upward vs downward recurrence."""
    print("\n" + "=" * 75)
    print("Demonstration 2: Upward vs Downward Recurrence Accuracy")
    print("=" * 75)

    print("\nComparing errors for F_n(T) computed via:")
    print("  - Upward recurrence from F_0")
    print("  - Downward recurrence from asymptotic F_{n+30}")
    print("  - Reference: scipy.special.gammainc")

    n_values = [0, 2, 5, 10]
    T_values = [1e-12, 1e-8, 1e-4, 0.01, 0.1, 1.0, 10.0, 100.0]

    print("\n" + "-" * 90)
    print(f"{'n':>4}  {'T':>10}  {'Reference':>16}  {'Upward Err':>14}  "
          f"{'Downward Err':>14}  {'Better':>10}")
    print("-" * 90)

    for n in n_values:
        for T in T_values:
            ref = boys_scipy_reference(n, T)
            upward = boys_upward_only(n, T)
            downward = boys_downward_only(n, T)

            err_up = abs(upward - ref)
            err_down = abs(downward - ref)

            better = "upward" if err_up < err_down else "downward"
            if err_up < 1e-14 and err_down < 1e-14:
                better = "both"

            print(f"{n:4d}  {T:10.2e}  {ref:16.10e}  {err_up:14.2e}  "
                  f"{err_down:14.2e}  {better:>10}")


def compare_series_recurrence():
    """Compare Taylor series vs recurrence for small T."""
    print("\n" + "=" * 75)
    print("Demonstration 3: Series vs Recurrence for Small T")
    print("=" * 75)

    print("\nFor small T, the Taylor series is more accurate than upward recurrence.")
    print("Series: F_n(T) = sum_{k=0}^inf (-T)^k / [k! * (2n+2k+1)]")

    n_values = [0, 2, 5]
    T_values = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4]

    print("\n" + "-" * 85)
    print(f"{'n':>4}  {'T':>10}  {'Reference':>16}  {'Series Err':>14}  "
          f"{'Upward Err':>14}")
    print("-" * 85)

    for n in n_values:
        for T in T_values:
            ref = boys_scipy_reference(n, T)
            series = _boys_series(n, T)
            upward = boys_upward_only(n, T)

            err_series = abs(series - ref)
            err_upward = abs(upward - ref)

            print(f"{n:4d}  {T:10.2e}  {ref:16.10e}  {err_series:14.2e}  "
                  f"{err_upward:14.2e}")

    print("\nNote: Series maintains accuracy even for T = 1e-12, while")
    print("upward recurrence loses many significant digits.")


def generate_stability_data():
    """Generate data for plotting stability comparison."""
    print("\n" + "=" * 75)
    print("Demonstration 4: Error Data for Stability Plots")
    print("=" * 75)

    print("\nGenerating log-log error data (T vs relative error)...")
    print("This data can be used to create stability plots.")

    n = 5  # Fixed n for this demonstration

    # Logarithmically spaced T values from 1e-14 to 1e2
    T_points = np.logspace(-14, 2, 50)

    print(f"\nData for F_{n}(T):")
    print("-" * 80)
    print(f"{'T':>14}  {'log10(T)':>10}  {'Upward RelErr':>16}  "
          f"{'Downward RelErr':>16}  {'Series RelErr':>16}")
    print("-" * 80)

    for T in T_points:
        ref = boys_scipy_reference(n, T)

        if abs(ref) < 1e-300:
            continue

        upward = boys_upward_only(n, T)
        downward = boys_downward_only(n, T)
        series = _boys_series(n, T, max_terms=100)

        rel_err_up = abs(upward - ref) / abs(ref)
        rel_err_down = abs(downward - ref) / abs(ref)
        rel_err_series = abs(series - ref) / abs(ref)

        # Clamp errors to reasonable range for plotting
        rel_err_up = max(rel_err_up, 1e-17)
        rel_err_down = max(rel_err_down, 1e-17)
        rel_err_series = max(rel_err_series, 1e-17)

        print(f"{T:14.4e}  {math.log10(T):10.2f}  {rel_err_up:16.4e}  "
              f"{rel_err_down:16.4e}  {rel_err_series:16.4e}")


def show_optimal_strategy():
    """Demonstrate the hybrid strategy used in our implementation."""
    print("\n" + "=" * 75)
    print("Demonstration 5: Optimal Hybrid Strategy")
    print("=" * 75)

    print("\nOur boys() function uses:")
    print("  1. T = 0: exact value 1/(2n+1)")
    print("  2. T < 1e-10: Taylor series (stable, avoids cancellation)")
    print("  3. T > 35 + 5*n: asymptotic + downward (numerically stable)")
    print("  4. Intermediate: F_0 via erf + upward (fast and accurate)")

    n = 5
    T_test = [0, 1e-12, 1e-6, 0.1, 1.0, 10.0, 50.0, 100.0]

    print(f"\nVerifying accuracy for n = {n}:")
    print("-" * 70)
    print(f"{'T':>12}  {'boys(n,T)':>18}  {'Reference':>18}  {'Abs Error':>14}")
    print("-" * 70)

    for T in T_test:
        ours = boys(n, T)
        ref = boys_scipy_reference(n, T)
        error = abs(ours - ref)
        print(f"{T:12.2e}  {ours:18.14f}  {ref:18.14f}  {error:14.2e}")

    print("\nAll errors should be < 1e-12 (machine precision).")


def main():
    """Run all stability demonstrations."""
    print("=" * 75)
    print("Boys Function: Numerical Stability Analysis")
    print("=" * 75)
    print("\nThis module demonstrates why careful implementation is needed")
    print("for numerically stable Boys function evaluation.")

    demonstrate_cancellation()
    compare_upward_downward()
    compare_series_recurrence()
    generate_stability_data()
    show_optimal_strategy()

    print("\n" + "=" * 75)
    print("KEY TAKEAWAYS:")
    print("=" * 75)
    print("1. Upward recurrence is UNSTABLE for small T due to cancellation.")
    print("2. Downward recurrence is ALWAYS stable (adds positive quantities).")
    print("3. Taylor series is ideal for small T (converges fast, no cancellation).")
    print("4. A hybrid strategy combining all methods ensures accuracy everywhere.")
    print("=" * 75)


if __name__ == "__main__":
    main()
