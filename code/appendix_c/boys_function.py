#!/usr/bin/env python3
"""
Boys Function Implementation

The Boys function appears in all integrals involving 1/r operators:

F_n(T) = ∫₀¹ t^(2n) exp(-T t²) dt

Key formulas:
- F_n(0) = 1/(2n+1)
- F_0(T) = (1/2)√(π/T) erf(√T)  for T > 0
- Upward recurrence: F_{n+1} = [(2n+1)F_n - exp(-T)] / (2T)

This module re-exports the stable implementation from appendix_d.
See appendix_d/boys_function.py for the full implementation details.

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix C: Gaussian Integral Formula Sheet
"""

import importlib.util
from pathlib import Path

# Load boys_function from appendix_d using importlib to avoid name collision
_appendix_d_boys = Path(__file__).parent.parent / "appendix_d" / "boys_function.py"
_spec = importlib.util.spec_from_file_location("boys_function_impl", _appendix_d_boys)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# Re-export the stable implementation
boys = _module.boys
boys_array = _module.boys_array

__all__ = ['boys', 'boys_array']


def main():
    """Demonstrate Boys function with key test cases."""
    import math

    print("=" * 60)
    print("Boys Function Verification")
    print("=" * 60)
    print("\n(Using stable implementation from appendix_d)")

    # Special case: T = 0
    print("\nSpecial case F_n(0) = 1/(2n+1):")
    print("-" * 40)
    for n in range(5):
        computed = boys(n, 0)
        exact = 1.0 / (2 * n + 1)
        print(f"  F_{n}(0) = {computed:.6f}  (exact: {exact:.6f})")

    # Various T values
    print("\nBoys function for various T:")
    print("-" * 40)
    T_values = [0.1, 1.0, 5.0, 10.0, 20.0]
    for T in T_values:
        vals = boys_array(3, T)
        print(f"  T = {T:5.1f}: F_0 = {vals[0]:.6f}, F_1 = {vals[1]:.6f}, "
              f"F_2 = {vals[2]:.6f}, F_3 = {vals[3]:.6f}")

    # Asymptotic behavior
    print("\nAsymptotic behavior (large T):")
    print("-" * 40)
    print("  F_0(T) → (1/2)√(π/T) as T → ∞")
    for T in [10, 50, 100]:
        computed = boys(0, T)
        asymptotic = 0.5 * math.sqrt(math.pi / T)
        print(f"  T = {T:3d}: F_0 = {computed:.6f}, asymptotic = {asymptotic:.6f}")

    # Test stability for moderate n, small T (was broken in old version)
    print("\nStability test (n=5, small T):")
    print("-" * 40)
    for T in [1e-6, 1e-4, 1e-2]:
        val = boys(5, T)
        exact_at_zero = 1.0 / 11  # F_5(0) = 1/11
        print(f"  T = {T:.0e}: F_5 = {val:.10f}  (F_5(0) = {exact_at_zero:.10f})")


if __name__ == "__main__":
    main()
