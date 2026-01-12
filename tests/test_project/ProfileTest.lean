import Mathlib

theorem simple_by (a : Bool) (h : a = true) : a = true := by
  rw [h]

theorem simp_test (n : ℕ) : n + 0 = n := by
  simp

theorem omega_test (a b c : ℕ) (h1 : a < b) (h2 : b < c) : a < c := by
  omega

-- Multi-tactic proof for profiling multiple lines
theorem multi_step (x y : ℕ) (hx : x > 5) (hy : y > 3) : x + y > 8 := by
  have h1 : x ≥ 6 := hx
  have h2 : y ≥ 4 := hy
  omega
