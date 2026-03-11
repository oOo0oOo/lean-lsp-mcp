import Mathlib

-- Clean theorem
theorem verify_clean : 1 + 1 = 2 := by norm_num

-- Sorry theorem
theorem verify_sorry : 1 + 1 = 3 := by sorry

-- Unsafe def
unsafe def verify_unsafe_fn : Nat := 42

-- Suspicious options
set_option debug.skipKernelTC true in
theorem verify_skip_tc : True := trivial

-- Local instance
local instance : Inhabited Nat := ⟨99⟩
