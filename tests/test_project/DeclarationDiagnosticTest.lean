import Mathlib

-- First theorem with a clear type error
theorem firstTheorem : 1 + 1 = 2 := "string instead of proof"

-- Valid definition
def validFunction : Nat := 42

-- Second theorem with an error in the statement type mismatch
theorem secondTheorem : Nat := True

-- Another valid definition
def anotherValidFunction : String := "hello"
