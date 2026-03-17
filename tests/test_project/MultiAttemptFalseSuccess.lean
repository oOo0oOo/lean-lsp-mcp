import Mathlib

def f (n : Nat) : Nat := n + 1
def g (n : Nat) : Nat := f n + 2

-- Regression fixture for the REPL-based lean_multi_attempt false-success bug:
-- the in-file tactic fails, but multi_attempt can incorrectly report success.
example : True := by
  let y := g 5
  suffices y + 3 = f 5 + 5 by trivial
  dsimp [g]
  omega
