import Mathlib

def f (n : Nat) : Nat := n + 1
def g (n : Nat) : Nat := f n + 2

example : True := by
  let y := g 5
  suffices y + 3 = f 5 + 5 by trivial
  dsimp [g]
  omega
