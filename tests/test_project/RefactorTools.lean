import Mathlib

def myHelper : Nat := 42

def usesHelper : Nat := myHelper + 1

theorem helperIsFortyTwo : myHelper = 42 := rfl
