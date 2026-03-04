import Mathlib

-- Line 3: Valid definition
def validDef : Nat := 42

-- Line 6: Error on this line
def errorDef : Nat := "string"

-- Line 9: Another valid definition
def anotherValidDef : Nat := 100

-- Line 12: Another error
def anotherError : String := 123

-- Line 15: Valid theorem
theorem validTheorem : True := by
  trivial
