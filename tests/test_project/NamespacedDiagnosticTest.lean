import Mathlib

namespace MyNamespace

-- Theorem with error inside namespace
theorem namespacedTheorem : 1 + 1 = 2 := "wrong proof"

-- Valid definition inside namespace
def namespacedDef : Nat := 42

end MyNamespace

-- Top-level theorem with error
theorem topLevelTheorem : String := 123
