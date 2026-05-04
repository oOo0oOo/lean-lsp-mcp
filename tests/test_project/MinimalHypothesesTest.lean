import Mathlib

-- Both hypotheses are referenced in the body → both load-bearing.
theorem minhyp_both_used (h1 : 1 + 1 = 2) (h2 : 2 + 2 = 4) : 1 + 1 = 2 ∧ 2 + 2 = 4 :=
  ⟨h1, h2⟩

-- The body never names h2 → h2 is removable.
theorem minhyp_one_unused (h1 : 1 + 1 = 2) (h2 : 2 + 2 = 4) : 1 + 1 = 2 :=
  h1

-- No explicit (h : T) hypotheses.
theorem minhyp_no_hypotheses : True := trivial

-- Mix of explicit + implicit + instance binders. Body uses h1 only.
theorem minhyp_mixed_binders {α : Type} [DecidableEq α] (h1 : 1 + 1 = 2) (h2 : 2 + 2 = 4) : 1 + 1 = 2 :=
  h1
