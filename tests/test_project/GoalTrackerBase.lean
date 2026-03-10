/-
  GoalTrackerBase: foundation file for cross-file goal_tracker testing.
  Contains sorry'd and clean declarations that GoalTrackerImport.lean depends on.
-/
import Mathlib

-- Clean foundations
def gtBase_clean : Nat := 42
theorem gtBase_clean_thm : gtBase_clean = 42 := by rfl

-- Sorry foundations (public)
def gtBase_sorry_val : Nat := sorry
theorem gtBase_sorry_thm : 1 + 1 = 3 := by sorry

-- Namespaced sorry
namespace GtBaseNs
def ns_sorry_def : Nat := sorry
theorem ns_clean_thm : 1 + 1 = 2 := by norm_num
end GtBaseNs

-- Private sorry (mangled name, cross-file users can't see it directly)
namespace GtBasePrivate
private def priv_sorry : Nat := sorry
def pub_uses_priv : Nat := priv_sorry + 1
theorem pub_thm_uses_priv : pub_uses_priv > 0 := by sorry
end GtBasePrivate

-- Chain: A → B → sorry (exported for cross-file chain testing)
def gtBase_chain_sorry : Nat := sorry
def gtBase_chain_mid : Nat := gtBase_chain_sorry + 1
def gtBase_chain_top : Nat := gtBase_chain_mid + 1

-- Instance with sorry (cross-file consumers will inherit sorry)
class GtBaseClass (α : Type) where
  baseVal : α
instance gtBase_sorry_instance : GtBaseClass Nat where
  baseVal := sorry

-- Noncomputable (no sorry, just Classical.choice)
noncomputable def gtBase_noncomputable : Nat := Classical.choice ⟨0⟩
