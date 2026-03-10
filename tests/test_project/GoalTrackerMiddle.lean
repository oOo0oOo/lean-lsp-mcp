/-
  GoalTrackerMiddle: sorry-FREE intermediate module.
  Imports GoalTrackerBase and re-exports/wraps its declarations.
  Used to test that the tracker correctly follows sorry through
  a clean intermediate module.
-/
import GoalTrackerBase

-- Clean wrapper around a sorry'd value
def gtMiddle_wrap_sorry : Nat := gtBase_sorry_val + 100

-- Clean wrapper around a clean value
def gtMiddle_wrap_clean : Nat := gtBase_clean + 100

-- Chain through sorry: middle adds a layer
def gtMiddle_chain : Nat := gtBase_chain_top + 1

-- Re-export namespaced sorry through a clean def
def gtMiddle_ns_wrap : Nat := GtBaseNs.ns_sorry_def + 1

-- Completely local clean def (no base deps)
def gtMiddle_local_clean : Nat := 999

-- Clean theorem about clean things
theorem gtMiddle_clean_thm : gtMiddle_local_clean = 999 := by rfl
