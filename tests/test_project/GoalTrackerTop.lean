/-
  GoalTrackerTop: imports GoalTrackerMiddle (sorry-free) which imports
  GoalTrackerBase (has sorry). Tests that sorry is correctly tracked
  through a sorry-free intermediate module.
-/
import GoalTrackerMiddle

-- Uses sorry through clean middle layer (Base → Middle → Top)
theorem gtTop_uses_middle_sorry : gtMiddle_wrap_sorry > 0 := by sorry

-- Uses clean value through middle layer (should be clean)
theorem gtTop_uses_middle_clean : gtMiddle_wrap_clean = gtMiddle_wrap_clean := by rfl

-- Chain: Top → Middle → Base chain → Base sorry
theorem gtTop_deep_chain : gtMiddle_chain > 0 := by sorry

-- Uses middle's namespace wrapper (sorry flows through)
theorem gtTop_ns_chain : gtMiddle_ns_wrap > 0 := by sorry

-- Uses middle's local clean def (no sorry anywhere)
theorem gtTop_fully_clean : gtMiddle_local_clean = 999 := by rfl

-- Mixed: local sorry + sorry through middle
def gtTop_local_sorry : Bool := sorry
theorem gtTop_mixed : gtMiddle_wrap_sorry > 0 ∧ gtTop_local_sorry = true := by
  constructor
  · sorry
  · sorry

-- Diamond through middle: two paths to base sorry via middle
def gtTop_path_a : Nat := gtMiddle_wrap_sorry + 1
def gtTop_path_b : Nat := gtMiddle_wrap_sorry + 2
theorem gtTop_diamond : gtTop_path_a + gtTop_path_b > 0 := by sorry
