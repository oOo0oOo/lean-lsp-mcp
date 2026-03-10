/-
  GoalTrackerHeavy: declarations with deep Mathlib dependencies.
  Used to test goal_tracker performance and heartbeat limits.
  The BFS should NOT walk into Mathlib — only project declarations.
-/
import Mathlib

-- Heavy type signature touching many Mathlib concepts
open MeasureTheory in
noncomputable def gtHeavy_sorry
    (Ω : Type*) [MeasurableSpace Ω] (μ : Measure Ω)
    (f g : Ω → ℝ) (hf : Integrable f μ) (hg : Integrable g μ) :
    ℝ := sorry

-- Theorem using heavy sorry
open MeasureTheory in
theorem gtHeavy_thm
    (Ω : Type*) [MeasurableSpace Ω] (μ : Measure Ω)
    (f g : Ω → ℝ) (hf : Integrable f μ) (hg : Integrable g μ) :
    gtHeavy_sorry Ω μ f g hf hg ≥ 0 := by sorry

-- Chain through heavy types
open MeasureTheory in
noncomputable def gtHeavy_chain
    (Ω : Type*) [MeasurableSpace Ω] (μ : Measure Ω)
    (f : Ω → ℝ) (hf : Integrable f μ) :
    ℝ := gtHeavy_sorry Ω μ f f hf hf + 1

-- Matrix-heavy sorry
open Matrix in
noncomputable def gtHeavy_matrix_sorry (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℂ) :
    Matrix (Fin n) (Fin n) ℂ := sorry

-- Uses matrix sorry
open Matrix in
theorem gtHeavy_matrix_thm (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℂ) :
    gtHeavy_matrix_sorry n A B = gtHeavy_matrix_sorry n A B := by rfl
