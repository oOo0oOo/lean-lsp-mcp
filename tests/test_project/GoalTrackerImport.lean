/-
  GoalTrackerImport: tests cross-file sorry tracking.
  Imports GoalTrackerBase and builds declarations that depend on its sorry'd decls.
-/
import GoalTrackerBase

-- 1. Cross-file: uses sorry'd def from GoalTrackerBase
theorem gtImport_uses_sorry_val : gtBase_sorry_val = gtBase_sorry_val := by rfl

-- 2. Cross-file: uses clean def from GoalTrackerBase (should be clean)
theorem gtImport_uses_clean : gtBase_clean = 42 := by rfl

-- 3. Cross-file chain: depends on chain that ends in sorry in base file
theorem gtImport_chain : gtBase_chain_top > 0 := by sorry

-- 4. Cross-file namespace: uses namespaced sorry def from base
theorem gtImport_uses_ns_sorry : GtBaseNs.ns_sorry_def = GtBaseNs.ns_sorry_def := by rfl

-- 5. Cross-file private: uses pub_uses_priv which depends on private sorry
theorem gtImport_uses_priv_chain : GtBasePrivate.pub_uses_priv > 0 := by sorry

-- 6. Cross-file instance: uses sorry'd instance from base
theorem gtImport_uses_sorry_inst : @GtBaseClass.baseVal Nat gtBase_sorry_instance = 0 := by sorry

-- 7. Mixed: local sorry + cross-file sorry dep
def gtImport_local_sorry : Bool := sorry
theorem gtImport_mixed : gtBase_sorry_val > 0 ∧ gtImport_local_sorry = true := by
  constructor
  · sorry
  · sorry

-- 8. Clean theorem in import file (no sorry anywhere)
theorem gtImport_fully_clean : 1 + 1 = 2 := by norm_num

-- 9. Cross-file noncomputable (no sorry, just axiom)
theorem gtImport_uses_noncomputable : gtBase_noncomputable = gtBase_noncomputable := by rfl

-- 10. Diamond: two paths to same sorry in base file
def gtImport_left : Nat := gtBase_sorry_val + 1
def gtImport_right : Nat := gtBase_sorry_val + 2
theorem gtImport_diamond : gtImport_left + gtImport_right > 0 := by sorry

-- 11. Deep cross-file chain (import adds one more level)
def gtImport_deeper : Nat := gtBase_chain_top + 1
theorem gtImport_deep_chain : gtImport_deeper > 0 := by sorry

-- 12. Local namespace wrapping cross-file dep
namespace GtImportNs
theorem ns_uses_base_sorry : gtBase_sorry_val = gtBase_sorry_val := by rfl
theorem ns_local_sorry : False := by sorry
end GtImportNs

-- 13. Where clause with cross-file sorry dep
theorem gtImport_where : gtBase_sorry_val + n > 0 := by sorry where
  n : Nat := gtBase_chain_mid

-- 14. Let binding referencing cross-file sorry
theorem gtImport_let_sorry : True := by
  let _x := gtBase_sorry_val
  trivial
