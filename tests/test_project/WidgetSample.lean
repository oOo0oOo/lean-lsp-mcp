import Lean.Widget

open Lean
open Lean.Widget

@[widget_module]
def helloModule : Widget.Module :=
  { javascript := "export default function Hello() { return null; }" }

#widget helloModule
