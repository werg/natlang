---
kind: directory-reducer
args:
  state: ComponentLoopState
returns: ComponentLoopState
---
Perform one component-search experiment and return the next joint source/state checkpoint. You own experiment sequencing. The search service supplies exact GEPA selection math, declared dependencies, independent evaluation and checked state assembly; it never runs a search loop.

Call search.plan(state.iteration). This draws the seeded per-case frontier parent and dependency update group, charges one experiment, and supplies the parent's typed component values plus complete training feedback. If stopReason is coverage-gap, finish with that stop reason and no acceptance. Otherwise install plan.parent in folder's components.json before proposing edits.

When plan.compose is true, call search.merge() for conservative three-way composition. A conflict gives null; then use a normal proposal. For a normal proposal, call folder.propose(rewriteComponents, {keys:plan.keys,components:plan.components,feedback:plan.feedback}). The returned value must equal the edited components.json. The rewrite child can page feedback rather than receiving a truncated first dozen examples.

Call search.check(candidate). A rejected check finishes this experiment with accepted:false and concrete diagnostics. Duplicates in plan.duplicateIds finish without evaluation or acceptance. For a new valid candidate, evaluate both parent and candidate on the same minibatch using search.evaluate(id,'mini',true) and search.evaluate(id,'mini'). Accept only when candidate gates pass and its quality strictly exceeds the parent's. A rejection still records a completed experiment. For an accepted candidate, explicitly evaluate full training and validation using search.evaluate(id,'train') and search.evaluate(id,'validation'); pass their evidence references to search.finish.

search.finish({candidate:id,accepted,train:training.evidence,validation:validated.evidence}) verifies exact acceptance, applies validation constraints and guidance coverage, protects the baseline/incumbent while pruning the per-case frontier, and returns {state,candidate}. It does not choose semantic edits. For an invalid proposal use {accepted:false,error:reason}. Never catch exhaustion or cancellation to create more capacity. The state iteration advances exactly once for each charged experiment.

Finally write returned candidate to folder.file('components.json') and explicitly `return finished.state;` from eval. The selected source can differ from the proposed child: preserve that distinction. Test evidence is absent and cannot drive this experiment. Do not fabricate metrics or ask for a host optimizer endpoint. Reply done after returning the completed typed state.
