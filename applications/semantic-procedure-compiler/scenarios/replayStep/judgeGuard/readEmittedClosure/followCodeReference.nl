---
description: Judge one generated body's real dependencies and advance the original-file closure frontier.
args:
  state: CodeClosureState
returns: CodeClosureState
---
1. Select the first pending path and locate its exact original ExecutionFile. Require it to be in
   unvisitedPaths. Copy the state, remove this path from pending and unvisited, and append this actual
   file once to visitedFiles. Earlier files, reference judgments and runtime reads remain unchanged.
2. Prepare a scoped judgment from this file's actual body, original available file paths/kinds and
   selected runtime clause IDs plus the actual defining type-file declaration texts. Type syntax is
   exact available code, not ideal behavior guidance. Exclude ideal plans, source policy meanings, scenario expectations,
   audited intended responsibilities and the source-required permission/effect channel.
3. In eval declare CodeDependencyJudgment matching the supplied type. Create a scoped NL literal
   whose varying input is this original file. Ask which named calls and type dependencies the actual
   body uses, and which source-definition data it explicitly reads. Directly await that literal.
4. Its semantic task is to distinguish an invoked callee from a quoted example or explanatory mention;
   interpret callable-folder relative references using the actual caller path. Resolve a type dependency
   only against an actual defining type file, returning targetEvidence with its exact declaration span.
   Preserve conditional calls even when a particular event
   would not take that branch: closure records possible actual code, not a source-required execution plan.
5. For runtime definitions, distinguish explicit clause selection, a read of the whole selected-clause
   collection, and no runtime source read. Return exact original IDs only when the actual body reads
   them. A name mentioned solely to explain an intended rule does not authorize loading that policy.
6. Consume the literal's actual references and runtimeReads. Bind callerPath from the original file;
   require every decided reference/read to have a nonempty exact quote within that file's text.
   For each resolved type reference also require nonempty targetEvidence at its actual target type file
   and validate every defining quote there; caller-only mention cannot establish a type definition.
   Verify target paths and explicit runtime clause IDs against the original inventories. A missing or
   ambiguous actual target remains unresolved; do not replace it with an ideal planned child.
7. Append checked references and reads once. For each resolved call/type target that is still unvisited,
   add its original path to pending only if absent. A target already visited retains its edge but creates
   no new work. Carry returned unresolved dependencies and invalid-span findings into copied state.
8. Return the copied CodeClosureState with strictly fewer unvisitedPaths. Parent closure extraction
   consumes these actual edges, visited files and read decisions; a correct-looking permission cannot
   substitute for missing transitive child text or a failed code dependency judgment.
