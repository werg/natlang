---
description: One measured improvement step on a natlang program.
args:
  state: RefineState
  program: ProgramBrief
returns: RefineState
---
1. Pick one cause of the remaining support failures that a change to the program could prevent on other invocations too. Use state.notes and state.tried so you do not repeat a change that did not help.
2. Propose one change for it with workspace.propose(change). Choose the kind that fits the cause: guidance for the whole program, one function or a class of inputs; a skill with a procedure, placed with the function that needs it; a patch to a function's instructions or to its argument and result types; or workspace.fit(spec) when the model's own habits, not missing knowledge, cause the failure. Write the fix in general terms; never copy one invocation's answer. Fix any problems propose reports and propose again.
3. Measure it: const result = await workspace.evaluate([...state.kept, id]).
4. Return the next state: add id to kept and set best to result.success when result.success is above state.best; otherwise increase misses. Add { id, success: result.success, kept } to tried, and update notes with what you learned in one or two sentences. Set done when misses reaches 2 or workspace.budget() has no evaluations left.
