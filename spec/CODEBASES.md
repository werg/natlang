# Code bases, calls and locals: design rationale

The normative text is in `SPEC.md`: §3 (lambda parts, locals, code bases),
§4 (combinators), §5 (the seven tools, `call`), §7 (how an interpreter works).
A worked example is `examples/triage/`; the tests are `tests/test_codebase.py`.
This note records why the design is what it is.

**Programs are pseudocode, and the structure is the author's.** A natlang
program is an algorithm: functions with typed signatures, subroutine calls,
loops, conditions, locals. Inventing a decomposition is the hardest thing to
ask of a 350M model and something a 27B has no reason to do; *carrying out* a
stated structure is a small step at a time. Prompt-like tasks (judge,
classify, extract, rewrite) are the leaves of such programs.

**No anonymous lambdas.** Every sub-task is an instance of a function of the
acting lambda's code base. Under constrained decoding the function names,
parameter names and type-fitting input paths are enums: the model chooses, it
does not spell, and it never authors instructions, its weakest skill. And
because a code base without cycles is all there is to call, there is no recursion.

**One action to call.** `call(function, to, inputs, ...)` places and runs. A
placed-but-unrun instance, a separate `run`, continuations with parameters to
be produced later, and outside-in construction all disappear: straight-line
locals give the order. Calling again with only `function` and `to` resumes.

**The code base is immutable; variation is copy, edit, call the copy.** The
copy lives in a local, so edits are visible state with provenance, and the
original stays what the author wrote.

**The lambda holds every zone of state** (`args`, `let`, `return`, `codebase`
beside type and body), so swap-out, cold restart and provenance deal with one
object.

**Lexical scope, links in frontmatter.** A function sees its companion folder
and its `uses`. More verbose than inheriting the caller's functions, much
simpler to reason about, and it survives git, archives and Windows, which OS
symlinks do not.

Not yet implemented: `types.ts` beyond simple aliases.
