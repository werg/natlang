# Bonsai `scope-eval-v1` failure analysis

Date: 2026-09-22

Model: `Ternary-Bonsai-2-27B-PTQ1_0`

Batch: the first ten interleaved cases from the 96-program coverage selection.
Generation was stopped before case 11 because none of the first ten results was
admissible.

## Summary

The ten cases produced one completed but semantically incorrect value and nine
quiesced executions. Across their traces the runtime recorded 377 actions. The
non-successful actions included 28 lexical redeclarations, 26 forbidden dynamic
imports, 12 propagated child quiescences, 10 invalid read ranges, eight wrong
record-field writes, six unavailable imported calls inside pure eval, five
stale pending locals later observed as `Null`, and five awaited calls that fell
through to the pure evaluator.

Those counts include retries and cascades. They do not mean there were 96
independent causes. The primary causes are:

1. The text preview and `read_value` range contract contradict one another.
2. Pure eval persists and infers every declared temporary without enough
   contextual type inference.
3. Imported functions are not as generally callable as the advertised lexical
   TypeScript surface implies.
4. The inspection prompt does not explicitly name the file tools or forbid
   module imports, producing a repeated filesystem-import reflex.
5. Completion feedback after `return_value` does not show the still-open lines.
6. Identical retries of a failed semantic child repeat the same deterministic
   mistake.
7. Bonsai still makes genuine semantic, schema, and line-marking mistakes.

## Case analysis

### 1. Dependency planner

Final symptom: `let/state` was reported as `Null` instead of `State`.

The first iteration completed correctly and chose `t3`. During the second
iteration the `choose` child evaluated several `.find(...)` expressions. One
temporary was a `Task` whose `needs` field was an empty list. The persistent
scope attempted to infer this temporary's complete structural type from its
runtime value and rejected the empty list as ambiguous. The temporary was not
the returned value, but atomic capture validates every declaration.

The child then quiesced. Later exploratory actions encountered pending locals
as `Null`, obscuring the original empty-list inference error. The teacher also
tried to call `ready_tasks` from the root even though it belongs to `step`'s
lexical codebase; that part is a model scope mistake.

Classification: runtime type-inference defect first, followed by weak failure
diagnostics and model recovery mistakes.

Required fix: infer element and selected-record types through array indexing,
`.find`, `.filter`, `.map`, and `.slice`, or avoid requiring value-only
inference for unused temporaries. Preserve the original child failure instead
of later reporting its pending destination as `Null`.

### 2. Source highlighter

Final symptom: the program completed, but the role list differed from the
reference at five lines.

The root correctly mapped `highlight_file` over its input. The child then
bypassed its `line_role` semantic helper and wrote all roles itself. It changed
several labels after an attempted redeclaration of `roles` failed. Two errors
are clear under the supplied rubric: it labelled an explicitly exact line as
`call_each`, and labelled the empty-record assignment as `exact` rather than
the expected prose step. Three `call` versus `call_each` disagreements expose
a narrower ambiguity in the reference wording, but the program explicitly
required calling `line_role`, so the trajectory is not admissible regardless.

Classification: model inlined semantic work that the program assigned to a
helper; the reference bank also deserves a wording review for loop-call roles.

Required fix: train contrastive examples in which semantic helpers must be
called rather than imitated. Review the role rubric and oracle for `for x in
items do f(x)` versus literal `for each` syntax.

### 3. Legal move

Final symptom: `read_value(position, start=0, end=123)` was rejected because
the string was treated as a one-item range.

The first read displayed:

```text
"X holds ..." … (123 chars)
```

The tool schema describes `start` and `end` as a zero-based JavaScript-style
slice. Requesting characters `0..123` was therefore a rational response. The
implementation instead routed Text through a line/item range and said the
valid extent was `0..1`. The model had no way to retrieve the omitted text.

Classification: direct interface bug, not a substantive teacher error.

Required fix: make Text ranges character ranges, add separate line arguments,
or return a short single-line input in full. The preview must never advertise
a character count that the read API cannot address.

### 4. Mail rules

Final symptom: the first mapped decision became `Null`.

The `from_landlord` child hit the same truncated-string/range contradiction as
the legal-move case. Bonsai then repeatedly redeclared the lexical input names
`email` and `isWrittenByLandlord`, causing ordinary JavaScript redeclaration
errors. It eventually reasoned to a Boolean, but the original child invocation
had already quiesced. The enclosing `handle_mail` and map subsequently exposed
the failed destination as `Null`.

Classification: range-interface bug is the first cause; input shadowing and
poor recovery are model mistakes; the final `Null` message is diagnostic loss.

Required fix: repair Text reads, explicitly teach that inputs are already
lexical variables and cannot be redeclared, and preserve the child error at the
parent boundary.

### 5. Moderation

Final symptom: no root result was staged.

The four `violates` leaves frequently redeclared `rule` and `post`, but they
recovered by using different local names and produced decisions. The single
`severity_of` leaf produced a plausible `low` value but left instruction lines
1 and 2 open. The mapped severity call therefore quiesced and the root never
reached its warning return.

Classification: model line-closure failure, amplified by weak completion
feedback. Lexical input redeclarations added wasted turns but were not the
final cause.

Required fix: add explicit training for prose spread across multiple numbered
lines and report remaining line numbers immediately when a value is staged.

### 6. NLProlog

Final symptom: the run ended after a forbidden dynamic import.

The `is_rule` leaves repeatedly redeclared the existing `statement` input,
causing 19 of the batch's lexical redeclaration errors. At the root, Bonsai
then tried to use `select_by_flags` and `is_rule` inside ordinary expressions.
The advertised architecture says imports are ordinary callables, but the
current runtime only recognizes selected awaited declaration/map/fold/repeat
forms. In a general pure snippet the helpers are undefined. Alternative await
forms fell through to the non-async evaluator, and the model finally attempted
a dynamic import.

Classification: major runtime/contract gap plus severe model thrashing.

Required fix: either provide actual callable import bindings to restricted
eval or compile calls with a real parser rather than regex-shaped top-level
forms. Teach the model not to redeclare inputs. Retain the import prohibition.

### 7. Order saga

Final symptom: after a correct state and successful effect, the model attempted
to mark nonexistent line 12.

The child completed the semantic work and staged the correct result. It marked
lines 3-4 skipped and 7-11 done, but forgot completed lines 2, 5, and 6. It then
called `return_value` 21 times. Each response merely said the typed result was
valid; it did not say which lines remained open. The tool set correctly stayed
available because completion was not legal. Bonsai eventually guessed line 12
instead of inspecting or closing 2, 5, and 6.

Classification: model line-accounting mistake made much worse by unhelpful
post-return feedback.

Required fix: every `return_value` response should say `result staged; lines
still open: 2, 5, 6` when applicable. Add repeated-return detection and
line-closure training. Do not silently complete the program.

### 8. Reconciliation

Final symptom: all twelve `assess` children rejected a dynamic filesystem
import and the labels map quiesced.

Every semantic leaf generated essentially the same
`import('fs/promises')` exploration snippet. The system prompt encourages
proactive codebase inspection but its opening tool list omits the names of the
file tools, and it never explicitly says that module imports are unavailable.
The model repeated the same reflex in every independent child. Parent retries
of a single item reproduced the same failure.

Classification: prompt-induced model behavior plus deterministic retry waste.

Required fix: explicitly name `list_files`, `search_files`, `read_file`,
`write_file`, `edit_file`, and `diff_files`; say to use them for `codebase/`
and never import filesystem modules. A retry of an unchanged failed child
should not rerun with an identical seed and state indefinitely.

### 9. Shopkeeper

Final symptom: `intent` was later reported as `Null`.

The step first called crisp `goods_of` synchronously inside pure eval and
received `ReferenceError`; adding `await` used the supported call lowering and
succeeded. The `read_intent` semantic child then used the same forbidden
filesystem-import reflex as reconciliation. Three identical retries failed,
after which the pending `intent` destination surfaced as a misleading `Null`
type error.

Classification: prompt/tool discovery failure, restricted-call inconsistency,
deterministic retry waste, and diagnostic loss.

Required fix: the same import and child-failure fixes as reconciliation, plus
real ordinary callable semantics or an honest statement that all imports must
be awaited.

### 10. Webserver

Final symptom: a final `edit_file` did not find exactly one source span.

The root first attempted a large ordinary control-flow eval that the restricted
surface could not parse, then split the work into smaller calls. The
`review_submission` child correctly judged the form but constructed a record
with `accepted` instead of the declared `accept` field. Validation caught the
mistake. The parent accurately diagnosed it, but retried the same deterministic
child many times and obtained the same malformed record. It then used the
desired live-codebase repair path: edit the child instructions and retry. The
exact search text crossed a line wrap and did not match; the model omitted
`fuzzy: true` and the diagnostic did not say whether there were zero or
multiple matches.

Classification: genuine model schema error; deterministic retry policy and
edit diagnostics prevented an otherwise promising recovery. The unsupported
large control-flow snippet is a runtime surface gap.

Required fix: add exact-field contrastive data, vary or explicitly distinguish
retries, report exact-edit match counts, suggest fuzzy editing on zero matches,
and support common branch-shaped eval programs.

## Cross-cutting conclusions

### The former regex evaluator violated the TypeScript contract

The diagnostic run showed that several apparently irrational model attempts
were valid TypeScript rejected by a regex lowering layer. That layer has now
been removed. Eval and crisp code use the same TypeScript compiler and Node host;
imported natlang functions cross a checked async host bridge from arbitrary
expressions and control flow.

### Caller-level validation is exposing useful evidence

Disabling local repair did what was intended: malformed ranges, fields, line
marks, and edits remained visible rather than being silently coached into
success. The experiment also shows that parent agents need precise child
failure values. Converting a failed destination to `Null` loses the useful
evidence without preventing cheating.

### Proactive codebase editing is viable

The webserver parent noticed the exact child schema mistake and attempted to
repair the existing `.nl` file. That is the desired behavior. The repair failed
because the edit span crossed formatting and fuzzy mode was omitted. The
correct response is better file-tool instruction and diagnostics, not removing
codebase edit authority.

### The full batch should remain stopped

The current traces are valuable diagnostic and negative examples, but none is
an admissible successful trajectory. Resume broad generation only after focused
probes cover Text reads, temporary type inference, general imported calls,
post-return line feedback, semantic-child failure propagation, and codebase
repair.

## Recommended fix order

1. Repair Text range semantics and post-`return_value` line diagnostics.
2. Preserve child failure details and prevent pending destinations from being
   reinterpreted as `Null`.
3. Add contextual type inference for common array operations and selected
   records.
4. Make imported functions genuinely callable throughout restricted eval, or
   replace the current regex lowering with a small parser/compiler.
5. Rewrite the inspection paragraph to name the file tools and prohibit module
   imports while continuing to encourage relevant source inspection and live
   edits.
6. Improve exact-edit diagnostics and explicitly suggest fuzzy mode.
7. Decide retry semantics: deterministic replay should require changed state,
   source, or an explicit varied retry rather than repeating the same child.
8. Generate contrastive training examples for input shadowing, helper use,
   exact record fields, line closure, honest failure, and successful codebase
   repair.
9. Rerun one focused case per failure class before restarting the 96-program
   and 132-Studio coverage runs.
