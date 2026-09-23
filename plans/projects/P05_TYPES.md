# P05 — Natlang type inference and checking

Status: the interactive application is Studio's P05 app
(`ts-host/studio/apps/`); the named programs in `codebases/type_studio` remain
as teacher corpus. The Python host application was retired with the Python
runtime. Live teacher quality and type diagnostic calibration remain unmeasured.

## Natlang prerequisites

Use C0 semantic functions and the current C7 structural type language. C3 must expose checked source definitions and exact type-fit operations through the eval environment. C4 can later supply observed execution evidence. No inferred type is automatically installed into runtime state, and this project does not require a second type system or full TypeScript checker.

## Programme and typed boundary

`infer.nl(sourceSnapshot, context) -> Suggestion[]` and `check.nl(sourceSnapshot, scope) -> Diagnostic[]` call `identify_calls.nl`, `infer_bindings.nl`, `propose_signature.nl`, `explore_path.nl` and `explain_conflict.nl`. Keep functions focused on one local definition and bounded context.

Use ordinary records for source spans, signatures, candidate bindings, obligations, alternatives and diagnostics. A suggestion records evidence and unresolved choices. A diagnostic records whether it is an exact incompatibility in a proposed call, a witnessed runtime failure, or a semantic hypothesis. None is represented as proof about every possible natural-language execution.

## Crisp environment

Expose illustrative `sources.definition(id)`, `sources.findCallers(id)`, `types.parse(text)` and `types.fits(a,b)` helpers. These wrap the existing loader/type machinery; they do not contain semantic inference. Search can scan the source snapshot directly before an index is justified.

Native parsed type/source objects can remain in the environment. Return stable textual/structural descriptions for the model. If a programme mentions another evaluator's native type, analyse its declared boundary contract; do not silently import that engine's entire type system into natlang.

## Reduction and stream shape

Finite Map analyses independent definitions; a bounded refinement pass incorporates caller evidence. Use explicit worklists or existing combinators because current codebase recursion is prohibited. The programme should stop with uncertainty rather than inflate its analysis into an unsupported complete proof.

An IDE integration later consumes an edit stream through Fold. Source revision accompanies every request and diagnostic; a stale analysis can remain historical evidence but cannot apply a patch to new text without revalidation. No live mutation of running source is necessary.

## Delivery and checks

1. Expose read-only exact APIs and analyse existing example codebases.
2. Remove signatures in copies and ask for candidate reconstruction. Gate: proposed types parse and stated exact obligations check.
3. Introduce binding/optional-field/return/effect mistakes. Gate: independently labelled diagnostics report evidence and avoid false certainty.
4. Integrate source-revision checking and optional trace evidence in P06.

Include ambiguous prose, unions, missing versus empty inputs, recursive data, rare branches and wrong callee names. Measure diagnostic precision/recall and useful abstention on held-out programmes.

## Trace and teacher

Trace which definitions/spans were read, which hypothetical calls were proposed and how exact checks answered. Captured runtime examples supplement rather than define the allowed type. Teacher training targets include alternative signatures and uncertainty; a type signature that happens to fit one trace is not automatically admitted as the correct general answer. This application can run over an in-memory source graph with no filesystem, stream service or host-type extension.

## Implemented boundary and next gate

The application masks a target's declared signature while retaining its body,
parameter names, named types, nearby signatures and frozen evidence. Natlang
proposes a candidate and possible calls; the host only parses, resolves named
types, applies the existing fit relation and checks witness IDs. This keeps one
type algebra in both the programme and checker. `consistent` means compatible
with supplied obligations, not complete inference. Evidence provenance belongs
to the caller: a forged evidence file can make an exact conditional check about
false premises. An IDE integration must build witnesses from trusted traces.

The fixture corpus groups variants by misconception, with train/eval separation.
The next gate is live teacher runs over held-out existing codebases, followed by
precision and abstention review before training samples are accepted.

The 20 September 2026 Bonsai pilot first exposed a CLI harness mistake: the
native constrained decoder treated Bonsai's chat-tool text as a reply. All four
application CLIs now share the repository's established Bonsai chat-tool adapter.
With that adapter, a P05 inference pilot reached `propose.nl` and read the
relevant body, parameters and named types, but did not finish within 180 seconds.
That observation is not grounds for shortening the algorithmic work. The
application evaluation harness now records each model request, complete raw
turns, and a live progress summary, with no default run limit. A fixed
signature fixture supplies an independent rubric. Admit a teacher trajectory
only after the whole inference completes and passes that rubric.

There is also an IDE integration boundary: the runtime loader requires a valid
declared signature before it can create a `CheckedGraph`. The application now
has `from_draft_text`, a read-only frontmatter/body view that accepts missing
signature types without installing an unchecked definition in the runtime.
Callers can supply a separate checked context graph. It does not yet recover
source spans from malformed, actively edited frontmatter; a proper editor
parser and span-preserving draft representation remain needed for that case.
