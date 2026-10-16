# Neuralese transport at typed value boundaries

A natlang function returns its declared value. A producer-written Neuralese
artifact is an opaque typed value; an ordinary string result is text. The runtime
must preserve that distinction and its artifact identity. It must never read a
payload to manufacture a string answer or disguise transport state as text.

Observed defect: `decodeTurnValue` turns native/legacy content parts into private
sentinel text before `coerce`. A declared string consequently accepts a genuine
native state reference as a literal opaque-ID answer. Four LF5000 task results in
`native-task-semantic-audit-root-20261016.json` exhibit this path. Existing typed
child-to-parent tests cover Neuralese results but omit ordinary-string rejection.

1. The model-turn service preserves structured transport parts in return/tool
   value leaves, recursively through records and lists. Plain text-only parts
   become text. Actual native and legacy block parts keep their IDs.
2. Declared result types guide raw string handling through records, lists,
   dictionaries, refinements and unions. Literal strings remain byte-exact.
   Explicitly Neuralese-typed stand-in producer literals use the existing writer,
   which retains both content parts and conversation text from one parse.
3. The existing exact coercer resolves a sole transport block only for a
   Neuralese slot (using its declared type/dialect), and rejects opaque transport
   at a string or other incompatible slot with the existing type-mismatch error.
   No implicit payload read or new result policy is introduced. Unknown can hold
   a sole shared Neuralese reference without exposing its payload.
4. Assistant conversation text and eval source retain sentinel encoding. Eval
   compilation continues using the existing typed literal calls. Literal raw
   strings containing marker-looking characters remain ordinary strings.
5. Focused tests cover both artifact formats, nested results and tool arguments,
   typed handoff, literal text, union/refinement cases, conversation and source
   compilation. A later live48-sample qualification uses admitted TRAIN tasks,
   records the exact runtime/actor/input split, and measures the existing error
   feedback before production adoption. Protected task packets are not training.

These are exact parsing, type verification and transport mechanisms. The policy
is the language's existing typed result contract, not a new prompt or heuristic.
