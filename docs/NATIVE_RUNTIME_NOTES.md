# Native runtime and training-capture notes (internal)

Recovery and contract notes for the native Neuralese server path: native state values and their archives, native
learning (graph gradients, adapters, optimizer steps), RL sampling receipts, in-run training guidance, the selected
semantic executor and learned compaction. They were accumulated in the natlang-authoring and natlang-integration
skills and moved here on 2026-11-12 so the skills stay a guide for writing and embedding ordinary natlang programs
(git history has the original placement). Nothing loads this file into a model. Paragraphs about the removed native
training controller (`foundation_ready`, adaptive `refresh_ready` publications, the shared semantic controller) were
dropped with it. The user-facing part (hosting a native server, passing `Neuralese<T>` values, student prompt
rendering) is `skills/natlang-integration/references/native-state.md`.

## Model-facing listings and tool output

Moved from the natlang-integration skill on 2026-11-13: these describe what the executing model sees, not what an
embedding host does.

- Native input and scope listings show at least 4,096 characters per value. A cut-off listing names the complete live eval variable; use ordinary code such as `value.slice(start, end)` to inspect omitted ranges. Tool-result text is sized from the call's context budget (`contextTokens`; `shownCharsFor` in `ts-host/src/native/cutoff.ts`: an eighth of the budget at 3.5 characters per token, at least 2,000), and only output longer than that is paged with `read_page`. A host that sets a larger `contextTokens` therefore shows longer outputs whole; `contextTokens: null` keeps the 2,000-character floor.
- If no tool output was cut off in the call, `read_page` directs the agent to inspect the in-scope value with `eval`; only an actual cut-off tool output creates a page ID.
- Eval exposes `inputs` and `read_inputs()` only for calls with formal input fields, unless those names are shadowed. A zero-argument call has no implicit caller-input object; ordinary local declarations may use those names.
- Error text a model reads is model-facing: measure a change to it on the live executor.

## Selected semantic executor

When the host selects semantic execution, await callable-folder module entry
and inline native boundaries. Captured mutable bindings are shared cells:
complete the current step before consuming the next Fold event, and use bounded
Map only for independent work. A capture-conflict means another writer changed
the binding; reread current state and retry the owning computation deliberately.
Do not retry a failed native write merely because the surrounding eval failed.
After a realm-authority change, restart the task and reload its modules instead
of keeping old exported functions. Keep source and invocation identities in
controlled fixtures, and distinguish those fixtures from learned-model results.

For a selected semantic executor, build a platform with `selectedSemanticPlatform(factory)` and pass it
to `createNatlangRuntime({ platform })`; it supplies the executor at both the fresh-task and retained-module
seams. Use `executeAsync` / `moduleInstanceAsync(platform, ...)`. The selected binding bridge awaits generated
inline capture accessors and guard callbacks; it never executes guest bodies natively.
Keep declared service return classes, resource identities and the actual
`EvalFailure` events/logs. Module instances belong to the platform value, so another platform never serves
this one's exports or callables; failed module initialization is not automatically retried. Native effects are
not rolled back by a failed eval. See the checkout's
`ts-host/SELECTED_SEMANTIC_INTEGRATION.md` for exact adapter surfaces and recovery.
For outside events use stable source/epoch and event identities, bounded ingress,
sequential Fold or bounded input-ordered Map, and explicit closure/cancellation.
Controlled selected trajectories establish engineering fidelity, not learned
execution quality; do not start training to fill an unfinished runtime seam.

## Learned compaction

- With learned compaction selected (a native Neuralese server and the `compaction` setting `nl`), a cut-off tool output is followed by `Compacted value: ` and a Neuralese state of all of it, and a cut-off argument's listing shows its compacted state; a `compact_history` note also carries `Earlier steps of this session, compacted: ` with the state of the replaced conversation. These states are summaries for orientation: exact details still come from the variable, `read_page` or `transcript`, as above.

- Learned compaction (plans/neuralese/LEARNED_COMPACTION_OPERATORS.md): a native server advertising `operator/1` serves `POST /v1/neuralese/operator` (`compact-value`, `compact-params`, `compact-history`, `extract`), which generates the operator's Neuralese span after its site (greedy, up to a per-operator cap) and returns its archived state; `nativeStateRuntimeOptions` wires it as `neuralese.compactor`. The native agent's pluggable `compaction` setting (agent option, `neuralese.compaction`, or env `NATLANG_NEURALESE_COMPACTION`) selects `crisp` (text cut only, the default until the live measurement adopts the learned path) or `nl`: cut-off tool outputs gain a framed compact-value state, cut-off argument listings use compact-params states, and `compact_history` notes carry a compact-history state. A failed operator call keeps the text cut and is traced as an `operator` event; the full value always stays reachable.

## Native state, learning and capture (authoring-side notes)

When a conversation carries a `native_state` content part, preserve its ID as an opaque exact-state reference through history and tool arguments. Same-model state can be spliced only when checkpoint, tokenizer, embedding and architecture identities match. A `neuralese-unknown-native-state` response means restore/upload that referenced artifact and retry the same request; an identity mismatch permits only explicit text retokenization, never reuse of its foreign tensors.

Tool-call argument JSON is decoded once before rendering, so a JSON string remains a string even when its contents resemble another JSON value. A native-state value nested in tool arguments is imported only when its ID is a complete canonical `nzs1_` content ID; keep the exact ID from the produced state receipt and restore that artifact before retrying.

Native positional-state artifacts are the existing typed `Neuralese<T>` values.
Pass the returned value directly to a child or parent; keep its content address
through captures and typed results. `neuralese-native-consumer-unavailable` means
the target must explicitly declare state adaptation or text reencoding before
consuming another model's native value. Preserve the source artifact and use a
configured native reader; do not paste its text to bypass the typed boundary.

Native computed Neuralese strings, including `""`, use the configured native
write port. Native `empty()` comes from a loaded native standard library's
model-bound zero-token state. For `neuralese-native-empty-unavailable`, rebuild
or load that library with its archived empty state. Contextual native body
initialization preserves its actual source capture and does not claim equality
with an ordinary-text prompt.

When hosting a server-backed native runtime, bind its real writer and reader
identity with `nativeStateRuntimeOptions({ endpoint, store, headers? })`; a store
alone cannot produce model-native state. If setup reports
`neuralese-native-text-write-unavailable`, use a server that exposes the forced
native text writer. Do not replace it with hash embeddings or projected token
values.

A native value from another checkpoint is read through its text
(`best-effort-text-retokenization`); keep the value reference intact. A
`neuralese-native-consumer-unavailable` error means the reader does not declare
text re-encoding: use a reader that does, or rewrite the value with the target
model's native writer.

Native graph learning targets explicitly named model parameters or consumed raw
KV/hybrid-state tensors. Native optimizer updates can publish backbone and
interface changes as a new checkpoint; reload that checkpoint and restore its
archived optimizer state before continuing. `neuralese-native-grad-order` means a graph
gradient request carried a derived or second-order computation; graph requests are
first order, so take a second-order gradient with `grad(f, a, { order: 2 })`, which
sends a derivative program. `neuralese-native-adapters-unavailable` means the call
binds weight adapters but its model server does not apply them; call it through a
native-state server that lists `native.adapters`.
If a native gradient/update binding error occurs, restore the artifacts for the
exact current checkpoint and repeat the request; never relabel tensors or reuse
an adapter against changed weights.
Portable native checkpoints preserve manifest identities and all parameter,
optimizer and training-state dependencies. If bundle verification reports missing,
changed or escaped artifacts, restore the complete verified bundle and the pinned
base model files; do not rewrite identity metadata or substitute another adapter.

Native guided chat now uses the same call/line guidance checks as ordinary chat.
Guided streaming publishes only the final committed reply, since a rejected
prefix cannot be retracted from a client. A
`neuralese-guidance-template-conflict` error means the requested forced template
violates guidance; choose a compatible call template or remove that forced
constraint. Keep the actual sparse `banned_token_ids` in each accepted sampled
action when retaining native RL receipts; forced template inputs are not actions.

Training guidance (JOLT+, memetic learning) is written in-run, not authored: for
each call of a training episode the runtime selects the verified facts that call
may see (`applications/guidance-writer` `selectGuidanceFacts`) and the policy
model writes guidance from them with `writeGuidance` as plain text; the runtime
writes that text as a native state, which the call imports after its leading
system turn. Do not add training facts to your instructions. A
`guidance-run-invalid` error names the malformed field of the run's
`x_natlang_guidance` control; regenerate it with rl/jolt.py `guidance_controls`.
A `training-guidance-unavailable` error means the run is not on a native-state
server; collect it with `--neuralese-server`.

Native pair conversion authenticates original provider/capture context and
selected targets before static role-span derivation. Missing earlier writer
evidence requires the complete original trajectory as context-only writer
input; it must not create extra targets. A source/value/writer binding error
requires the exact approved original record, typed block/name and payload hash.
Never edit old runtime receipts to match normalized markers, promote an
unapproved endpoint, or use a preview renderer to bypass those checks. Derived
receipts distinguish original source evidence from changed rendered bytes.

Existing reasoning boundaries keep their tokenizer IDs and spelling. Native
models may learn sparse input/readout overrides for `<think>` and `</think>`;
these are separate from the four Neuralese role markers. Old parents retain
exact behavior until the declared `--learn-reasoning-boundaries` upgrade. A
sparse-token checkpoint mismatch requires the exact checkpoint's token-ID
contract; do not rename markers or replace its tokenizer. An optimizer schema
mismatch requires the explicit sparse-row migration: added reasoning rows and
exact BF16/FP16-to-FP32 promotion of existing sparse masters are the only
permitted schema changes. Preserve moment values/RNG/exposure; never reset the
optimizer to bypass it. Sparse row shape/dtype errors require the exact saved
row contract, not broadcasting or silent casts. Reload actors from
the new actual checkpoint and rebind/requalify semantic communication after
updates. No new instruction text or reasoning-format workaround is needed.

Native functional adapters use exact-base named tensor coefficients and frozen
SVD bases. On a native service,
`nativeAdapters.create(nativeIdentity, {parameterNames, rank, kind})` selects
actual named layer matrices; retain the returned `NativeAdapter` and bind it
with `withNativeAdapters`. Native first-order graph learning uses
`nativeGraphGradient` with `target: {kind: 'adapter', id}` and the same active
adapter IDs, followed by `nativeAdapterOptimizerStep`; this does not enable the
legacy recorded-turn GradSession API or higher-order objectives. Keep the
coefficient, basis and optimizer artifacts together when saving or restoring.
`neuralese-native-adapter-invalid` requires restoring the exact coefficient and
basis artifacts for the current checkpoint and selecting its advertised matrix
names. `neuralese-native-tensor-as-state` requires binding adapters with
`withNativeAdapters`, rather than passing named tensor artifacts as communication
state. Native adapter policy learning retains sampling/3 receipts and a durable
actor bundle containing coefficients and frozen bases. Bind the exact same
ordered IDs/scales when evaluating its admitted native-policy objective with
`nativeGraphGradient`. `neuralese-native-actor-changed` means finish or cancel
the current invocation before changing its adapter binding. Mixed adapter
bindings across one replay graph require explicit per-invocation declarations
and a versioned immutable behavior policy. The typed native learning service accepts `invocationAdapters` with a
`native-policy-objective/2` binding admitted from actual sampling receipts. Keep
each invocation's ordered scales and frozen bases; current coefficient IDs may
advance through `nativeAdapterOptimizerStep`. On
`neuralese-native-grad-invalid`, restore its behavior bundle and match the
admitted graph, invocation inventory and basis/scales. External observations
require the original consumer occurrence and explicit `same-base-adapter-state`
when actors differ. Production full-backbone stage admission still rejects
adapted actors until its separate coefficient-policy stage is implemented.

Native coefficient learning may explicitly select
`adapterGradientPrecision: 'fp32'` in `nativeGraphGradient`. This retains the
actor's original forward storage values and carries coefficient cotangents in
FP32 through selected native Linear and dense-expert matrices. The default is
`storage`; changing this setting does not replace coefficients, bases or
optimizer moments. For an invalid precision error, select one of those two
advertised values. It does not enable heterogeneous adapter-policy graphs or
higher-order objectives.

An admitted native adapter-policy objective retains its original behavior
coefficients, sampled log-probabilities and observations. You may continue
coefficient updates against that objective with current coefficients on the
same exact base and ordered scales/frozen bases. Pass the current coefficient
IDs and exact-parent optimizer state; retain the original behavior bundle.
For a behavior scope/basis mismatch, restore those bases/scales or collect a
new admitted objective for the deliberately changed scope. Do not rewrite old
sampling receipts to the current actor identity.

Under default strict transport, changing coefficients while reusing genuine
external behavior KV requires a fresh rollout. Internal producer state is
replayed under current coefficients; external KV cannot be silently relabelled.
An explicitly supported/requested same-base adapter-state mode is required to
retain external KV across actor changes.

Native adapter-state exchange: an explicit `same-base-adapter-state` declaration
may import raw state from another adapter actor only on the exact same base,
tokenizer, embedding and architecture. For `neuralese-native-state-mode`, restore
the source coefficient/basis artifacts and correct the exact input-ID declaration;
use learned semantic adaptation for different bases. Use this mode only when the server advertises it in native-state consumption
modes; matching tensor dimensions alone never establishes support.

An explicit `same-base-adapter-state` contextual import keeps distinct source
and target actor receipts. Preserve `native_state_modes` when recording or
replaying the call. For `neuralese-unknown-native-artifact` during restoration,
restore the source actor's exact coefficient artifacts and frozen bases alongside
its raw state; restoring only the state cannot authenticate the producing actor.

Native graph coefficient learning may declare `invocationAdapters` by exact
invocation name. Continue from the returned `invocation_actors` receipt when
updating several call actors; a heterogeneous graph has no single effective
model identity. `neuralese-native-grad-invalid` for an invocation declaration
requires actual graph node names and exact-base coefficient/basis artifacts.
Keep protected test rows out of gradient requests. Heterogeneous policy-objective
admission/replay remains unavailable until its per-turn behavior contract is
implemented and qualified; graph CE support does not imply that RL support.

A declared native coefficient production policy binds generation and decisions
to the same current actor. `neuralese-native-coefficient-policy` means restore
the exact declared policy and coefficient archive; retry with that binding.
Stable adapter handles are mapped by the declared policy, so retain those handles
in the program rather than substituting an old coefficient artifact yourself.

A produced Neuralese value remains an opaque typed reference. For a string-slot
`type-mismatch`, return actual answer text, or pass the reference to a declared
`Neuralese<T>` reader that returns the requested type. Do not return an artifact
ID or internal marker as answer text. Literal string contents remain literal.

Native services advertise direct `xs`/`tiny` adapters after integrated service
qualification. Supply explicit `parameterNames` selecting actual native layer
matrices and retain their exact coefficient/basis/optimizer archives. On an
invalid native adapter request, correct those names or restore the matching
checkpoint closure. Native `grad_order: 1` remains first order; advertised
adapters and same-base state exchange do not imply higher-order gradients,
legacy projection decoders or cross-base raw KV compatibility.

A native imported-state gradient can be applied with the typed host operation
`nativeStateOptimizerStep`. Its result is new KV/hybrid state, with the original
producer preserved as origin and exact gradient, update and optimizer-history
artifacts retained. Keep its full `.nz` dependency closure when saving or moving
it. For `neuralese-native-optim-invalid`, restore the exact parent state and
matching named optimizer history; explicitly omit history only for an intended
new optimizer. A compact inference view cannot construct optimizer updates;
restore its complete source parent before requesting an update.
Do not relabel an optimized state as a fresh producer capture or
replace missing state with decoded text. Public automatic `grad(...,{order:2})`
remains unavailable until its native recorded-turn/DAG path is qualified.

Optimized KV/hybrid states bind their exact scalar optimizer arithmetic runtime.
For a native optimizer arithmetic/runtime mismatch, restore the state with its
bound CPU Torch/runtime; do not relabel it for another executor or discard its
history. Archive byte integrity is distinct from numeric optimizer verification.

Native recorded learning needs the actual server input/write receipt and a durable
archive. For `neuralese-native-grad-invalid` reporting a missing receipt or
conflicting turn occurrence, keep each recorded call with its own execution
graph and use the matching native server; do not reconstruct producer edges
from equal text or artifact IDs. For `neuralese-native-archive-unavailable`,
configure a store supporting native artifacts before recording communication.
Greedy structural receipts have no behavior probability and cannot be used as
on-policy RL samples. This capture mechanism does not advertise public order-2
GradSession support.
Assistant-only producer blocks follow the same typed result checks as tool returns.
For an ordinary string result rejected with `type-mismatch`, read the imported
Neuralese through a typed natlang call and return its actual answer text; an
internal artifact ID is not the answer. Explicit Neuralese returns retain the
original block. Literal text that merely spells a marker remains literal.

When a server reports an actual native-state content part as the assistant's
final completion, the declared return type determines its meaning. A compatible
`Neuralese<T>` result keeps the typed state reference. An ordinary string or
JSON-compatible result reads the state's authenticated stored text and passes
that value through the usual exact result check. Mixed text and state parts keep
their reported order. This applies to inferred assistant completions; an
explicit `return_result` call still follows its existing value validation.

For a `neuralese-native-grad-invalid` from a decision objective or other native
objective validation, provide finite numeric rewards or nonnegative finite probabilities with
positive total mass, exactly one entry per declared option. Keep the recorded
conditional token queries bound to their actual graph and avoid assigning a
likelihood to a forced input after an imported state.

For an explicit native query readout binding error, restore the complete immutable
readout snapshot for the matching base and layout, including final norm, tied
vocabulary table and sparse rows. A teacher hidden state alone does not select
a different teacher head; stop the declared teacher distribution. This does
not enable arbitrary heterogeneous full-backbone parameter overlays.

For a native derivative objective-origin error, restore the exact declared
program, graph and bound values. Older internal program gradients without an
exact graph receipt need a genuine reexecution; do not synthesize or relabel
provenance. This stronger program check does not invalidate ordinary prior
first-order state/gradient history.

Recorded native decisions need the exact rendered option receipt and native
input archive. For a missing decision receipt or archive error, use the matching
native scoring service and restore its original/decoded state dependencies;
never substitute a sampled text reply or infer producer links from equal IDs.

For a serving error `System message must be at the beginning`, preserve the
ordered system/recovery messages and use a qualified compatible server template.
Do not move recovery instructions into the initial prompt or change their roles.
The pinned Qwen compatibility renderer preserves originally valid renderings;
its changed recovery path needs live executor qualification before adoption.

For a whole-episode native gradient error, restore all original model turns and
their execution-graph write/read occurrences before re-admission. Correct plain
execution remains rewardable; do not add Neuralese traffic to pass a gate or
replace an omitted in-episode producer with detached archived state. Decoded
imports need their actual differentiable lowering, not relabeled provenance.

`stopGradient(value)` preserves native typed values and soft-function bodies as
constants for that use, including after context snapshots and `.nz` save/load.
An equal live value remains differentiable. Keep the typed value when passing
it to another call or `withNativeAdapters`; do not extract its artifact ID to preserve
a constant boundary. No payload is decoded or rewritten by this operation.

Native Neuralese values retain their source-call occurrence through returns,
context snapshots and `.nz`; equal artifact IDs may come from different calls.
Pass the original typed value rather than rebuilding a reference from its ID.
If native graph verification rejects ambiguous origins, restore the complete
recorded episode and its typed source references; do not substitute a detached
external state for an in-episode producer.

Native marker failure recovery: when the native executor reports
`neuralese-native-state-open`, `neuralese-native-state-close`,
`neuralese-native-state-consume-marker` or `neuralese-native-state-nested`,
that sampled reply executes no tools and supplies no return value. Repair the
same call with a complete native value or its declared ordinary result. Any
completed writes from the failed reply retain their actual provenance; an
unfinished span is never a communication block. Do not copy an opaque state ID
into an ordinary string result. The runtime retains the failed draw for graph
learning instead of silently retrying the model transport.

Native state IDs are complete content IDs. A shortened ID copied into a tool
argument stays ordinary data for the declared result type; it cannot identify
a cached state. Recover from the normal type error by returning the required
ordinary value or by using the complete original state reference. Never guess
a full ID from a prefix. Actual complete references keep artifact restoration
and provenance checks.

Recorded callable Neuralese returns retain the sampled instruction body and
their typed runtime reference. If conversion reports an unlinked callable
return, restore the complete original raw tool call, invocation output and
block-write trace; do not replace the sampled body with an artifact ID or
invent a successful outcome. The converter authenticates the exact declared
callable type, body hash and producing turn before linking the return.

History reasoning follows the checkpoint's bound rendering profile. An
explicit `history_reasoning_policy: keep` makes earlier assistant reasoning
visible in complete conversation history; it does not change the original
source targets or authorize rewriting old graph receipts. If a saved checkpoint
or tokenizer reports a conflicting history policy, restore the matching
checkpoint/tokenizer binding rather than forcing a different template switch.
The opt-in profile remains unmeasured until its live executor comparison is
recorded.

For a function declared to return `Neuralese<string>`, a final assistant text reply is the exact text value. The runtime writes it through the configured Neuralese writer and returns the resulting typed reference. Intermediate text that leaves the invocation without a result is incomplete; finish the same invocation with a typed value instead of treating that text as success.

## Native state, learning and capture (host-side notes)

For native shared-state turns, keep `{type: 'native_state', id}` references structured through message history and nested tool-call arguments. Before a request, restore each referenced artifact from the runtime store; if the server reports `neuralese-unknown-native-state`, fetch the same ID and retry without rewriting its arguments. The server may splice KV/hybrid state only when the exact model identities match; a foreign block uses its stored text for target-tokenizer re-encoding and discards foreign tensors. Retain the `natlang.native-rl-sampling/1` receipt and copy every referenced produced state artifact before releasing the serving owner.

Tool-call JSON text is decoded exactly once while building model history. Preserve scalar strings as strings, and treat a nested native-state reference as importable only when it carries the full canonical `nzs1_` ID. If a state ID is incomplete, keep the source argument unchanged and recover the complete ID from the original state receipt before retrying.

Native `nzs1_` artifacts use the same typed Neuralese reference and ownership
lifecycle as other Neuralese values, with an exact multi-tensor codec. Configure
a native-capable archive and bind the reader's checkpoint identities and actual
consumption modes from `/v1/neuralese/info`. On
`neuralese-native-archive-unavailable`, use a store implementing native-state
persistence. On `neuralese-native-consumer-unavailable`, bind a reader declaring
adaptation or reencoding. Named dialect types remain strict; a best-effort
consumption receipt is not exact KV interchange or a learned semantic dialect.

For native forced writes and contextual body initialization, use the native
text-write/init-body endpoints and archive the returned artifact plus every
native import in `source_capture`. Preserve that receipt in library provenance.
A `neuralese-native-source-integrity` error requires repairing the mismatched
source identity, payload or role markers before retrying. Native views retain
all chunk/combine `source_turns`; archive their imports and outputs before use,
and retain the observer receipts when running outside an invocation trace.
For a server-backed `NativeRuntime`, configure the complete binding with
`nativeStateRuntimeOptions({ endpoint, store, headers? })`. Passing only the
store enables archival but leaves typed text writes without a producer port;
`neuralese-native-text-write-unavailable` means the server lacks the required
producer capability, so use a compatible native server rather than a stand-in
embedder or legacy projection. Callers that only consume native state may pass
`{ write: 'available' }` and must not claim they can synthesize new state.
Native empty states have zero token rows and full model/layer identities;
`neuralese-native-empty-unavailable` means load the native library's empty state.
When a body placeholder shares its content ID with another imported value,
`neuralese-native-state-placeholder` means pass `placeholderOccurrence` with the
zero-based matching native-part occurrence of the intended body; other matching
parts retain their actual imported state. Never replace all equal content IDs.

Native state from another checkpoint is consumed through
`best-effort-text-retokenization`; only the exact identity splices raw state.
Contextual `source_capture.imports` binds original references and
`observed_native_state_id`; preserve and archive both when publishing a `.nz`
body.

Native graph learning preserves named tensor artifacts through the normal store.
`nativeOptimizerStep` accepts actual backbone or interface gradient names and
publishes a new immutable checkpoint; reload it and restore the returned
optimizer-state artifact for continuation. SGD supports explicit momentum.
Graph gradient requests are first order: `neuralese-native-grad-order` means a
derived or second-order computation was sent as a graph request; send it as a
derivative program (`grad(f, a, { order: 2 })` does). `neuralese-native-adapters-unavailable`
means a call binding weight adapters reached a transport or server that does not
apply them; route it to a native-state server that lists `native.adapters` instead
of sending an adapter value that would be ignored.
For `neuralese-native-gradient-binding` or `neuralese-native-update-binding`,
restore the exact current checkpoint's gradient/update/optimizer artifacts and
retry; never change identity metadata to make unrelated tensors appear compatible.
Move native learned actors with `python -m natlang_neuralese.model.native_bundle
export --checkpoint CHECKPOINT --out BUNDLE`; verify with the same module's
`verify --bundle BUNDLE`. Use the returned checkpoint path and an explicit identical HF
base location when relocated. Missing/changed/escaped dependency errors require
restoring the complete verified bundle, never changing its identity declarations.

Native services advertising `guidance` restore the full prompt KV/hybrid-state
anchor and replay accepted own-token roles after a rejection. Guided streams
buffer until the reply commits; consumers must not infer a stalled request from
absent early deltas. Preserve `natlang.native-rl-sampling/2` inputs, own positions,
write coordinates and each action's sparse `banned_token_ids` unchanged for
normalized-mask replay. Never substitute the unmasked model log-probability.
`neuralese-guidance-template-conflict` requires a compatible forced call template
or removal of the conflicting forced constraint, rather than retrying unchanged.

Native adapters share the existing xs/tiny coefficient algebra and immutable
named tensor transport. `nativeAdapterCreate` requires the exact native identity
and named layer matrices; `nativeGraphGradient` accepts active `adapters` and
`target: {kind: 'adapter', id}`. `nativeAdapterOptimizerStep` returns new
coefficients and an exact-parent resumable optimizer state. Archive the frozen
basis dependency, not just the coefficients; `.nz` native tensor inventories
preserve this closure and wire metadata numeric spellings. Existing artifact
IDs do not change when transferred between Python and JavaScript.

An adapted producer exports raw state bound to the base identity and exact
ordered adapter IDs/scales. Its effective checkpoint hash differs from the base;
a consumer must apply that same binding to splice it exactly, or explicitly
reencode its text. Never reuse a base-bound semantic decoder as if it had been
qualified for changed weights. For `neuralese-native-adapter-invalid`, restore
matching coefficients/bases and use actual native matrix names. For
`neuralese-native-tensor-as-state`, bind the adapter with `withNativeAdapters` instead
of sending it as communication state. Adapted-policy sampling/3 receipts bind
an effective actor SHA, exact base identity and ordered coefficient IDs/hex
scales. Retain its recursive RL artifact bundle, then evaluate the admitted
native-policy objective with the same adapters and a coefficient target. Keep
forward, checkpoint recomputation and backward inside the native adapter scope.
`neuralese-native-actor-changed` requires finishing or cancelling that invocation
before changing bindings. Heterogeneous coefficient-policy requests use explicit `invocationAdapters`
and an admitted `native-policy-objective/2`, with immutable per-call behavior
receipts and a descriptor-owned bundle. Preserve original behavior coefficients
as well as current actors when archiving gradients. For
`neuralese-native-grad-invalid`, restore the exact bundle, actual invocation
occurrences and frozen basis/scales; changed external consumers need explicit
`same-base-adapter-state` and their authentic original consumption receipts.
Full-backbone adapted actor export/rebinding and production coefficient-stage
scheduling remain explicit unavailable paths. A store lacking named tensor support
must report `neuralese-native-archive-unavailable` before loading these artifacts.

`nativeGraphGradient({adapterGradientPrecision: 'fp32', ...})` sends the exact
`adapter_gradient_precision` operational setting. It preserves the existing
forward weight storage/kernel and retains FP32 coefficient cotangents instead
of rounding an intermediate dense-weight adjoint to BF16. Artifacts record the
selected arithmetic; default `storage` retains prior behavior. Preserve the
same coefficient/basis/optimizer closure across continuations. Invalid values
require selecting `storage` or `fp32`, rather than dropping requested adapters.
This setting does not advertise mixed per-invocation actors or changed-base
adapter rebinding.

Native adapter GRPO/RFT separates captured behavior from the current evaluated
coefficients. Continue the same admitted objective with updated coefficient IDs
while preserving its exact base, ordered scales and frozen bases. Gradients
record both identities; external observations retain behavior identity and
internal producers replay under current coefficients. Preserve original bundle
bytes and pass the current optimizer-state parent for each update. A behavior
scope/basis error requires restoring the original scope or explicitly collecting
new behavior, never relabelling old actor receipts. Changed-backbone rebinding
and heterogeneous invocation policies remain separate unsupported contracts.

Native adapter-state transport uses `x_natlang_native_state_modes` on chat and
`native_state_modes` on scoring/gradient/context-init requests, mapping actual
input IDs to `same-base-adapter-state`. A mode error requires restoring exact
source coefficient/basis closure or correcting the base/input declaration; never
retry by silently dropping the declaration. Preserve source and target actor
bindings and consumption receipts in archives. Require the server to advertise this exact native-state consumption mode;
restore its complete source coefficient/basis closure before importing.

Native coefficient production requests may carry the typed
`x_natlang_native_coefficient_policy` control. Restore its exact policy/archive
and verify the server's full policy attestation before generation or decision.
`neuralese-native-coefficient-policy` requires restoring that exact binding;
never discard the requested policy or evaluate the base actor as a substitute.
Coefficient checkpoints preserve a full native base separately from coefficients,
named optimizer state and RNG/exposure. A backbone change creates explicitly
new bound artifacts and requires fresh rollout collection; old KV and behavior
must not be relabeled. Explicit stale-row cancellation retains completed credits
and records untrained visits as uncredited, then recollects on the new base.

For an inferred assistant completion containing native-state parts, the runtime
uses the function's declared result type: compatible Neuralese results retain
the typed reference, while ordinary results materialize each part's authenticated
stored text in order and then use the normal exact return checker. Mixed text
and state parts preserve their order. This rule does not change explicit
`return_result` calls or weaken type validation.

Native protocol recovery is negotiated by `native.protocol-recovery` and the
boolean `x_natlang_native_protocol_recovery` request field. Treat a response's
host-verified `x_natlang_native_failure` as one non-executable model turn,
retain its complete sampling/4 (or greedy structural-capture/2) receipt and
completed artifacts, then give the same call the typed delimiter feedback.
Never reconstruct an unfinished block or execute a parsed prefix tool call.
The open/close/consume-marker/nested error codes mean repair the native reply
or return the declared ordinary type. A `neuralese-native-protocol-integrity`
error instead requires repairing the client/server receipt or restoring a
matching runtime; model-generated text cannot supply a trusted receipt.

The native checkpoint's optional `history_reasoning_policy: keep` must bind
the same tokenizer template in training evaluation and serving. Old
checkpoints without the field use the backbone-declared default. If the
checkpoint and tokenizer disagree, load the matching tokenizer/template or
restore the original checkpoint profile; do not override the template locally.
The opt-in profile changes model-visible history and needs its live executor
comparison before production adoption.

When the declared result is `Neuralese<string>`, route a terminal plain-text assistant reply through the existing typed Neuralese writer before completing the invocation. Keep the target as the original assistant text and bind the writer receipt to that exact model turn, body hash, and captured invocation result. A no-call reply followed by another turn in the same invocation is not a completed result.

Native `eval` rejects code that is not TypeScript before running it, with the scope compiler's `typescript-syntax`
diagnostics. For Python-shaped code or prose inside the `code` argument the first diagnostic is a recovery sentence
(`The code argument is run as TypeScript in this call's scope. ...`) naming the offending line. It is attached only
after the code failed to parse, so valid imports of project packages, private fields and ordinary syntax slips keep
their plain parser messages. A collector or evaluator that counts repeated failing evals should treat this diagnostic
as the single teaching signal and compare the next action: a different, parseable action is a recovery.

## In-run training guidance

A collection's `--chat-request-config` may carry `x_natlang_guidance`
(`natlang.guidance-run/1`, from rl/jolt.py `guidance_controls`, with
`x_natlang_rl_capture: true`). The collector hands it to the agent and never sends
it on the wire. On each call's first model turn the runtime keeps the facts that
call may see (`applications/guidance-writer/visibility.ts`), selects them
(`selectGuidanceFacts`, crisp or NL by `selection`; a failed or invalid NL
selection falls back to crisp and is recorded) and runs `writeGuidance` in-process
as a child call on the same model and recorder; it returns plain text and the
runtime writes that exact text as the native state (`typed-text-result`, bound to
the writer's final turn). A teacher imports the written state after the leading system turn, a student only writes it, and a failed or
switched-off writer gives a teacher its facts as text and records the same text
turn for a student's training twin (`text-twin`; the student never sees it). Recovery: `guidance-run-invalid` names the bad
field (regenerate the control); `training-guidance-unavailable` means use a
native-state server (`--neuralese-server`); a `writeGuidance returned text without exactly
one recorded writer call` fallback means the writer ran outside its consumer (report it as a runtime bug);
`guidance-writer-missing` means set
`NATLANG_GUIDANCE_WRITER` to the package directory.
