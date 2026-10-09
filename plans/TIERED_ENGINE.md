# Tiered execution engine

Prototype of ARCHITECTURE_IMPROVEMENT.md §E1, integrated with the trace specializer (TRACE_SPECIALIZATION.md): one
architecture, not two. One natural-language function runs at a ladder of tiers, chosen per call, with guards and
deoptimization. The ladder is built from what the specializer stored for the function; evidence and promotion are one
module shared with the compilation cases. Code: `ts-host/src/calls/tiers.ts` (engine, tiers, ledger),
`calls/evidence.ts` (evidence, rule, decisions), `runtime/promotion.ts` (policy selection, review),
`applications/specializer/promote.nl` (the natural-language policy), a small hook in `runtime/kernel.ts`
(`tieredCall`, `InvokeOptions.tierAttempt`), `tierRows` in the store, `natlang traces tiers`.
Opt-in: `createNatlangRuntime({ tiers: new TierEngine({ functions: { name: settings } }) })`. Without it, or for a
function it does not name, nothing changes. The case-promotion changes below are neutral for users without a tier
engine: the default policy is the old rule.

## Tiers and what is behind them

The specializer's compilation is crisp TypeScript with an input guard per behaviour group (`cases.ts`: `when` and
`run`). That is exactly what the first design called tier 3 ("crisp code behind a guard"), and the first prototype's
"tier 2, compiled cases" was the same artifact reached through the store. The two were one tier with two entrances, so
they are collapsed: **compiled cases are the crisp tier, 3**. What the first design lacked was the tier between the
student model and crisp code: the student model under trace-specialized instructions or few-shot examples. That is tier 2.

| Tier | Class | Served by | Guard before | Check after |
| --- | --- | --- | --- | --- |
| 0 | `ModelTier(0)` | the function's own model, or `models[tier0.model]`; the kernel's normal agent path | none (terminal) | declared type and refinements (kernel); failure is recorded, not retried |
| 1 | `ModelTier(1)` | `models[tier1.model]` (the `model: NAME` frontmatter mechanism, set per attempt) | profile exists | kernel checks, optional `tier1.verify`; shadow comparison to promote |
| 2 | `SpecializedTier` | the student (or own) model with the compilation's `instructions.md` added to the system prompt (`TierAttempt.guidance`) | profile exists | kernel checks, optional `specialized.verify`; shadow comparison |
| 3 | `CrispTier` | the compilation's active cases through the kernel (`admit`, `runCrispCase`) | the case's own `when` and its case state (`active`) | refinement check by the kernel, optional `crisp.verify` |
| 3 | `ImplementationTier` (`tier3:<id>`) | a hand-registered `CrispImplementation` as a synthetic case (hash `tier:<id>`) | `when(args)` | `verify(args, out)` before serving, shadow replay |
| 4 | `NeuraleseTier` | stub: `canServe` false, `serve` throws | | |

Hand-registered implementations are the same kind of artifact as stored cases (crisp code behind a guard), so they share
level 3; they are separate ladder entries only because the store keeps no case row for them, so their state lives in
the tier ledger instead.

### The ladder is built from specializer output

`ladderOf(settings, found)` takes `found = specializerOutput(store, definitionKey, interfaceHash)`
(`calls/compilations.ts`): the number of **active** cases in the current compilation for this definition revision and
context interface, and the compilation's `instructions.md` if present. Configuration names only what the specializer
cannot know:

```ts
{ tier0: { model }, tier1: { model, start?, verify?, shadow? },     // model profiles
  specialized?: { enabled?, start?, verify?, shadow? },             // options of tier 2
  crisp?: { enabled?, verify?, implementations?, start? },          // options of tier 3, hand-written extras
  shadowRate?, same? }
```

- Tier 3 (stored cases) appears when the compilation has an active case; a function with only shadow cases, or none,
  has no tier 3 and so no deopt noise. Case states stay in the store (`cases.tier`); the ledger does not duplicate them.
- Tier 2 appears when the compilation carries `instructions.md`. **The specializer does not write that file today**:
  it writes cases, reports and declines. The tier, the guidance hook and the ladder slot exist so that a future
  specializer pass (trace-specialized instructions, few-shot examples chosen from the groups) only has to store one
  more compilation file. Until then nothing produces tier 2 and it is absent from every ladder. It starts in `shadow`.
- Thresholds are not configuration here. They are the store's settings (next section).

The `Tier` interface is `id`, `level`, `canServe(input)`, `serve(input)`, `verify(input, output)`, optional
`shadow(input)`. `input.run(plan)` performs one attempt of the existing dispatch: `TierAttempt { strict, compiled, model,
guidance, cases }`. A strict attempt serves from crisp cases only: with none, the kernel discards the half-opened call row
and throws `Deopt` before any model runs, so tier 3 costs nothing when it declines.

## Guards

A guard has two halves. Before serving: `canServe(input)`, a synchronous predicate over the arguments (a case's `when`,
an implementation's `when`, a model profile's availability). After serving: `verify(input, output)`, true or a reason
(implementation `verify`, the declared return type and `Is<T,...>` refinements the kernel already checks, optional
per-tier checks, shadow comparison for promotion). A guard miss is not a failure: it is recorded as `tier_deopt` with
`reason_kind: guard` and counts as nothing against the tier.

## Deoptimization

On a guard miss, an error, or a failed `verify`, the engine goes to the next lower tier whose state is `active` and
records `tier_deopt { function, tier, reason_kind: guard|verify|error|infrastructure, reason, to }`. A crisp case that
fails after effects hands the call to the model below it with the existing hand-off note (`handoffNote`). Verify
failures after a finished attempt re-run the call on the lower tier, so tiers above 0 should be used where a rerun is
harmless or the attempt had no effects. Failure kinds:

- `guard`: the tier declined; never counted.
- `verify`, `error`: the tier answered wrongly or its own code failed; counted as `handed_off`, the same count the store
  keeps for a case that hands a call to the agent.
- `infrastructure`: a timeout, a refused connection, a rate limit (`failureKind` in `calls/evidence.ts`); bad luck, not
  counted against the tier.

## Evidence and promotion: one rule, one policy seam

Cases and tiers are judged by the same code. `calls/evidence.ts` holds:

- `Evidence`: `served`, `handed_off`, `guard_misses`, `infrastructure`, `compared`/`worse`/`better`, `live_compared`/
  `live_worse`, `audited`/`audit_worse`, `recent_compared`/`recent_worse` (the last 10 comparisons). One meaning each:
  *compared* is any comparison with the reference executor (held-out replay, shadow, audit); *live* is a comparison on a
  call made after the subject existed (shadow, audit; replays of the training calls are weak evidence); *audited* is a
  comparison of a call the subject served; *worse* includes `diverged`; *handed off* is a call the subject started and
  could not finish correctly.
- `EvidenceRule` from the store's settings: `acceptanceBound` (0.05), `promotionComparisons` (10),
  `promotionLiveComparisons` (3); the fixed minimums for demotion are 5 audits and 10 served-plus-handed-off calls.
  These are the only thresholds; the tier engine has none of its own. (Removed: `promoteAfter`, `bound`, `demoteAfter`
  of the first prototype. A tier now needs the same 10 comparisons to be promoted and the same 10 calls to be demoted
  that a case needs.)
- `crispPolicy(summary)`: the rule of TRACE_SPECIALIZATION.md §6.2, unchanged for cases. A `shadow` subject is promoted at
  `comparisons` comparisons with at most `bound` worse and `liveComparisons` live ones within the bound; an `active` one
  is demoted when audits, hand-offs or comparisons exceed the bound. A demoted tier is judged as `shadow` over the
  evidence gathered since its demotion (the counters reset when it is demoted), so a tier can earn its way back; a
  demoted case stays demoted until `natlang compilations enable` or a new compilation, as before.
- Folding: `foldEvent` turns tier events into counters; `TierLedger` keeps them per function and tier and rebuilds them
  from the store's `tier` annotations (`hydrate`), including `promoted` / `demoted` events written by a policy loop in
  another process. Case counters are the `cases` columns (`store.caseSummary`).

### Mechanism and policy

The owner's rule: policy is natural language, the mechanism stays crisp, hot paths are pluggable.

- **Mechanism (crisp)**: collecting and folding evidence, building an `EvidenceSummary`, applying a decision
  (`applyDecision`, `store.decideTier`, `TierLedger.decide`), recording it (`promoted` / `demoted` events on the call
  record; the case's `note` and `promoted_at` / `demoted_at`).
- **Policy**: given a summary, answer `promote | keep | demote` with a reason. Two implementations of the same part:
  `crispPolicy` (today's thresholds) and `applications/specializer/promote.nl`, which spells out for a small model the
  failure kinds (guard misses and infrastructure outages are not failures), sample sizes (do not decide on a handful),
  the share of worse results, live versus replay evidence, and recency (a recent run of worse results outweighs a
  clean history for an active subject). `runtime/promotion.ts` selects between them with `pluggable()` under the store
  setting `promotionPolicy: 'crisp' | 'nl' | 'shadow'` (default `crisp`; `natlang traces config promotionPolicy=nl`).
  `shadow` asks both, applies the crisp decision, and traces agreement as `pluggable_shadow`.
- **Off the hot path**: under `crisp` the rule is arithmetic and runs as evidence arrives, exactly as before (store
  `reviewTier`, ledger `apply`). Under `nl` nothing is decided per call: `store.reviewTier` and the ledger leave states
  alone, and the specializer loop calls `reviewPromotions(store, policy)` once per pass (after the jobs), which judges
  every case of a current compilation and every tier with evidence, and applies the answers. A malformed answer is a
  `keep` (`sanitizeDecision`). Running engines pick up tier decisions within 5 s (`hydrate` reads new
  `promoted`/`demoted` events).

Stored cases (tier 3) are promoted by the store, so the engine reports the first call a case serves as `tier_promoted` for
tier 3, as the first prototype did for tier 2.

## One view

`natlang traces tiers` is the per-function, per-tier view: state, calls, deopts and their rate, tokens and ms per served
call, tokens wasted by attempts that were deoptimized after running, and the evidence columns (`compared`, `worse`,
`live`, `audited`) from the shared fold. The tier-3 row is not a second accounting: its calls, state, evidence, ms per
call and `net_tokens` (saved minus spent) are the compilations' own (`CallStore.savings`, `caseSummary`), summed over the
current compilation's cases. `natlang compilations savings` ends with a pointer to the tiers view; `natlang traces
status` shows the same settings. `natlang compilations list|show` still show each case with its own numbers.

Trace events: `tier_served`, `tier_deopt`, `tier_promoted`, `tier_demoted`, `tier_shadow` (a synthetic `tiers <fn>`
trace per call). Every event is stored as a `tier` annotation (with the tier's state at the time, so a store can be
folded without the configuration); no schema change.

## Limits of the prototype

- Tier 3 needs a call store (it is served through its records). Without one the ladder is tiers 0 and 1.
- Directory reducers and calls with a caller-supplied folder bypass the engine.
- Shadow of model tiers duplicates effects; enable only for pure functions.
- The specializer writes no tier 2 artifact yet; hand-registered implementations are not synthesized.
- Ladder order is by level, not cost.

## Next steps

1. Tier 2 synthesis: have `natlang-specializer` store `instructions.md` per compilation (instructions sharpened by the
   groups, a few verified examples), verify it by replay against the teacher like cases, and let `promote.nl` judge it.
2. Persist the tier config next to `natlang.json` and load engines in the CLI.
3. Decision-readout confidence as the tier 1 guard (E1's table), and cost-aware ordering from `traces tiers`.
4. Tier 4: a `Tier` over an adapter or soft program, guard = held-out agreement; blocked on the learning continuum.
