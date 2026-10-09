# Tiered execution engine

Prototype of ARCHITECTURE_IMPROVEMENT.md §E1. One natural-language function runs at a ladder of tiers, chosen per call,
with guards and deoptimization. Code: `ts-host/src/calls/tiers.ts` (engine, tiers, ledger), a 40-line hook in
`runtime/kernel.ts` (`tieredCall`, `InvokeOptions.tierAttempt`), `tierRows` in the store, `natlang traces tiers`.
Opt-in: `createNatlangRuntime({ tiers: new TierEngine({ functions: { name: settings } }) })`. Without it, or for a
function it does not name, nothing changes.

## Tiers and the code behind them

| Tier | Class | Served by | Guard before | Check after |
| --- | --- | --- | --- | --- |
| 0 | `ModelTier(0)` | the function's own model, or `models[tier0.model]`; the kernel's normal agent path | none (terminal) | declared type and refinements (already in the kernel); its failure is recorded, not retried |
| 1 | `ModelTier(1)` | `models[tier1.model]` (the `model: NAME` frontmatter mechanism, set per attempt) | profile exists | kernel type/refinement checks, optional `tier1.verify(args, out)`; shadow comparison to promote |
| 2 | `CompiledTier` | the stored compilation's cases (`calls/compilations.ts`, `admit`, `runCrispCase`) | the case's own `when` plus its case tier (`active`) | refinement check on the case result (kernel), optional `tier2.verify` |
| 3 | `CrispTier` | a `CrispImplementation` served as a synthetic case (hash `tier:<id>`) through `runCrispCase` | `when(args)` input-shape predicate | `verify(args, out)` before serving, plus shadow replay against the tier below |
| 4 | `NeuraleseTier` | stub: `canServe` false, `serve` throws | | |

Note: today's compilation cases are crisp TypeScript with guards. Per the E1 table they occupy tier 2 (the stored,
store-promoted artifact); tier 3 is the same machinery for code with an input-shape guard plus shadow replay, which the
specializer would later write. The ladder reuses `runCrispCase` for both, so type check, refinements, records, effect
tracking and hand-off notes are identical.

The `Tier` interface is `id`, `level`, `canServe(input)`, `serve(input)`, `verify(input, output)`, optional
`shadow(input)`. `input.run(plan)` performs one attempt of the existing dispatch: `TierAttempt { strict, compiled, model,
cases }`. A strict attempt serves from crisp cases only: with none, the kernel discards the half-opened call row and
throws `Deopt` before any model runs, so tiers 2 and 3 cost nothing when they decline.

## Guards

A guard has two halves. Before serving: `canServe(input)`, a synchronous predicate over the arguments (tier 3 `when`, tier 2
case `when`, tier 1 profile availability). After serving: `verify(input, output)`, true or a reason (tier 3 `verify`, the
declared return type and `Is<T,...>` refinements the kernel already checks, an optional per-tier check, and shadow
comparison for promotion). A guard miss is not a failure: it is recorded as `tier_deopt` with `reason_kind: guard`
and never counts toward demotion.

## Deoptimization

On a guard miss, an error, or a failed `verify`, the engine goes to the next lower tier whose state is `active` and
records `tier_deopt { function, tier, reason_kind: guard|verify|error, reason, to }`. A crisp case that fails after
effects hands the call to the model below it with the existing hand-off note (`handoffNote`). Verify failures after a
finished attempt re-run the call on the lower tier, so tiers above 0 should be used where a rerun is harmless or
the attempt had no effects (tier 3 verifies before the case returns; model tiers are restricted to functions without
folder arguments). Failures other than guard misses count as consecutive failures; `demoteAfter` (default 3) of them
demote the tier (`tier_demoted`), which then is skipped until shadow evidence re-promotes it. A success resets the count.

## Promotion

Tier state is `shadow | active | demoted` per function and tier (`TierLedger`). Tier 3 starts in `shadow` (tier 1 in
`active` unless `start: 'shadow'`). A call served below a shadow or demoted tier is, with probability `shadowRate`, also
run through that tier's `shadow(input)` (crisp implementations marked `pure`, model tiers with `shadow: true`) and compared with
`same` (default: canonical value equality) giving `shadow_equal` / `shadow_worse`. After `promoteAfter` (10) comparisons
with at most `bound` (5%) worse the tier is `promoted` and serves. This is the same rule as `CallStore.reviewTier` for
compilation cases; tier 2 is not duplicated: its cases are promoted by the store's `promotionComparisons`,
`acceptanceBound` and live-evidence rules, and the engine reports the first call a case serves as `tier_promoted` for tier 2.

## Evidence and measurement

Every event is stored as a `tier` annotation on a call record (`CallStore.tierRows()` joins the call's tokens and wall
time). The ledger is a fold over these events, so state survives restarts (`hydrate`). Per tier, `natlang traces tiers`
reports calls served, deopts, deopt rate, tokens and ms per served call (from `calls.tokens_in/out`, `wall_ms`, the same
columns `CallStore.savings` uses) and tokens wasted by attempts that were deoptimized after running. Savings of tier ≥2
versus tier 0/1 comes from the per-tier cost per call. Trace events: `tier_served`, `tier_deopt`, `tier_promoted`,
`tier_demoted`, `tier_shadow` (a synthetic `tiers <fn>` trace per call).

## Limits of the prototype

- Tiers 2 and 3 need a call store (they are served through its records). Without one the ladder is tiers 0 and 1.
- Directory reducers and calls with a caller-supplied folder bypass the engine.
- Shadow of model tiers duplicates effects; enable only for pure functions.
- No synthesis (tier 3 implementations are hand-written), no tier 4, no cost-aware ordering (ladder order is by level).

## Next steps

1. Tier 3 synthesis: have `natlang-specializer` emit `CrispImplementation` modules (input-shape guard from `induceRules`,
   body from the approach template) and register them as `tier3` for a function; shadow replay then uses recorded calls
   (`ReplayServices`) so effectful functions can be shadowed offline.
2. Persist the config next to `natlang.json` and load engines in the CLI; read `specialization` thresholds from store settings.
3. Decision-readout confidence as the tier 1 guard (E1's table), and cost-aware ordering from `traces tiers`.
4. Tier 4: a `Tier` over an adapter or soft program, guard = held-out agreement; blocked on the learning continuum.
