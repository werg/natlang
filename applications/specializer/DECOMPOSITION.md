# Specializer: decomposition, part by part

Status: draft for owner review (plans/OWNER_REVIEW.md). Nothing is restructured until the owner has reviewed it.
`README.md` in this folder already lists the parts in one table; this document extends it unit by unit with
line references, the policy that sits in `main.ts`, and the natural-language functions' steps.

The specializer compiles recorded model calls into guarded crisp cases (plans/TRACE_SPECIALIZATION.md). The crisp
half is mostly in the runtime package (`ts-host/src/calls/`: `study` at `specializer.ts:22`, `crispDecline` at
`specializer.ts:60`, `groupsOf` at `groups.ts:21`, `measure` at `groups.ts:42`, `renderGroup` at `groups.ts:69`,
`assembleCases` at `groups.ts:114`, `verifyCases` at `specializer.ts:207`, `saveAccepted` at `specializer.ts:299`,
`runJob` at `offline.ts:88`, `detectFindings` at `findings.ts:14`). These are exact analyses and verifiers and are
correct crisp (they are reviewed with the calls package, not here). `main.ts` (292 lines) holds the orchestration and
four policies in TypeScript: which functions are worth looking at, when to wait, how many rounds, and what to say
about a decline.

Decisions: **fn**, **inline**, **implicit**, **crisp**, **service**, **host**, **pluggable**.

## Policy

- **Natural language: judgments about code and meaning.** Whether a condition decides by structure or by meaning
  (`semanticCheck`), whether two programs do the same work (`sameApproach`), the writing of a case from a group of
  calls (`writeCase`), and, as pluggables, whether a function is worth looking at now (`worthLooking`) and the
  summary of a decline (`summarizeDecline`).
- **Crisp: measurement, replay, equality, storage.** Rule induction with exact precision, anti-unification,
  `group.measure`, case assembly, replay on recorded calls, the runtime judge's blinded comparison, saving, the
  call store. A model cannot be asked for these.
- **Process control stays with the owning session.** The specializer reads the executor load (`waitForIdle`) and
  waits for it; it never stops or starts other processes. The heartbeat program (plans/HEARTBEAT_PROGRAM.md) is the
  place where unit-level decisions are advised.
- **Interruptible and resumable.** Every step reads from and writes to the call store; `SIGTERM` finishes the
  current step (`main.ts:259-260`). This stays.
- **State model.** One pass over a definition is decide-then-commit: study (snapshot), writers decide per group,
  verification (crisp) accepts, `saveAccepted` commits atomically. Derived values form a DAG: calls, groups,
  conditions, cases, checks, compilation. The rounds are a bounded repair loop over groups whose measure is "groups
  not done" (see question 1).

## Parts

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| CLI options and defaults (rounds 3, jobs 50, interval 600 s, maxBusy 2, idleWait 1800 s) | host | `parse`, `main.ts:21-28` | CLI. Each default becomes a named setting with its reason beside it. |
| Which definitions are worth a look: enough agent calls, tokens spent, no standing decline unless volume doubled, enough new calls since the last compilation | pluggable | `targets`, `main.ts:31-47` | A policy over counts. Crisp default is the current rule; natural language `worthLooking` weighs estimated savings against the specializing spend. |
| `minCalls`, doubling rule (`2 * decline.calls_at_decline`) | data | `main.ts:37`, `41` | Parameters of the crisp default. |
| Writer budget: 150 actions, 2 h stall bound, 2 writers at once | host | `WRITER_LIMITS`, `WRITERS`, `main.ts:53-55` | Resource limits with a stated reason (a shared executor can stall); they stay and are listed with their reasons. |
| Did a writer stop on its budget | crisp | `exhausted`, `main.ts:64-68` | Error classification from a typed `outcome` first, message text second. |
| Token meter around the model driver | crisp | `metered`, `main.ts:73-89` | Accounting plumbing. |
| Judge and writer runtimes | host | `main.ts:92-94`, `126-130` | Mechanism. |
| Wait for an idle executor: vLLM running plus waiting requests at most `maxBusy`, give up after `idleWait` | crisp | `waitForIdle`, `main.ts:101-123` | Reads a metric and compares; thresholds are settings. |
| Study a definition's recorded calls | crisp | `study`, `specializer.ts:22` | Exact. |
| Decline without a model when nothing repeats | crisp | `crispDecline`, `specializer.ts:60`; `main.ts:177-178` | Costs no call. |
| Findings from recorded calls | crisp | `detectFindings`, `findings.ts:14`; `main.ts:179` | Exact. |
| Group calls by what they did (at least 3 training calls, at most 8 groups) | crisp, with `sameApproach` | `groupsOf`, `groups.ts:21`; `main.ts:180` | Normalization and anti-unification are exact; sameness of differently written programs is a judgment. The 8-group cap is a named setting. |
| Do two normalized programs do the same work | fn, decision | `writeCase/sameApproach` | A judgment. |
| Write a case for one group, or skip with a reason | fn (directory reducer) | `writeCase` | Authoring. Today one call does condition search and body writing; see the split below. |
| Choose a condition and measure it | fn | `writeCase/chooseCondition` (proposed) | Steps 1-3 of `writeCase.nl:22-24`. |
| Is a condition structural or semantic | fn, decision | `writeCase/semanticCheck` | Its own question for small models. |
| Write `case.ts` for a chosen condition | fn | `writeCase/writeBody` (proposed) | Step 4 of `writeCase.nl:25`. |
| Assemble cases, check each loads | crisp | `assembleCases`, `groups.ts:114`; `main.ts:200-215` | Exact. |
| Replay, compare with the agent, judge where they differ, accept within a bound | crisp with the runtime judge | `verifyCases`, `specializer.ts:207`; `main.ts:206` | Exact replay and equality; the judge is blinded to which side is the case. |
| Rounds: a rejected case returns to its group with the report, up to `rounds` attempts | host | `main.ts:186-226` | Bounded repair loop. |
| "Better" finding | crisp | `betterFinding`, `findings.ts:51`; `main.ts:220-221` | Exact. |
| Final re-verification of accepted cases, then store as shadow | crisp | `main.ts:235-243`; `saveAccepted` | Exact durability. |
| Decline reason: most frequent group reason | crisp | `main.ts:244-245` | Exact mode of a list. |
| Decline summary text | pluggable | `summarizeDecline` (crisp default: the joined group lines, `main.ts:229`, `246`) | A reading for a person. |
| Unstable-results findings | crisp | `main.ts:230-233` | Exact. |
| Shadow replays and audits, up to `jobs` per cycle | host | `main.ts:268-287`, `runJob` | Mechanism over the store. |
| Loop interval | host | `main.ts:288-289` | Scheduling. |

## Natural-language functions, step by step

### `worthLooking` (natural-language side of the pluggable)

```
args: candidate: { name, agent_calls, tokens, new_calls_since_compilation, decline: { calls_at_decline, reason } | null,
                   has_compilation: boolean }, min_calls: number
returns: { look: boolean, reason: Is<string, "one sentence naming the deciding count"> }
```

1. When `agent_calls` is below `min_calls`, answer look false (reason: too few calls).
2. When `tokens` is 0, answer false (nothing to save).
3. When `decline` is present, answer true only when `agent_calls` is at least twice `decline.calls_at_decline`.
4. When `has_compilation` is true, answer true only when `new_calls_since_compilation` is at least `min_calls`.
5. Otherwise answer true.

(This is the crisp rule spelled in steps. The natural-language side may weigh a function whose recent decline
reason was "unstable" differently; the first version keeps the crisp rule and exists so shadow runs can show
disagreement.)

### `summarizeDecline` (natural-language side of the pluggable)

```
args: groups: { id, label, calls, outcome, reason, why }[]
returns: Is<string, "two sentences: the main obstacle and what would change the outcome">
```

1. Count the groups by `reason`; name the most frequent obstacle in plain words.
2. For that obstacle, say what input or instruction change would remove it (for example, anchor the condition on the
   whole input form).

### `writeCase/chooseCondition` (proposed split)

```
args: definition, group  (folder: function.md, group.md, examples/, others.md, report.md)
returns: { kind: "condition", condition: string, admits: number } | { kind: "skip", reason, why }
```

1. Read `group.md`, then `function.md`; open an example when you need details.
2. Pick a condition from `group.md` that admits this group's calls and none of the calls in `others.md`.
3. Call `group.measure(condition)`; read `ofGroup`, `others` and the counterexamples. Adjust until `others` is 0.
4. Call `semanticCheck(instructions, condition, examples)`. When it answers "semantic" for a word inside free text,
   try an anchored form (for example `/^refund (\d+)$/i`) and measure again.
5. Return the condition and `admits`. When no condition separates the group, return a skip with reason `semantic`,
   `unstable`, `effects` or `no-condition` and a one-sentence `why`.

### `writeCase/writeBody` (proposed split)

```
args: definition, group, condition   (the same folder; writes case.ts)
returns: { kind: "case", admits: number } | { kind: "skip", reason: "effects" | "unstable", why }
```

1. Write `export const when = (args) => <condition>` with the chosen condition.
2. Write `run`: make the same service calls with the same arguments as the executor made for these calls (see the
   examples), and return the same result.
3. Remove a service call that the results show to be wasted, and make inconsistent results consistent, when the
   instructions allow it.
4. When an input does not fit after all, `run` throws `new Error(reason)` and the function takes over.

`semanticCheck` and `sameApproach` keep their current contracts and wording (`semanticCheck.nl`, `sameApproach.nl`):
each is already one question with its criteria stated positively.

## Pluggable points

| Point | Setting | Crisp default | Natural language | Verifier |
| --- | --- | --- | --- | --- |
| Targets | `worthLookingMode` | `targets` filter | `worthLooking` per candidate | boolean; the crisp bound `agent_calls >= minCalls` stays |
| Decline text | `declineSummaryMode` | joined group lines | `summarizeDecline` | stored text only; no decision depends on it |

Neither is hot (`targets` once per cycle). Both are for shadow comparison and so the owner can change the rule in
words. The settings use the shared vocabulary `crisp | nl | shadow`.

## Refinement candidates

| Slot | Proposed type | Check |
| --- | --- | --- |
| `CaseResult.admits` | `Is<number, "equals the ofGroup value group.measure returned for the written condition">` | crisp |
| `CaseResult.skip.reason` | the closed union (already) | crisp |
| `CaseResult.skip.why` | `Is<string, "one sentence naming what prevents a case">` | judged |
| `case.ts` `when` | `Is<string, "a side-effect-free test over args that returns true or false immediately">` | crisp (parse and dry call) |
| `chooseCondition.condition` | `Is<string, "a JavaScript expression over args that group.measure accepts as valid">` | crisp (`measure(...).valid`) |
| `ConditionKind`, `Sameness` | closed unions (already) | crisp |
| `semanticCheck.examples`, `sameApproach` programs | `Untrusted<string>` | crisp marking (recorded inputs are data) |
| `summarizeDecline` result | `Is<string, "two sentences: the main obstacle and what would change the outcome">` | judged |
| `worthLooking.look` | `Is<boolean, "false whenever agent_calls < min_calls">` | crisp |

## Model-facing changes needing live measurement

The directly measurable quantity is the verification acceptance rate per group (`verification.checks[].accepted`,
`main.ts:222`), available from the call store.

1. **Split `writeCase`** into `chooseCondition` and `writeBody`. Compare accepted-case rate, rounds used and
   tokens per accepted case.
2. **`writeCase.nl:16`**: "Your condition must not admit them" becomes "Your condition admits none of them."
   Measure with the split (same samples).
3. **Report text** returned to a group (`main.ts:196`, `204`, `212`, `renderReport`): wording is model-facing; keep
   byte-identical in the first move.
4. **`Untrusted<string>`** for recorded inputs in examples (rendered into `examples/*.json`, `group.md`, `others.md`).
5. **`worthLooking` and `summarizeDecline`** are new text; they do not reach the writer, so they need agreement
   measurement only (shadow mode), not writer-quality sampling.

## Questions for the owner

1. The repair loop stops at `rounds` attempts per group (default 3, `main.ts:186-187`). Termination could be
   structural instead: iterate while some group is not done and the last round improved a group's report (a
   progress predicate). Keep `rounds` as a named resource limit, or move to the progress rule?
2. The 8-group cap and 3-call minimum (`main.ts:180`, `groups.ts:21`) bound spend. Keep as settings with reasons?
3. Should `semanticCheck` run as `shadow` between two prompt variants first, since it gates which conditions become
   code? The runtime judge audits cases later, so this is an efficiency question, not a safety one.
