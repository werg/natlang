# First execution batch (2026-11-23)

Independent items: none depends on another in this batch, except where a brief says it can start before its
dependency and how. Ordered by expected value. Each brief is self-contained. Paste the **common preamble** above it
when dispatching.

## Common preamble (paste at the top of every brief)

> You work in `/home/werg/natlang` on main, as one of several agents working in parallel. Read `AGENTS.md` first; its
> product rules bind you. **Natlang-native:** a component that decides is a natlang program (typed `.nl` stages plus
> crisp helpers, `pluggable({ crisp, nl })` where hot). Policy is natural language, mechanism is crisp. Write
> DECOMPOSITION.md first for any new program. **Teach in errors, not prompts:** never spell out a wrong form in model-
> or user-facing text. Teach the fix in a one-sentence error, add it to `spec/ERRORS.md` and to the authoring skill's
> error table. **Fix the mechanism everywhere it lives:** before editing, write the inventory of every component that
> implements or consumes the mechanism. That means all four engines (managed llama-server fork, in-page wllama,
> Python native engine, SGLang actor), ts-host Node and browser, compiler, learner, held eval, collectors and
> renderers of recorded prompts. A component you leave out needs a code citation showing it does not use the
> mechanism. **No premature limits; one implementation per concept.** Model-facing changes ship as the default on
> first principles, with an opt-out and targeted tests, and an entry in `plans/MODEL_FACING_CHANGES.md`. The system
> message and tool block must stay a byte-identical prefix across calls. Anything that loads a model or runs tests
> goes through `scripts/memory_ledger.py run` (see AGENTS.md "Machines"). Commit only your own paths with
> `git commit -m ... -- PATHS`. Run `scripts/check-main.sh --quick` under the ledger, and push only after it exits 0.
> `git pull --no-rebase` before committing. Never stash, reset, amend or rebase. Record each decision in
> `plans/neuralese/DECISIONS.md`. The campaign plan is `plans/semantic-fp/` (README and the WS document your brief
> names). Report what you shipped, what you verified, what you did not check, and any fact in the plan you found
> wrong.

---

### B1. held-v4: code, reservations, and the first set (WS5 H1; 4 agent-days)
**Why first:** the campaign measures what it wants before it trains on it. held-v3 stays the gate series alongside.
**Inventory:** `training/neuralese/natlang_neuralese/common/data_registry.py` (`role_path` L772 requires exactly one
entry per role; `role_paths` L767); `scripts/neuralese_data/grpo.py` (`default_held()` L137, `load_tasks` L3759,
`--exclude-manifest` L5422); `training/neuralese/natlang_neuralese/rl/held_eval.py` (`exclusion_ids()` L274 follows
`also_excludes`; `pin` L1215 takes one `--cases`; `verify` L1234); `scripts/neuralese_data/expert_iteration.py`
(`select` hardcodes `exclude_manifest=[]` at L156; `dgx-held` L2968); `ops/held-v3/exposure/collect_exposure.py` under
`/home/werg/data/lfm3-rl-dgx-20261029/` (hand-written `POOLS` list); every other reader of the `held-eval` role
(`git grep -n "held-eval\|default_held\|exclusion_ids"`).
**Steps:**
1. Role `held-eval` stays held-v3. Add role `held-eval-semantic-fp` for held-v4. Training exclusion is the union over
   every manifest with a role starting `held-eval`. Apply it in `grpo.load_tasks`, `expert_iteration.select` and
   every collector or builder path the inventory finds. `collect_exposure.py` reads pools from the registry.
2. **Reservations now** (`ops/held-v4/reservations.json`, registered). 20 design briefs chosen by seed
   `held-v4-design-briefs` (their `split_group` `design-brief-app-NNN`, from
   `training/semantic_applications/design_briefs/`); 2 of the 11 Astra/Sol apps (seed `held-v4-astra`); a 20% split
   group share of every flagship fixture pack as it appears. They are excluded from training from this commit on.
3. **Set.** 256 tasks stratified over the construct families of `plans/semantic-fp/05_TRAINING_DATA_AND_ORACLES.md`
   §3: inline-lambda pipeline, library discovery, convergence with an NL stop, higher-order policy and `.in`, folder
   transaction, untrusted contract, decision calibration, Neuralese pipeline, plus program execution over the
   reserved briefs' 160 cases. Author them with the existing kit (`plans/neuralese/RL_TASK_AUTHORING_KIT.md`,
   `scripts/rl_tasks/sources/authored.py`, `qualify_authored.py`, `verify.mjs`), exact or normalized oracles. The
   construct must be *required* for the reference solution, so set the kit's construct flags (e.g. `inline:
   required`, not `optional` as `authored.py:157` does). Verifier-oracle tasks are added when B6 lands, as a
   versioned update before any construct family is admitted to training. Pin with `held_eval pin --also-exclude
   ops/held-v3/manifest.json`. `held_eval verify` against every pool.
4. Measure the base (5ff07638) and `ops/current-policy.json` on the DGX series (`expert_iteration.py dgx-held`
   machinery, 4 samples, medium priority, preemptible). It must not delay any SFT round.
**Tests:** `tests/neuralese/test_held_eval.py`, `test_expert_iteration.py`: two held roles, union exclusion,
`default_held()` still returns held-v3, a held-v4 sibling never selected.
**Done:** both roles are registered; training code excludes both; the set is pinned and verified; base and current-policy
results are in `ops/held-v4/runs/`; DECISIONS entry with the numbers.

### B2. The model-facing prompt: cache map, stable prefix, sentence pass (WS2 R0 + R1; 3 agent-days)
**Inventory:** `ts-host/src/native/prompt.ts` (all exports), `native/agent.ts:115-127` (`callNotes`, opening
layout) and `:1276` (`edit_code`/`diff_code` added only when `session.codeEditable()`), `native/system-prompts.ts`
(piece registry and `Neuralese<SystemPrompt>` forms), `native/runtime.ts:270-325` (`read_code` builtin docs),
`:2329-2336` (`prompt-example` rejection), `runtime/kernel.ts:526-545` (`SAME_QUESTION_REFUSAL`, depth limit),
`teacher/cli.ts`, `teacher/serving-request.ts`, `tests/neuralese/test_serving_prompt_rendering.py`,
`scripts/neuralese_data/sft_sources.py` and `filled_reasoning.py` (how they render the system message of an own
success: recorded or current?). Engines receive the ts-host rendering; prefix reuse per engine is in
`docs/ENGINE_PARITY.md`, and the learner's prefix tree is in `train/lfm3_fast.py`.
**Steps:**
1. **R0.** From real call records (a DGX held run under `ops/held-v3/runs/`, an async round's collection, an
   `examples/triage` run, a VibeStudio capture), compute in LFM3 tokens the shared-prefix structure. For each call:
   the longest prefix shared with earlier calls in its episode and window, and where the first differing byte falls.
   Write `ops/semantic-fp/prompt-cache/2026-11-23.json`.
2. Fix every break of the stable prefix the data shows. First suspect: the conditional `edit_code`/`diff_code` tool
   entries, against MODEL_FACING_CHANGES 2026-11-12 "one constant opening for every call". Add
   `ts-host/test/prompt-prefix-stability.test.mjs`: system message and tool block byte-identical across call shapes
   (plain, own code, folder reducer, depth limit, Neuralese signature, decision readout, different return types).
3. If SFT renders own successes under the *current* rather than the recorded system message, fix it to the recorded
   one, with a test. That is history differing from what was sampled.
4. **R1.** Classify every sentence of `SEMANTIC_RESULT_PROMPT`, `SEMANTIC_CALL_COMPLETION_PROMPT`, `TOOLS_PROMPT`,
   `GENERATION_GUIDANCE`, `FUNCTION_TOOLS_PROMPT`, `directoryReducerPrompt` into `plans/semantic-fp/prompt-audit-20261123.md`:
   wrong or contradicting / missing built-in / task-specific scar tissue / wrong form spelled out /
   anti-delegation / copyable domain example / duplicate / capability text for some calls / correct. Act per
   `02_RUNTIME_PROMPT_AND_COMPILATION.md` §2.3. Known instances are listed there. Add the `decide` sentence to the
   stable core. Replace the refund example with a schematic one, then delete `PROMPT_EXAMPLE_TEXTS` and its
   rejection. Replace `SEMANTIC_CALL_COMPLETION_PROMPT` with one positive sentence (the runtime already refuses the
   same question). Move text out of the stable core only where R0's numbers or a teaching argument justify it.
   Per-call text goes only after the call's instructions.
5. Opt-out `NATLANG_PROMPT_REVISION=2026-11-22` restores the old text byte for byte. Re-initialise the soft forms of
   changed piece IDs. One MODEL_FACING_CHANGES entry per class of change, with the sentences.
**Done:** prefix test green; R0 re-run shows no shorter shared prefix on the same records; new text is the default on
main; the opt-out is tested.

### B3. `natlang init` writes a semantic-FP program, and its AGENTS.md (WS4 F1; 2 agent-days)
**Inventory:** `ts-host/src/package/authoring.ts` (`createProgram`, `answerSource`, `programReadme`, `agentGuide`,
`createFunction`); `ts-host/src/cli/commands.ts:984-994` and `cli/main.ts:172` (init takes no flags today);
`ts-host/src/dashboard/server.ts:18,429` (Playground calls `answerSource(custom)` and `createFunction`); tests that
snapshot init output (`git grep -n "createProgram\|agentGuide\|answer.nl" ts-host/test`); `README.md` "Your first
program"; `skills/natlang-authoring/SKILL.md` working-loop table (describes init's files); `docs/RUNNING_PROGRAMS.md`
"First result and first edit".
**Steps:** implement exactly `plans/semantic-fp/04_DOCS_EXAMPLES_AND_AUTHORING.md` §2-§3: the `types.ts`, `route.nl`
(`readout: decision`), `main.ts` (inline `nl<Reply>`, `Untrusted`, `Is`), `examples.json`, `messages.json`,
`natlang.json`, README and the generated AGENTS.md text. Keep today's program as `--template classifier`. Verify every
API and command the text names. Settle the host-code form of `decide`. If none exists, add one (reusing
`runtime.decide`, `runtime/runtime.ts:583-592`), and drop or rewrite any line that does not hold today. Update the
`createFunction` stub text. Update the three docs in the inventory to the new files.
**Tests:** init output snapshot; `natlang check` clean on the new program; `natlang test` passes on local LFM3
through the ledger; first `natlang run` under ~60 s on DGX CPU after setup (if not, change the sample to the `fyi`
path as §2 says).
**Done:** shipped as the default; the classifier template works; the docs match.

### B4. Authoring skill and README carry the paradigm (WS4 F2 + F4; 3 agent-days; may start with B3)
**Inventory:** `skills/natlang-authoring/SKILL.md` and `references/{language,patterns,refinements,verification}.md`,
`assets/review/`; `skills/README.md`; `README.md`; `ts-host/src/cli/commands.ts:765-800` (`natlang skills install`
copies the folders); any test reading skill text (`git grep -n "natlang-authoring" ts-host/test tests`).
**Steps:** apply WS4 §4 exactly: the new opening; "Design the program" as the eight items; delete the three sentences
listed; new `references/composition.md`, each section one runnable example taken from `examples/` (no wrong form
anywhere); rewrite `references/patterns.md` around the curated programs (`examples/triage`, `examples/review`,
`applications/notebook`, `applications/helpdesk`, `applications/nldb`) and drop its Astra-derived paragraphs. Apply WS4
§5 to README: the first-screen text, the triage snippet with `readout: decision`, a confidence gate with `fn.on`, an
`Is` summary type, and "Your first program" on B3's template (coordinate wording with B3 if it is still in flight).
Verify every API and command against the code. A construct not yet built is not mentioned.
**Done:** shipped; `natlang skills install` installs the new text; links resolve.

### B5. The decision objective in expert-iteration SFT (WS5 O3; 2 agent-days)
**Inventory:** `training/neuralese/natlang_neuralese/train/decision_supervision.py`, `common/decision_objective.py`,
`data/decision_graph.py`, `train/native_graph_loss.py`, `own_output_supervision.py` (today's only users);
`exp0_sft.py` (the SFT trainer; LoRA core `train/lfm3_learner.py`, `train/lfm3_fast.py`);
`scripts/neuralese_data/sft_sources.py` (own decision-readout replies kept as plain rows, empty think masked
`own_empty_think` L599-610; teacher replays decline scoring L903); `lower_sft_sources.py:58` (skips decision rows);
`rl/decision_policy.py` (RL side); the ts-host scorer `model/decision-scoring.ts` (what a decision call's options are).
**Steps:** carry each decision row's option set and verified label (own successes) through `sft_sources build` into
`exp0_sft`. Train them with the existing option-normalized cross-entropy over the options' sequence log-probs. Make it
the default, with `--no-decision-objective` as the opt-out. Teacher option distributions come only where the teacher
endpoint returns logprobs; otherwise use the label. Check that the option rendering matches the scorer's byte for
byte.
**Tests:** unit test that the loss equals the scorer's softmax CE on a fixture; build test that decision rows carry
options; `lower_sft_sources` no longer drops them.
**Done:** the next async round's build reports decision rows trained with the objective; DECISIONS entry.

### B6. Executable-verifier oracle for authored tasks (WS5 O1; 2 agent-days)
**Inventory:** `ts-host/src/teacher/trusted-exact-verifier.ts` (allow-list: `.mjs` under
`training/semantic_applications/`, sha256, `verifyCandidate` export); `scripts/rl_tasks/sources/authored.py`
(`ORACLES = {'exact','normalized'}` L42); `authored_checks.py`; `qualify_authored.py`; `verify.mjs` (reference replay,
negative controls, alias probe); `ts-host/src/evaluation/oracles.ts` (`OracleLevel`); `rl/task_rollouts.py` and
`--reward graded` (file-effect credit); the held scorer (`held_eval`), so held-v4 can use it.
**Steps:** authored tasks may declare `oracle: verifier` with a `trusted_exact_verifier`; add
`training/semantic_fp_tasks/` to the allow-list. The verifier returns `{ pass, score }`, and graded reward uses
`score`. Qualification requires the reference to pass and ≥ 2 negative solutions plus a keyword-baseline solution to
fail. Held scoring stays binary on `pass`.
**Tests:** `tests/test_authored_checks.py` (accept, reject a non-discriminating verifier, keyword baseline passing →
reject); a ts-host verifier test.
**Done:** a 10-task sample batch passes qualification end to end and is registered (not promoted).

### B7. Equivalence service and baselines for the legacy corpora (WS7 M1; 2 agent-days)
**Inventory:** `training/semantic_applications/design_briefs/<NNN>/cases.json` (800 cases), `applications/design-briefs-host/`
(`runtime.mjs`), `ts-host/scripts/design-briefs/lib.mjs`, `ts-host/test/design-brief-*.test.mjs`;
`training/semantic_applications/{astra,sol}/<app>/project.json` (fixture, oracle, `verify-result.mjs`,
control scripts) and `ts-host/test/trusted-exact-verifier.test.mjs`; the collector endpoint conventions
(`pop_rl_actor.remote_endpoint`, DGX collection actor).
**Steps:** `scripts/semantic_fp/equivalence.py`: run a program (path) on its corpus cases against an endpoint at k
samples. Report per case: pass, calls, prompt+completion tokens, wall time. Compare two program versions on the same
seeds. Run the 100 briefs and 11 apps as they are now on LFM3 (current policy, k=4), at low priority, on a
collection actor that is already up. Never start one that displaces training. Write
`ops/semantic-fp/migration/baseline.jsonl`.
**Done:** baselines for all 111 programs; a summary of how many cases each passes; DECISIONS entry.

### B8. Malleable code through a transaction (WS1 L3; 3 agent-days)
**Inventory:** `native/agent.ts:1276`; `native/runtime.ts:1438-1445` (`codeEditsDenied`, `codeEditable`),
`:1990` (stale comment), `:2089-2114` (edit path, revision record); `adaptation/program-view.ts` (`ProgramView`
revisions, run-scoped); `runtime/loader.ts:150` (`withhold:`); `runtime/runtime.ts:109` (`codeEdits`);
`improvement/adoption.ts` (adopt/rollback); call store (`calls/store-core.ts`) for the call record;
`ts-host/src/cli/improvement.ts`; dashboard trace page; `FUNCTION_TOOLS_PROMPT`; conformance
`conformance/programs/17-reopen-refine.yaml`. Coordinate with B2 on the tool block: `edit_code` must not break the
stable prefix.
**Steps:** per `01_COMPILER_AND_TYPE_SYSTEM.md` L3: protected items (frontmatter `protect:`, `types.ts`, services,
items named by `ensures:` checks once L7 lands) are rejected with the one-sentence error in WS4 §8. Successful calls
record `code_revisions` in the call record. `natlang improve adopt-call CALL_ID [--into DIR]` builds a Folder
proposal, runs `natlang check` and `natlang test`, accepts or reports, and reuses `adoption.ts`. Fix the stale comment.
The default stays allow.
**Tests:** `ts-host/test/code-edits.test.mjs` (protected rejection, revision record, adopt accept/reject, rollback
byte-identical); conformance 17 green.

### B9. Higher-order semantic functions with checked signatures (WS1 L4; 3 agent-days)
**Inventory:** `compiler/inline.ts:370-389` (`nl-not-called` cases), `native/values.ts:200-202` (lambda check),
`runtime/callable.ts`, `compiler/intrinsics.ts`, `runtime/surface.ts` (callable methods, `.in`, `.on`,
`.iterateOn`), `native/types.ts` (`fitsType` variance), `native/agent.ts` call notes (lambda `returns:`), browser
build.
**Steps:** per L4: allow uncalled `nl` in return position, function-typed properties and callable parameters;
`NATLANG_SIGNATURE` tags with `fitsType` checks, and a boundary proxy for untagged functions; `fn.partial`; a call note
for lambda results. Skill: add an example to `references/composition.md` (coordinate with B4).
**Tests:** `ts-host/test/higher-order.test.mjs` (Node and browser).

### B10. Structured `readout: decision` on all four engines (WS2 R3; 4 agent-days; two agents: client+llama/wllama, Python+SGLang)
**Inventory:** `native/decision.ts:19-37`, `runtime/loader.ts:278-299`, `model/decision-scoring.ts`,
`runtime/runtime.ts:583-592` (`decide` shape); Python `serve/native_learning.py` (`decision_items`,
`_score_rendered`, `SCORE_ITEM_FIELDS`), SGLang `serve/sglang_actor.py`; guards
`tests/neuralese/test_engine_parity.py`, `scripts/wllama_smoke.mjs`; `docs/ENGINE_PARITY.md`; RL side
`rl/decision_policy.py`; B5's objective (record decisions are option sets too).
**Steps:** per R3: finite records (product ≤ 32 jointly; above, field by field on the shared prefix) and finite
tagged unions; `decide()` gains `fields`. One client implementation serves llama-server and wllama; Python native and
SGLang take the same items.
**Done:** parity tests pass; wllama smoke ranks like native within 2.5 nats; Python vs SGLang argmax agree on 20
record decisions; ENGINE_PARITY row.

### B11. The idiom floor in `natlang check` (WS3 I1; 2 agent-days)
**Inventory:** `compiler/inline.ts:12-26` (`NatlangDiagnostic`, severities), producers `compiler/policy.ts`,
`compiler/sequential-loops.ts` (the one warning today, the model for these), `compiler/project.ts:596,688,730`,
`cli/commands.ts:978` (`hint` in JSON), `native/decision.ts` (`finiteValues`), browser `check` path.
**Steps:** add `compiler/idiom.ts` with the four exact rules of `03_SEMANTIC_STDLIB_AND_PATTERNS.md` §3.1, as
warnings, with the exact messages in `04_DOCS_EXAMPLES_AND_AUTHORING.md` §8. No suppression syntax. Run it on the repo
and record counts per directory in the commit message. Do not fix the programs here; WS7 does.
**Tests:** `ts-host/test/idiom-floor.test.mjs` (true positives from cave-mission and natural.mjs texts; zero on
`examples/triage`, `examples/review`, `applications/notebook`, `applications/helpdesk`).

### B12. Structural recursion and `ensures:` (WS1 L7; 4 agent-days; two agents)
**Inventory (recursion):** `runtime/kernel.ts:526-545`, `runtime/context.ts:13,223-264`, `runtime/hooks.ts:22`,
`runtime/loader.ts` (frontmatter keys), `TOOLS_PROMPT` sentence "functions may not call themselves recursively" (a
model-facing fact fix; coordinate with B2), skill error table.
**Inventory (ensures):** `native/refinement.ts:79-97`, `compiler/refined-slots.ts`, `runtime/kernel.ts` repair budget
(`maxFailureRepairs`), `runtime/loader.ts`, `native/untrusted.ts` fencing, judge cache keys.
**Steps:** per L7. **Tests:** `nl-recursion.test.mjs`, `relational-refinements.test.mjs`.

### B13. `@natlang/std` skeleton with convergence and workspace combinators (WS3 S2; 2 agent-days)
**Inventory:** new `ts-host/std/` package and its exports in `@natlang/node` / `@natlang/browser` (package.json
exports, build); `codebases/std/` (fold in, remove); `runtime/iterate.ts`; `native/scoped-fs.ts:239-650` (Folder
transactions); `runtime/pluggable.ts`; how a package's `.nl` files ship and load (`runtime/loader.ts`,
`virtual-project.ts`).
**Steps:** create the package. Ship `semanticRepair`, `semanticMerge` (with `intentOf.nl`, `compatible.nl` as shipped
`.nl` functions), `fixedPoint`, `proposeAndVerify`, `branchAndSelect`. Calls nest so traces show per-step records.
**Tests:** `ts-host/test/std-converge.test.mjs`, `std-folder.test.mjs` (Node + browser, mock driver) and one live
LFM3 smoke via the ledger.

### B14. Flagship benchmark harness (WS6 A0; 2 agent-days)
**Inventory:** teacher provider transport for Arm A (`ts-host/src/model/pi-provider.ts`, `openai-codex` profile, how
`teacher_gen.py` calls Luna); local serving of the current policy (`ops/current-policy.json`,
`serve_native_checkpoint.py`, `natlang setup --native lfm-latest`); `natlang specialize`/`compilations`, `natlang
tune`; call store for token and latency accounting; `held_eval compare`'s statistics (reuse its permutation and
bootstrap code).
**Steps:** `scripts/semantic_fp/bench.py`, which runs arms A-D (WS6 §4) on a flagship's held split and writes
`bench/results/<date>.json` with the WS6 metrics. Validate on `examples/triage` with a small held split.
**Done:** a triage result file with all four arms.

### B15. Memetic tasks in expert-iteration collection (WS5 F1; 0.5 agent-day)
**Inventory:** `scripts/neuralese_data/expert_iteration.py` `collect` (forces `--memetic-share 0`), `grpo.py`
(`--memetic-share` default 0.02, L5634; `--memetic-tasks`), registry role `grpo-memetic-tasks`,
`rl/memetic_tasks.py`, `rl/memetic_workspace.py`, `sft_sources.py` (how a memetic supervisor episode's own success
becomes rows).
**Steps:** collection draws memetic tasks at 0.02 by default (`--memetic-share 0` opts out). Own successes of the
supervisor's calls enter the mix as own successes. Check that `memetic_workspace` evaluation runs on the collection
actor without starving it.
**Done:** a collection log shows memetic groups; a build shows their rows; DECISIONS entry.

### B16. `natlang:learning` conformance (WS8 N1; 2 agent-days)
**Inventory:** `conformance/neuralese/cases/termination-learning-graph.yaml` (`grad-context-item`,
`grad-stop-gradient`, `grad-nested-first-order`, `grad-log-likelihood`, `graph-replay-fixed-choices`: pending; the
`iterate-*` cases there are pending too, take them as well); `ts-host/src/neuralese/learning.ts`,
`native-learning-client.ts`; Python `serve/http.py:352` (`/v1/neuralese/grad`); `browser/index.ts:42`
(`ModuleUnavailableError`); `docs/ENGINE_PARITY.md`.
**Steps:** make each pending case run and pass (fix code where it fails). Wire them into the conformance runner that
`check-main.sh` uses. Add an ENGINE_PARITY row naming the per-engine blocker for gradients.
**Done:** cases `implemented` and green.

## Next batch (start as this one frees agents)
L1 tuples, L2 static checks, L6 modules, R2 `search_code`, R5 `natlang profile`, F5 website, F6 playground rows 1-8,
A1-A3 flagships, M2 migrator (after B7), S1 collections, T3 task-author (after B6 + S1), H1c verifier-task update
(after B6).
