# Open items register

This register lists every known open item, each with an owner or a deliberate deferral. It is kept by the coordinator and reviewed at every 4-hourly review.

An item leaves the register when its fix lands on main. A finding reported by an agent enters here the same day, unless that agent is fixing it itself.

## In flight

| Item | Owner |
|---|---|
| B10 four-engine smoke, fork pin move to b12088 (reasoned decisions on llama-server and wllama), ENGINE_PARITY rows | B10 agent |
| Decision option scoring under the standard tokenization only. The model may generate another split (`"`,`spam`,`"` vs `"sp`,`am`). Sum both tokenizations per option on all four engines and the learner (8b34839ab) | B10 agent, after the fork release |
| Tool-call grammar on the collection actors: move the DGX and Colab SGLang checkouts to fork 17e3458c2e (`NATLANG_GRAMMAR_PREFIX`; `training/neuralese/sglang-fork.json`) at the next boundary with no live actor; the Python engine and the learner already apply it (DECISIONS 2026-11-23 "constrained decoding with an exact learner law"). Watch `law_masked_fraction` in `train.jsonl` and `natlang_tool_grammar_refused_openings_total` | coordinator, at the next actor boundary |
| Managed llama-server and page drop the tool grammar under `x_natlang_rl_capture` (ENGINE_PARITY gap row): exact replay needs the fork to compile the same xgrammar structural tag (server and wasm build) and record it | unassigned (needs a fork release; no training source reads fork captures today) |
| Benchmark arm D on sft-v9 (specialize, tune, escalate); then regenerate `examples/triage/bench/results/2026-11-23.json` with D | armed driver `semantic-fp-bench/triage-20261123b/driver.sh` (waits for the policy server's 18.9 GB ledger claim); the coordinator commits its `results-ABCD.json` |
| Memetic layout fix (c64d250b2) on a real actor | queued unit `memverify-dgx3`, then the round 1 health report |
| held-v4 sft-v6 reference measurement | Pop unit `popheld-hv4-sft-v6`; the coordinator appends the result to DECISIONS |
| Remove the legacy `expert_iteration.py cycle` mode once aei1's round 1 has a held verdict (AGENTS.md: async-drive replaces cycle) | coordinator, after the r0001 verdict |
| aei1 round 1 build, SFT and held verdict | aei1 driver; the coordinator reads its health report |

## Reasoned decisions in training

Found by the GRPO-reasoning agent (DECISIONS 2026-11-23 "GRPO trains the action reasoning"). Owners: the scorer re-tokenization goes to the B10 agent, after its pin move and with the tokenization sum, as one fork release.
- **The scorers re-tokenize the reasoning.** Every engine scores the options after the reasoning *text*, re-encoded, not
  after the sampled ids. In aei1 r0000 (16 action readouts, groups g000689/696/701/720) the scored prompt leaves the
  sampled ids mid-reasoning in 4 of 16 (after 847 of 3476, 531 of 1051, 535 of 2085 and 953 of 1687 tokens; the decoded
  text is the same). The choice is then scored, and trained, after a tokenization the model never produced. Fix: the
  score item carries the sampled reasoning ids, and every engine (Python native, SGLang, llama.cpp fork, wllama) uses
  them. GRPO's lowering counts it each step (`decision_reasoning_ids_rendered` in `train.jsonl` `lowering`).

## Corpus migration (M2 follow-ups, DECISIONS 2026-11-23 18:30)

Unowned until the coordinator assigns them:
- The 6 rejected design-brief gates (012, 013, 017, 021, 026, 028): retry through `nextAttempt.nl` with the comparison's problems.
- Driver pilots 036, 040 and 057 (Luna-written, type-checked): measure the judged workflow cases, then settle.
- 061: rejected because the signature list changed.
- LFM3 fails `rewriteDriver` on all 4 drivers: this is a training target.
- The rest of the design-brief corpus (100 programs, 1,390 `.nl` files; 15 gate leaves are migrated) and all 11 Astra/Sol apps (their world baselines never completed).
- 65 P2 candidate attempts ran on the Colab front (BF16) against DGX (FP8) originals: re-measure them on one engine.
- Admitting `semantic-fp-migrated-tasks-m2-20261123` (42 rows) and the 14 pairs into expert iteration.

## Language and runtime gaps

Closed 2026-11-23 by the language-gaps agent (DECISIONS "Semantic-FP language gaps and applications closed"):
`natlang improve DIR` proposes from the program's runs; `escalate`'s nl policy reads its history from the call store;
`.d.nl.ts` `import type` for `uses:`; `nl.with<T>` in callable-folder factories typed from `import type`; `natlang check
--idiom`. Found while closing them, not changed:
- **`natlang improve DIR`:** memetic guidance changes are not a transformation yet (WS8 N4 names them); the improver
  revises `.nl` instructions only. Owner: language-gaps agent's next session, or N5.
- **`Untrusted<T>` in the migration loop:** codemod file texts stay fenced (the fence is keyed by text), but a site
  excerpt or an edited revision of an untrusted file is new text and is not marked. Owner: the untrusted-contract
  follow-up.
- **Type-only imports in callable-folder TypeScript** that resolve to nothing still make the type `any` silently (the
  module's TypeScript diagnostics are not reported). Owner: language-gaps agent.

## Applications

codemod-studio (now `applications/migration/codemod`, on the migration loop), sem-grep-sql (now `applications/nldb/grep`,
on nldb's SQLite engine) and ci-doctor (natural-language bisection stop, root-cause clustering) are closed. Open:
- diligence has no Neuralese views or tuning.
- `ts-host/test/repository-migration.test.mjs` keeps its own copy of the scripted migration stages
  (`test/support/migration-stages.mjs` is the shared one); fold it in when that test next changes.

## Held and data

- **held-v4:** no `branch_and_select` task; B1's 104 rows keep `invented:` groups; the 100 rubric workflows and 2 reserved apps are outside the set. Owner: next held-v4 extension; due before the first held-v4 cadence measurement after round 3.
- **Flagship held splits** are 2–5 fixtures per app, too few for p < 0.05. Owner: task-author tool batches, next semantic-FP batch.
- **`compatible.nl`** judges additive edits incompatible. This is a training target through the sfp3 rows, which are in the pool; watch its family in the health report.

## Held failure taxonomy (2026-11-23, `plans/neuralese/HELD_FAILURES_20261123.md`)

- **Neuralese read-back loses the content.** `natlang:model-read/1` returned the producer's text in 1 of 87
  read-backs (sft-v7), 1/87 (sft-v9), 0/24 (sft-v6 held-v4) and 13/465 in aei1's DGX collection. The state is imported
  (`exact-native-state`, producer's id), yet a two-line string reads back as `jinja2`. Spec-following episodes fail on
  it, while bypassing episodes pass (33% vs 64% pass rate), so expert iteration rewards not following the program.
  Find where the content is lost: the producer's state vs a text-primed span, the import positions, the template
  readout, or an untrained reader. Episodes: `ops/ei/sft-v9/held/failure-taxonomy-20261123/readback.py`. Owner:
  coordinator assigns a bug hunt, ahead of more Neuralese-variant collection.
- **Harness messages found, not changed** (`HELD_FAILURES` 3.8). Owner: runtime-messages agent, next session:
  - the "cut off" footer after a `console.log` that printed the whole value;
  - a rejected service write that reads as a partial write;
  - `finish:true` accepts record fields outside the declared type;
  - `return_result` refuses a JSON string of the declared record, with an unrelated `nl.with<T>` hint;
  - `moveTo`/`folder.file` silently take `[object Object]` paths or nest a file under itself;
  - the sandbox `mv` will not overwrite and says "cannot safely determine whether … are the same file";
  - `read_code("refine")` answers no-such-function;
  - "Binding is immutable" never names `const`;
  - `const intent = await intent(x)` gives a raw TDZ ReferenceError;
  - leaked `</think><tool_call>` text in eval code.
- **Held-v3 errata review:** six tasks with a suspect gold or oracle (`HELD_FAILURES` 3.9: rows 130, 095, 235, 249,
  204 and the QA span oracle). Owner: next held errata pass.
- **Decision-readout tie-break:** three 2–2 ties of the four samples all went to needs-review, wrong each time
  (held-v4 design-brief gates). Check the tie rule. Owner: B10 agent.

### Task-authoring briefs (owner: next task-author batch)

Seed from new domains and existing datasets, never from held inputs or their siblings (`held_eval.exclusion_ids()`).
Each task gets an exact oracle with reference, negative and alias replay. Measure repetition against the pool before
admitting the batch.

- **A. Directory reducers with exact effects.** Programs that read a folder of notes and must move, not copy, each
  note into destination folders at the folder root, write a summary file in an exact line template, and return a
  count. Vary:
  - where destinations live (root vs nested, stated explicitly);
  - move vs copy vs delete;
  - cut-off rules ("stop when …");
  - null or absent fields that select a default branch;
  - name vs path in output lines;
  - literal words in the template that must appear verbatim;
  - a final newline.

  Negative replays: the right summary with no moves; moves into nested folders; a copied file left behind; a dropped
  template word. Domains: depots, clinics, archives, workshops, municipal permits, school rosters (not courts, ferries,
  harbours, claims or bike docks).
- **B. Child-call discipline in eval.** Programs whose root loops over items and calls a named child function once per
  item, then aggregates. Construct pressure:
  - results stored once and reused in the next eval (an oracle on the model-call count: at most N + small);
  - children that answer directly rather than calling themselves or a new `nl` with the same instructions;
  - 3–5-level chains where each level does one step;
  - `Promise.all` over async children;
  - `for … of await folder.files(…)`.

  Negative replays: re-running the loop (budget), a self-call, by-hand computation where the child is the specified
  path (`hinted_technique` oracle).
- **C. Exhaustive codebase exploration with exact paths.** Callable-program exploration tasks: leaves under nested
  namespaces, direct children excluding the parent, functions whose instructions mention an exact identifier
  (not substring, not frontmatter args), parameter-type users including the root. Answers are full dotted paths from
  the root. Programs have 12–20 functions over 3–4 levels, so passing needs reading all of them. Use new program
  domains, not `codebases-design-briefs-0xx`.
- **D. Decision-readout calibration on hard negatives.** `decide`/`readout: decision` tasks whose label sets have
  near-miss out-of-scope requests, fine-grained question types, money words in personal (ham) messages, and
  two-team routing where the request names one team's symptom and the other's remedy. Roughly 30% of items should be
  the out-of-scope or near-miss class. Sources: intent and classification datasets outside held source groups.
- **E. Fresh siblings of the zero-rate capabilities** (held families with 0% historic pool rate). These are written
  from the capability, not the held task:
  - an obligation carried across sections of a long register into per-section notes;
  - request-to-service-call queues with conditional reject reasons;
  - reconciliation over damaged or missing folder entries;
  - an edit stream applied to a document;
  - choosing a procedure from a skill library;
  - multi-round state games with a stop rule.

  Give each a Neuralese-free variant until the read-back defect above is fixed, so own successes and teacher
  demonstrations are possible.

## Operations

- **Benchmark arm D (specialize, tune, escalate on sft-v9):** its 24 GB tune job (`bench-tune-b2`) was cancelled at 18:40. It sat at the head of the DGX ledger during round 1's SFT and froze 18 jobs behind it. Rerun arm D (`/home/werg/data/semantic-fp-bench/triage-20261123b/driver.sh`) after the r0001 SFT finishes. Owner: coordinator.
- **Constrained-decoding rollout (9ab63c622, SGLang fork 17e3458c2e):**
  - The DGX checkout `/home/werg/sglang-natlang` moved to 17e3458c2e on 2026-11-23 17:10 (xgrammar 0.2.7 in its venv).
  - Read round 2's collection throughput, and set `NATLANG_TOOL_GRAMMAR=0` if it drops.
  - Move the Colab VM's checkout at the next actor rebind.
  - Install xgrammar in Pop's serving env between held measurements.
  - Owner: coordinator.
- **NVMe stays near its floor:** the cold NVMe data is many-small-file trees, which the per-file archiver (files of at
  least 1 GB) never moves:
  - `ops/ei/sft-v7/collect/groups` 26 GB and `sft-v8/collect/groups` 16 GB;
  - `ops/grpo-v4/run2/groups` 23 GB.

  Pack each closed run's group directories into one zst per group, which `rl_rollout_files` already reads as
  `g*.rollouts.jsonl.zst`, or move the trees to the HDD/Drive after checking their readers. Owner: coordinator.
- **The HDD is dominated by data outside natlang's keep rules:** `bgkit-data` holds 591 GB (another project). It is the
  first Drive candidate (the tier exists since 2026-11-23: `drive_tier.py archive --root bgkit-data` once the owner
  decides; `NEVER` in drive_tier.py excludes it until then).
- **`vibe-native-v9-reader-pin*` units failed at 08:15 on 2026-11-23:** `.venv-decider-systempython conflicts with the intended shared root link` in `archive-code-00ab72d28dbc`. The Vibestudio session later reached v9 through other units (`vibe-student-fill-keeper-v9`, `vsingest-semantic-poll-v9`), so this is superseded; the owner is the Vibestudio session.
- **The Colab keeper's real replacement and upgrade path is untested live.** It is watched by the ops watchdog (`colab` checks).

## Deferred on purpose

- **A preempted cycle collection gets a fresh `--hours` window:** affects only the legacy cycle path, not async rounds.
