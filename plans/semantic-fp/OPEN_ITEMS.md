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
| M2 migrator: one shared `stepKind`/`instructionsOf` in std, `migrate.nl` replaced by a TypeScript driver, P1/P2 measurements, pairs dataset | M2 agent |
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

## Operations

- **Benchmark arm D (specialize, tune, escalate on sft-v9):** its 24 GB tune job (`bench-tune-b2`) was cancelled at 18:40. It sat at the head of the DGX ledger during round 1's SFT and froze 18 jobs behind it. Rerun arm D (`/home/werg/data/semantic-fp-bench/triage-20261123b/driver.sh`) after the r0001 SFT finishes. Owner: coordinator.
- **Constrained-decoding rollout (9ab63c622, SGLang fork 17e3458c2e):**
  - The DGX checkout `/home/werg/sglang-natlang` moved to 17e3458c2e on 2026-11-23 17:10 (xgrammar 0.2.7 in its venv).
  - Read round 2's collection throughput, and set `NATLANG_TOOL_GRAMMAR=0` if it drops.
  - Move the Colab VM's checkout at the next actor rebind.
  - Install xgrammar in Pop's serving env between held measurements.
  - Owner: coordinator.
- **Drive cold tier and Colab-via-Drive wait on one owner login (2026-11-23).** The DGX's ADC has no Drive scope, and a
  Colab `drivemount` needs a browser consent per runtime. The plan is sized to the measured link: uplink 5.2-5.8 MB/s,
  Google downlink 11.9 MB/s, and an HDD inflow of about 100 GB/day (439 -> 48 GB free from 11-19 to 11-23). After the
  owner adds `drive.file` to ADC (DECISIONS 2026-11-23 storage), build the third tier:
  - archive HDD-cold data to Drive at a cap of about 3 MB/s (about 260 GB/day), paused while a Colab transfer runs;
  - verify the md5 Drive reports against the local file before deleting the HDD copy;
  - keep a manifest of what lives where, so a restore is one command;
  - push short-lived access tokens to Colab VMs, so bootstrap pulls checkpoints from Drive and learners write state
    there.
  Owner: storage agent, as soon as the scope exists.
- **NVMe stays near its floor:** the cold NVMe data is many-small-file trees, which the per-file archiver (files of at
  least 1 GB) never moves:
  - `ops/ei/sft-v7/collect/groups` 26 GB and `sft-v8/collect/groups` 16 GB;
  - `ops/grpo-v4/run2/groups` 23 GB.

  Pack each closed run's group directories into one zst per group, which `rl_rollout_files` already reads as
  `g*.rollouts.jsonl.zst`, or move the trees to the HDD/Drive after checking their readers. Owner: coordinator.
- **The HDD is dominated by data outside natlang's keep rules:** `bgkit-data` holds 591 GB (another project). It is the
  first Drive candidate once the tier exists, and the owner's call.
- **`vibe-native-v9-reader-pin*` units failed at 08:15 on 2026-11-23:** `.venv-decider-systempython conflicts with the intended shared root link` in `archive-code-00ab72d28dbc`. The Vibestudio session later reached v9 through other units (`vibe-student-fill-keeper-v9`, `vsingest-semantic-poll-v9`), so this is superseded; the owner is the Vibestudio session.
- **The Colab keeper's real replacement and upgrade path is untested live.** It is watched by the ops watchdog (`colab` checks).

## Deferred on purpose

- **A preempted cycle collection gets a fresh `--hours` window:** affects only the legacy cycle path, not async rounds.
