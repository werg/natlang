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

## Typed output contracts

Landed 2026-11-23 (DECISIONS "typed output contracts"): `writes:` on directory reducers, alias doc comments shown, the
authoring rule (a prose `Output contract:` paragraph is an error), the migrated pool (`*-typed` sources), a root's
`ensures:` checked in collection. Open:
- **File text is typed `string` plus a doc comment.** The runtime checks which files a call writes and literal-typed
  contents, not line formats: a line template (`<id> | <change>`, one decimal place) is still prose, in the field's
  doc comment. A crisp line grammar (TypeScript template literal types per line, or a record per line with a declared
  rendering) would make it a checked type on every engine's result schema too. Owner: typed-contracts follow-up, after
  the next held measurement shows whether the migrated rows change pass rates.
- **`Is<string, P>` file types are a load error** (the check runs at return and would need a judge on every written
  file). Owner: same follow-up, with the crisp-checker registry.
- **Declarations derived from instructions can miss a conditionally written input file** that its step names only as
  read (`ledger.csv` in `verified-folder-edits` w437-p4a: the reference installs nothing, so the declaration reveals
  it). verify.mjs dropped that row through its keyword baseline; rows without a baseline are not caught. Owner: next
  authoring round writes `writes:` by hand (AUTHOR_BRIEF_ROUND23.md); the migrated rows stay until then.
- **32 rows that already failed qualification in their original form** (17 reference replays: `selected-drafts` w438
  wrong_return, `settings-merges` w432 unanswered child `intentOf`; 15 with negative controls that error on the missing
  Neuralese writer of the replay) were left out of the typed sources with the others. They are in the old datasets and
  in no current role. Owner: task-author batch that fixes the w432/w438 references.

## Held and data

- **held-v4:** no `branch_and_select` task; B1's 104 rows keep `invented:` groups; the 100 rubric workflows and 2 reserved apps are outside the set. Owner: next held-v4 extension; due before the first held-v4 cadence measurement after round 3.
- **Flagship held splits** are 2–5 fixtures per app, too few for p < 0.05. Owner: task-author tool batches, next semantic-FP batch.
- **`compatible.nl`** judges additive edits incompatible. This is a training target through the sfp3 rows, which are in the pool; watch its family in the health report.

## Held failure taxonomy (2026-11-23, `plans/neuralese/HELD_FAILURES_20261123.md`)

- **Neuralese read-back: fixed (d16437be8, f109bedfd; DECISIONS 2026-11-23 21:30).** A text-written state was encoded with no
  context and could not be read. Engines now read it as its tokens, inline in the reader's context with no markers
  (`in-context-text`, opt-out `NATLANG_NEURALESE_TEXT_IN_CONTEXT=0`). Native probe: 11/12 after, against 0/12 before
  on the live SGLang front. Follow-ups:
  - **Re-enable Neuralese variants in aei1's collection and SFT mix** at the next pin with d16437be8. Measure the
    collection's read-back rate with the failure-taxonomy `readback.py`. Owner: coordinator.
  - **Marker rows: done (0462075ea; DECISIONS 2026-11-23 23:00).** Text-initialised, with their own AdamW group at
    10x LR and an FP32 master copy in merged checkpoints. `exp0_sft.write_checkpoint` is removed. `produce_close` is
    explained: producer documents end at the payload, and no loss reads the close. Consume markers are banned in every
    draw on every engine (cd6d9c3dc). These enter aei1 at the next round boundary on a learner pin with 0462075ea.
    Owner: coordinator. Open: spontaneous marker emission at temperature 1, and the rows' held effect.
  - **Produce spans read in context: done (d6b3218e7, cd6d9c3dc).** On the native probe the span arm went from 0/8
    to 4/8. Reader rows from recorded writes were not added; the reason is in DECISIONS. Open: measure the
    collection's read-back rate on the new pin, and reconsider trained readers if it stays well below the inline arm.
    Owner: coordinator.
  - **Cross-engine moves: done.** `producer_kind` is in the state header and every engine applies one rule: Python
    native, SGLang, and fork natlang-b12092 for llama-server and the page (f9bee8f6c). Remaining: the consume-marker
    ban reaches llama-server and the page only with the next GGUF export (ENGINE_PARITY gap row). Owner: whoever
    exports next.
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

### Task-authoring briefs A–E: first batch admitted (DECISIONS 2026-11-23 23:45 "Held-failure task batch")

343 verifier-judged tasks (w440–w459) are qualified and blind-solved. `natlang-rl-task-supply-20261123hfb` holds 242
train rows in `grpo-pool`; the held rows are in `natlang-rl-authored-{dev,confirmation}-20261123hfb`.

| Brief | Admitted | Train rows |
|---|---|---|
| A | 67 | 42 |
| B | 49 | 39 |
| C | 60 | 50 |
| D | 66 | 44 |
| E | 101 | 67 |

Brief E covered only the Neuralese-free capabilities. Open:
- **The current policy's pass rate on the 242 rows.** The Colab front was lost after 13 episodes (3 scored, 0 passed).
  Measure on the next actor: `runs/hfb/make_measure_manifest.py <supply cases> <dir>`, then `held_eval measure
  --allow-other-engine` from a pin. Owner: coordinator, with round 2's collection health report (families
  `auth_depot_folder_dispatch` … `auth_multi_round_state_games`).
- **Declare the hfb folder tasks' files in `writes:`** (A, and the folder programs of B, D and E). They are verifier
  tasks, which rule 6 exempts, and their file formats sit in the numbered steps. Do it once collection pins read
  `writes:`; the migration must declare conditional files too. Owner: typed-contract agent or the next author batch.
- **Brief E with Neuralese:** the capabilities above with Neuralese variants, now that read-back is fixed (d16437be8).
  Owner: next task-author batch.
- **`register_authored_held.py` and `build_supply.py` write under the running checkout's `runs/`.** From a worktree, the
  packets land in the worktree, and custody loses them when it is removed. Resolve `runs` to the canonical data root, or
  refuse a worktree. Owner: next task-author batch.
- **The static verifier controls lack `read_code`**, which the collector replay has. Keyword baselines that explore code
  must inline the sources. Owner: next task-author batch.

## Operations

- **aei1 remote collector runs old code:** `aei1-collect-v0000-remote0` was started at 19:15 on 2026-11-23 from pin 108cd84. Its ts-host lacks 98651c669, so against an actor serving the read-back fix it throws `neuralese-native-grad-invalid` on in-context reads. Replace it with a collector at the driver pin at the round boundary, or when the A100 actor is served again after round 1's SFT. Owner: coordinator.
- **aei1 driver controller owned by the Vibestudio session since 22:41 on 2026-11-23:** `resume-semantic-reader-owner.py` replaced the coordinator's driver with the same argv (pin bdf4c8721, variants on), with controller code from 97e82bcd5 on origin/main and `--data-repo`/`--fill-repo` at its reader pin. The coordinator accepted this: the code contains the hand-back. Its unit is `aei1-drive-reader-v14-final-20261123`. Coordinate before restarting it.
- **aei1 round 1 -> 2 boundary checklist (coordinator, about 06:00 on 2026-11-24):**
  1. Move the driver to a pin at or after 78fb05054: read-back fix, decision-budget fix, memetic leak fix, Colab-via-Drive.
  2. Re-enable Neuralese variants once the L4 actor runs the read-back fix (rollout agent, 2026-11-23 21:45).
  3. Take the Neuralese marker rows and produce spans: these landed on main (head dda393bc7). Learner and collection must move together, because learners score the consume-marker bans that collection records: 0462075ea plus cd6d9c3dc for the learner, d6b3218e7 plus cd6d9c3dc for the SGLang fronts. The Colab VM actor needs the same commit.
  4. Admit the new task batch from briefs A-D (authoring agent): done, `natlang-rl-task-supply-20261123hfb` is in grpo-pool (242 rows; briefs A–E).
  5. The typed-contract pool replaces prose `Output contract:` tasks (typed-contract agent).
  6. Rerun bench arm D and the GRPO gradient bench.
  7. The held reference (sft-v9) is re-measured on the new scoring series.
- **GRPO decision-reasoning gradient bench:** `gdr-bench-9` (20 GB, low priority) was cancelled at 20:20 on 2026-11-23 because it held the DGX ledger head for 2 h behind round 1's SFT. Rerun `/home/werg/data/lfm3-rl-dgx-20261029/ops/gdr-bench/bench.sh` after the r0001 SFT finishes; its pin `pin-91b26036f` is kept until then. Owner: coordinator.
- **Neuralese variants are off in aei1's collection** (`--no-neuralese-variants`, driver restarted 19:15 on 2026-11-23). Read-back of Neuralese values is broken: the reader gets the producer's content in about 1% of read-backs (HELD_FAILURES_20261123 finding 1), so variant groups were teaching the policy to skip program functions. Two investigators are on it: a live hunt and an independent static analysis (`NEURALESE_READBACK_STATIC_20261123.md`). Turn variants back on at the round boundary after their fix lands and read-back accuracy is re-measured. Owner: coordinator.
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
