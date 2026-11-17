# Vibestudio fixtures through canonical program IR and backbone SFT

Vibestudio suites are a frontend for `natlang.program/2`, not a separate training
format. The full model source, real entrypoint, JSON inputs, host fixture source,
selected case, expected requirements and source lineage remain durable in the IR.
The standalone Vibestudio episode-to-chains implementation was removed; Git is its
history. No measured corpus had used that path.

## Shared mechanism inventory

| Component | Responsibility |
| --- | --- |
| Vibestudio `scripts/natlang-teacher.mjs` | Export original source and complete split inventory to ProgramRecords, launch the ordinary collector, perform isolated semantic review and bind its evidence. |
| `ts-host/src/teacher/program.ts`, `evaluation-fixture.ts` | Neutral canonical fixture descriptor and execution frontend; real NL entry or authored TS entry with explicitly identified NL semantic anchor. Host fixtures/scorers stay outside model-owned files. |
| `scripts/neuralese_data/teacher_gen.py` | Explicit repeatable source pools, held exclusion via `grpo.load_tasks`, root mode, resumable teacher collection. |
| `ts-host/src/teacher/collector.ts` | Ordinary hosted/native provider execution, request/reply journals, `programRow` and `natlang.teacher_trajectory.native/1` containing `task.program_ir`. Hosted teachers need no native sampling probabilities. |
| `rl/task_rollouts.py` | Ordinary collector scoring, source-custody binding and `natlang.rollout/1`; real invocation metadata follows the canonical turns. |
| `scripts/neuralese_data/teacher_admission.py` | Validate supplemental source/review evidence and invoke the existing collector replay. It does not render tokens or produce a second trajectory protocol. |
| `scripts/neuralese_data/sft_sources.py` | Shared extraction, exact collector replay, held/protected exclusions, rendering, loss masking, selection and function target-mass census. |
| `scripts/neuralese_data/exp0_sft.py` | Existing FP32-master/BF16 learner consuming the shared builder's `chains.pkl`. |
| `register_corpora.py`, `sync_training_corpora.py` | Existing immutable registry/manifests and data transport. Collection, admission, staging and learner exposure remain separate. |

No engine protocol, scoring, sampling, native-state or cache change is introduced.
The source adapter emits ordinary canonical teacher/rollout records. Tokenization
uses `sft_sources.TemplateRenderer` (tokenizer-only, no model); training uses the
existing `exp0_sft.chains_of` representation. Managed llama-server, wllama, Python
native serving and SGLang continue to consume their existing serving protocols
(`docs/ENGINE_PARITY.md`).

## Canonical collection

The host first exports the **full** source inventory (train, validation and test)
and a selected training pool. Each ProgramRecord records explicit split, stable
`source_namespace`, source ids/groups/revisions and the real fixture entry.
Source inputs include the selected fixture case/world as well as function JSON
arguments: service-only functions legitimately have empty argument records.

Run from the experiment pin through the memory ledger on the DGX:

```
python scripts/neuralese_data/teacher_gen.py --work POOL_WORK \
  --pool SOURCE_IR.jsonl --held-manifest HELD_V3/manifest.json --root-mode authored \
  --teacher '{"provider":"openai-codex","model_id":"gpt-6-luna","reasoning_effort":"low","slots":2}'
```

`--pool` is repeatable and replaces the default GRPO pools. Default old campaigns
retain `--root-mode student`. An evaluation fixture's authored mode preserves its
real source entry and records `authored_fixture_entry`; it never fabricates a
reference eval or an NL forwarding wrapper. Tasks and source-pool receipts bind
resumes; changing explicit pool content, held selection or root mode needs a new
work directory. Existing default student campaigns predate this receipt and can
still resume.

Ordinary durable output is `manifest.jsonl`, per-task `cases.jsonl` and queues,
canonical native teacher results/journals, and scored `rollouts.jsonl`. The native
teacher result contains the original ProgramRecord, actual requests/replies,
traces, execution graph and fixture outcome evidence.

## Supplemental admission

For a fixture task the teacher-pool manifest entry declares
`admission: {path, sha256}` for a `natlang.teacher-admission/1` sidecar. The generic
checker is shared by host diagnostics and SFT extraction:

```
python scripts/neuralese_data/teacher_admission.py --pool POOL_WORK
```

The sidecar binds canonical cases, rollouts, full original ProgramRecord split
inventory, exact teacher identity, and per-sample native result file hashes and
collector program-IR digests. Each sample retains the actual suite score/gates,
full fixture outcome evidence, collector code provenance, exact replay verdict
and an isolated semantic review with its full request/reply and reviewer identity.
Semantic review is fallible admission evidence, never labelled exact gold and
never supervised as an executor target. It may use the same model in an isolated
conversation; this does not alter the stock collector's distinct-model judge
policy.
The review request must be exactly one system message and one user message;
its decoded evidence must match the original selected case and actual native
fixture outcome, observation and observation store. Rehashing another case's
accepted review cannot attribute it to this result.

The generic replay helper uses `sft_sources.ReplayServer/replay_episode` and emits
receipt drafts, not a new trajectory representation:

```
python scripts/neuralese_data/teacher_admission.py replay --pool POOL_WORK \
  --runtime-repo PIN --work FRESH_REPLAY_WORK --out admission-drafts.json
```

The host adds full-inventory bindings and retained semantic reviews to these
drafts, records rejected samples separately and attaches the sidecar hash to the
ordinary teacher-pool manifest. Replay resume identity excludes these admission
links and uses git provenance rather than ephemeral checkout paths.

Admission excludes nontraining splits, protected source-group siblings and
identical input/world scenarios crossing splits, including renamed groups and
new program revisions. It requires full fixture success, all gates, no failed
tool actions, a bound exact replay and accepted semantic review. Review failures
remain collected evidence, not training targets. File hashes and replay receipts
are trusted collector-code attestations, not signatures or adversarial proof.

## Shared SFT and the next experiment

```
python scripts/neuralese_data/sft_sources.py extract --work SFT_WORK --teacher POOL_WORK
python scripts/neuralese_data/sft_sources.py replay --work SFT_WORK --runtime-repo PIN
python scripts/neuralese_data/sft_sources.py build --work SFT_WORK --out BUILD \
  --tokenizer TOKENIZER --held-manifest HELD_V3/manifest.json
```

Use the normal builder's source-mix, base-rate and budget flags for the experiment
recipe; there is one renderer, loss objective and trainer. Fixture sources without
validated sidecars remain unadmitted. Admission evidence is checked again before
build, and current-runtime exact replay must pass. Held-v3/v2 exclusions apply via
the existing shared builder. The teacher's own actions and answers carry loss;
prompts, earlier history, tool observations and an inserted empty think prefix
carry zero weight under the existing masking mechanism.

`build.json.target_mass` reports actual target and weighted target tokens by
suite/function/root-or-child and totals for root, child and unknown. Names,
definition ids and parentage come from actual invocation ledgers. Missing identity
is explicit unknown; forwarding-wrapper mass is measured, not silently reweighted.

Register the actual source collection and built data through the existing corpus
registry and immutable manifests, record project-generated/provider/template
license provenance, then transfer to the actual DGX/Pop owner. Registration is not
learner exposure. Adopt the corpus at the next experiment pin; never change data
inside a measured live run. Judge task success on paired held-v3 evaluation and
separate unseen Vibestudio scenarios, not training loss or fixture case counts.

## Source held separation checked, 2026-11-17

Ledger unit `vibe-teacher-held-source-verify-20261117` ran the standard
`held_eval verify` on Tick against all 18 preflight-exported canonical source
pools: 186 training roots/source ids and 219 split/source groups. New source alone
has **0 held task-id and 0 held-group overlaps** against both held-v3 and its
held-v2 predecessor. Combined with the original held-v3 exposure snapshot, 9,613
ids and 1,608 groups likewise have **0/0 overlaps** against held-v3. The preceding
baseline snapshot check independently found 0/0 for 9,427 ids and 1,389 groups.

The immutable held manifests were unchanged (v3 `45471060`, v2 `dfe0a915`). To
verify on a machine without DGX's external mount, `held_eval verify --pool PATH`
now accepts relocation only when the file SHA256 equals the original manifest's
pool SHA256; held case bytes are also checked. The data-only relocated source
pool is 341,446,694 bytes, SHA256
`b3fef4c7dcd36e0bedbd03c67353634b5d7323ad0468ac1d0f73533f2e5de966`.
No protected inputs were sent to a model. No job was launched on the DGX.

Full per-source input hashes, original exposure artifact hashes, relocated pool
identity and the three actual verifier reports are in
`/home/werg/data/held-verification-20261117/source-verification.json` (SHA256
`576c26d079cc2eccbe35cd7cb305b32b9abe4dbf4c46a2a6086ecc6109f8bba2`).
The source paths are under
`/home/werg/.cache/natlang/vibe-teacher-ir-preflight-20261117/`; the receipt binds
each `pool.jsonl` and full `inventory.jsonl`. Some early descriptor schemas were
subsequently amended; their source ids, source groups and inputs were unchanged.
This verifies the source exposure closure, not final descriptor execution or
learned quality. Final-schema canonical collection exports fresh pinned IR.

Verifier relocation/integrity tests and the existing curriculum tests passed:
45 tests in ledger unit `held-source-relocation-tests-20261117`. The actual
three-report source verification completed successfully with a 403 MB peak.

## Qualified live pilot and final training input

The executable inventory now has 39 original NL functions across the same 18 suites and 450 cases
(186 train, 128 validation, 136 test). Documentation query intent became a typed `keywords.nl` function;
filesystem paging, lexical query indexing and ranking remain exact helpers, and `fits.nl` judges support.
The live failures exposed truncated search recall, scoped directory-name semantics, an unnecessary query vocabulary
restriction and whole-phrase indexing. Those mechanisms were corrected generally, with original-source regressions.

Luna collection on Natlang `67a3eb693` qualified GitHub access, server-log briefing and documentation lookup:
three admitted episodes and 39 real executor calls (3 root, 36 child, zero unknown attribution). Each has complete
suite quality, no failed actions, exact replay and a separate accepted semantic review. The docs sample quotes
the actual single-panel `checkPanel` instructions in `skills/workspace-dev/TOOLS.md`; no source labels were inserted
into its model context. Qualified template/host source is embedded in each canonical ProgramRecord.

The shared pipeline then replayed all three on merged Natlang `efdecc9fe` and built with the real LFM3 tokenizer:
39 `chains.pkl` sequences, 55,947 forward tokens, 308 target / 269 weighted target tokens. The source and build pins
are deliberately distinct provenance: replay qualifies compatibility rather than relabeling the collection commit.
Build unit `vibe-teacher-final-sft-build-20261117` exited zero, peak 929 MB. The only learner input is
`/home/werg/data/vibestudio-teacher-pilot-20261117/build-v2/chains.pkl`.

`pilot-build-v2-result.json` in that parent directory records checks and immutable hashes, including chains SHA256
`5bf3e251b74bf343303bea6f4d86380d80764fb91a79dee8f1441c270e3f1c90`. Raw canonical sources, verification evidence,
current extraction/replay and rendered build were transferred as data to identical DGX paths: 337 selected files,
139,355,613 bytes, independently SHA256-verified. Exact selection and transfer receipts are at the adjacent
`vibestudio-teacher-pilot-20261117-v2.transfer-{plan,receipt}.json`; the registration candidate is the adjacent
`-v2.registration-entry.json`. The previous two-episode build is not an additional training corpus.

Five excluded diagnostic roots remain evidence, including a recovered success with a failed local action.
Across eight development attempts there are only three unique selected task IDs, not eight independent tasks.
The manifests retain 64 rollout turns; the budget-stopped attempt separately retains 40 observed partial turns.
Only 39 verified calls enter training. Source and program mechanisms changed during diagnosis, so this sequence
is not an unbiased teacher acceptance-rate estimate. No optimizer or learner has consumed these artifacts.

Current-main integration is qualified by full merged checks plus scoped corrections: 1629 general Python,
2229 Neuralese, 2618 other TS tests and 22 conformance cases passed on `6d921fca4`; its one Pi-bundle dependency
failure was fixed in `36ca3a313` and verified by 28 actual bundle/fixture/adaptation checks, both TS configurations
and layering. The incoming operator-budget bypass was fixed in `efdecc9fe` and verified by 23 operator/decision
checks. The latest main quick gate passed. Browser forced capture remains explicitly blocked on its released public
tokenizer/marker-ID ABI, rather than being claimed as implemented; hosted fixture collection does not use it.
