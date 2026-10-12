# Pop direct-SFT run and evaluation audit — 2026-10-12

## V18 objective, masks, and monitor

The audited run is `direct-stage-union-r2-epoch1-v18` at
`/srv/storage/natlang-artifacts/luna-neuralese-direct-lfm-base-20261011-v8/`.
Its recipe pins `latent_optimizer: lionsr`; it is historical and is not the
current Muon run. The report is at
`direct-stage-union-r2-epoch1-v18/direct_sft/report.json`: step 5595,
`completed: false`, `qualified: false`, and no final behavior-gate result.
It consumed 26,190 windows, 4,459,372 positions with positive objective weight,
and 2,118,404 explicit target-suffix positions, out of a 19,668-update,
one-epoch plan (765,345 training windows).

The renderer keeps each window's prompt/context but does not supervise system
prompt positions. For each target-bearing window it predicts the span after
the prefix through the end of the document. `balanced_position_weights`
splits loss mass evenly between pre-suffix context and the decision suffix;
tool context receives role weight 0.25, other roles 1.0. If a window has no
explicit suffix, assistant positions in the span still receive role-weighted
loss. Context-only spans and assistant turns followed by recorded runtime
failure feedback remain available as input context but have zero target weight.

The 48-row monitor is generation-based, not a structural-only heuristic. At a
drift warning it greedily generates all 48 probes. Twelve exact-math probes
are checked against exact text; tool probes are checked for a offered tool name
and required argument shape, without executing the tool. Open-answer and
`natlang_test_turn` probes only check stopping/repetition in the pinned rows;
the latter rows do not require a tool call. Thus this monitor does not execute
full tasks or verify resulting files/values. The held-distribution drift probe
uses 16 rows and compares response KL and greedy agreement to the frozen base.
V18 warned at step 5595 with response KL 0.2826, but its final gate remains
unrecorded because the run was incomplete. The gate-at-warning was 26/48, the
same as the base reference and above its configured failure floor.

## Rendered coverage of the 103 native records

V18's `text_data` input is
`runs/natlang-target-union-20261011-r2/program-header/lfm25/text-data.jsonl`
(SHA-256 `9ab8e89b447a7ba9397f881fe13cdc20197c6ba4f32a9973a4d5b730d698b883`).
The separately pinned `repair-native103.jsonl` (103 records; SHA-256
`a261046…1889e6`) and pieces file (182 rows; SHA-256
`b3a30b…95592f0`) are not appended: `load_text_rows()` takes its `text_data`
branch instead of its records/pieces branch.

All 103 source record IDs nevertheless occur in the union as row IDs and
self-referential `source_record_ids`, preserving the original 87 train / 16
test split. Source-group identity is exact for 96; the other seven are held/test
rows whose union closure records broader groups. This is source provenance
coverage, not by itself proof of target-byte coverage.

I re-rendered the 103 records using their pinned pieces and the program-header
renderer. It produced 61 rows and 42 omissions caused by unresolved or missing
hash-bound piece references. All 61 produced targets matched a union row's
suffix token IDs exactly: 15 rows matched the complete rendered document and
46 matched the exact target suffix while their full document/prefix differed
(the regenerated suffix began four tokens earlier). Therefore 61 target
spans are independently verified; the remaining 42 are represented by source
ID/split metadata but their target spans cannot be verified from the currently
declared piece map. Do not append duplicates for the 61. Resolve the 42 against
their exact piece digests before claiming byte-level target coverage.

The exact 42 omitted row IDs are grouped by their unresolved piece ID in the
pinned `native-pieces.jsonl` (SHA-256
`b3a30bae2767cbeb7fa5dd4ebd7b8bd062f5eb82449c48d13b9de6a6965592f0`):

| Piece ID | Omitted row suffixes (`teacher-program:` prefix) |
| --- | --- |
| `nz1_kslnre6yvlb4ww7qffgqt2p7upek6diefmmk3pwgvm3wtyuxsxla` | `a2e316895ae3f91fa75c:0010–0017` |
| `nz1_qhvg36hduiyqtpesyg2jogazle37ccj2jynh2i4iegqjli4klbna` | `a2e316895ae3f91fa75c:0022,0046`; `ed0d8ee445379a37a9a4:0013,0018,0051`; `9e0adef8c817ecb60d57:0013`; `29cff39239d683d4735f:0027` |
| `nz1_fyvnu2yt4hxe6pvijnvrltnmrqkdld7ec5fbp3gbyhbyk6kqhojq` | `a2e316895ae3f91fa75c:0040,0042–0044` |
| `nz1_fqx2vqyfavn6fcqv4iiei5okxpsl2hjkf7ddlk4yb4np374273ra` | `ed0d8ee445379a37a9a4:0002,0003`; `29cff39239d683d4735f:0002` |
| `nz1_bpph7xgpln6h56zew276hiwaqs4z6gqyotxky4t6cbj2tqjg7hqa` | `ed0d8ee445379a37a9a4:0048,0049` |
| `nz1_ucf5u2nwn2gerrm2mvamgv75ma2jeyyvuqzg4vz6yaphv4i3bbhq` | `9e0adef8c817ecb60d57:0007–0012` |
| `nz1_4a2dvwfgwnm4rz4t7kahz6hhd5ateq3cwbgoxwfvh5cjzjuus76a` | `9e0adef8c817ecb60d57:0039,0040` |
| `nz1_hubsqrxm3gyw6tffxikklobdwuw4hvapdusb4dcf3qwc3fsnw3ya` | `9e0adef8c817ecb60d57:0042` |
| `nz1_i6w4i4g7bluk2depq4jktviilz44cqsf6xkwsiuhrt3p5yubcjrq` | `29cff39239d683d4735f:0013–0016,0020,0022–0024` |
| `nz1_gr2tsqauyalbrivo44ydndmagqimfvrfboi3xxjaklayrkee3d3a` | `20771e90da6ba1487728:0002` |

An omission means no unique same-split, same-source hash-bound body in this
pieces file, not proof that a body is unavailable from all registered
provenance.

The r2 receipt reports 136,191 program-header documents (133,462 train,
2,729 test; 16,032,340 train suffix-supervised tokens) and 76,106
whole-trajectory documents (75,983 train, 123 test; 11,717,655 supervised
tokens). These are different objectives: program-header documents supervise a
selected decision suffix with context loss, while whole-trajectory documents
supervise admitted assistant turns along clean invocation chains. Whole
trajectory data must remain part of the intended training union; it is not
replaced by the program-header view.

## Actual-task evaluation path

The 48 monitor probes do not execute tasks. Existing components support a
realer held-task evaluation without another model wrapper:

1. `train.direct_export.load_direct_artifact()` strictly restores a direct
   checkpoint against its manifest-pinned local base snapshot, exact file
   inventory, and source proof.
2. `python -m natlang_neuralese.serve` now accepts
   `--direct-artifact DIR --direct-base-snapshot DIR`. It passes the restored
   model's existing `Engine` to the ordinary `serve.http.serve()` implementation;
   this is the same HTTP engine and not a second serving stack. It logs the
   artifact manifest hash and checkpoint/status binding at startup.
3. The existing TS teacher collector and interpreter can call that
   OpenAI-compatible endpoint via `run_bonsai_queue.py --chat-completions-url`.
   With a reviewed authored-root-eval queue, the interpreter can execute
   generated tool/native actions and retain exact outputs, task values, and
   file effects for the protected task packet.

The evaluator must keep expected answers in evaluator-only metadata, pin the
held packet and execution code, and record per-case task success separately
from invalid actions, repetition, generation-budget exhaustion, and
infrastructure errors. Do not use the training-source A/B set as held
evaluation. A suitable existing protected packet is
`runs/decision-rich-heldout-20261005-v1/cases.jsonl`: 48 rows with
`split: test` (SHA-256 `d6b3b71f2b885cc273e2f91d299ce832fe95168bffe7c819ec5035806df62925`).
Its authored reference results are separately available at
`reference.results.jsonl` (SHA-256
`04172f1d15b110cf609aec96a663171073f015861a3e0a002df37fb55698e40d`) and must
remain evaluator-only. The registered `clef-rich-source-20261005-v1` entry
classifies these as verified source cases, not training trajectories. A second
disjoint held source packet, `decision-skills-heldout-20261005-v1`, has four
rows; include it only if its task contracts are supported by the same runtime.

I checked the 48 held rows against the registered paired train source file
`runs/decision-rich-train-20261005-v2/cases.jsonl` (72 rows; SHA-256
`6f2a470a8de79d4fb9966fc5a5157ebdf16d3c58a02ae034322a2bdfcde20dca`).
The held packet has four distinct `source_ids`/`source_groups`; none overlap
the 72 train rows. Canonical SHA-256 digests of each task's complete
`semantics.inputs` plus its `services`, `effects`, `folder_files`, and `world`
payload have zero matches in those 72 train rows; `semantics.inputs` alone
also has zero matches. I then streamed each r2 training text file and checked
its `source_record_ids`, `id`, and `source_groups`: all three views have zero
overlap with the held packet's source IDs/groups. The r2 `receipt.json` SHA is
`e0d1ba6bc5793b788859a7f8cfdb0c6f77b51f2e9d994c8f3d687e2a4d333018`; its
pinned LFM text-data SHAs are program-header
`9ab8e89b447a7ba9397f881fe13cdc20197c6ba4f32a9973a4d5b730d698b883`, full
`c240dfad4bf326c42d85fd3448fd9fc3615f8defe8dd3e6ed3fe9cbd7ca7bd2a`, and
whole-trajectory
`2b6cf2e17e146715a10e7fbc954eab72616f3817694229b88be0ad81952bdec5`.

The direct-serving artifact is explicitly `direct-writer-pending`; this entry
point makes its trained backbone available for ordinary text/tool generation,
but does not qualify or enable a native-latent writer. Held-task results must
be scoped to that ordinary text/tool path. The first FP32 run used a 2 GiB
server and later exited after 11 real task results; the remaining 37 rows were
provider connection failures serialized by the collector. The scorer now marks
those rows `infrastructure_error`, so that run is not a baseline. A fresh 3 GiB
FP32 server was `OOMKilled` on the second task after the first partial result.
A 4 GiB FP32 server loaded with 2.436 GiB idle; I stopped it before running
tasks when host `MemAvailable` fell to 1.9 GiB. The pinned CPU host exposes
AVX512 BF16 and the pinned PyTorch reports BF16 CPU support. The matched BF16
plan below keeps the full task context and the existing per-request token
limit; it uses a hard 4 GiB memory-plus-swap cap.

An immutable 48-row evaluation queue is prepared at
`/srv/storage/natlang-artifacts/neuralese-direct-held-task-eval-20261012-v1/`.
Its derived source packet SHA-256 is
`c5d248a8ee6af62f23b4db2730c40af3f9d2d6b0ebd4c011a283f7768bea65d9`, queue
SHA-256 `aa601092539ed7b286db43742d106bd7ba61c82b069d207f5d6d6acae9f0516c`,
and frozen runtime manifest SHA-256
`5b1486b5b473c6bab9c8613fa481c66c486b0ffacaffe2683849d1d7604b9f3e`.
The derived rows only add a SHA-bound `authored_root_eval` contract to each
existing authored root; the expected task values remain evaluator metadata.
The actual-task scorer and checkpoint runner are shared code at commit
`da3b9b1d`, with the pending Pop changes extending it to matched BF16 CPU
serving, cgroup memory telemetry, and a single shared evaluation lock. They
strictly export each candidate, restore it through the same serve engine, run
the exact authored queue, check complete collector accounting, and retain the
highest task-success count only across identical packet/runtime/inference
contracts. Candidate checkpoints and the mutable outer launch receipt are
snapshotted and hash-checked before evaluation. The immutable BF16 comparison
plan is
`/srv/storage/natlang-artifacts/neuralese-direct-held-task-eval-20261012-v1/plan-v2-bfloat16.json`
(SHA-256 `e3237a8e32498e79f01da0175868f266b18873b13efdbdad0f80959000e9eab9`).
Its fixed queue is unchanged. At this audit update, the base run at
`base-bfloat16-v1` has four complete exact results and the fifth task is
running; the first case completed in 60.4 s with all expected effects/status
checks passing and one answer mismatch. Server logs confirm
`model_dtype=torch.bfloat16`; observed server use was about 2.1 GiB and host
`MemAvailable` stayed near 5 GiB. This is interim evidence, not the final held
score. The candidate watcher will use the v2 main outer launch receipt and the
same plan after the base result completes.

## Direct artifact serving change

`training/neuralese/natlang_neuralese/serve/__main__.py` gained the two
explicit direct-artifact flags above. The path rejects a missing base snapshot,
conflicting checkpoint/model overrides, and runtime qualification intended for
recurrence checkpoints. Restoration remains in the strict shared loader. The
existing default and recurrence loading paths remain available.

## Step-4000 actual-task candidate and mixed prompt-source resolution

The v2 fixed held-task evaluation uses the reviewed program-header plan
`/srv/storage/natlang-artifacts/neuralese-direct-held-task-eval-20261012-v2/plan-v4-program-header-v1.json`
(SHA-256 `530db4d3e3994c359dac943f60d2e0284737e1da012886cb12a175ef7dbdb646`).
The matched BF16 base queue completed 48/48: 0 task successes, 21 explicit
budget exhaustions, 27 model-incomplete results, 0 infrastructure errors after
retrying only the two host-interrupted cases (12 and 22). Original raw results
were preserved; the retries used the same reviewed plan and inference contract.
The direct child sanity request reproduced case 23's first response byte-for-byte
(request SHA-256 `65771c1925fa0aa3e2c99b5d35d324e9f7482b222debf2983685ea3520bed20c`,
response SHA-256 `4ce62495f5b8b2369e3500a892703c6f6a3502d33644a11092edf9fdd1ec88ad`),
with stop completion, three generated tokens, and no action. This did not reveal a
nested-request rendering discrepancy.

The step-4000 strict export initially exposed mixed source declarations rather
than a checkpoint or loader fault. Shared exporter commit `058a5269` now requires
an explicit serving policy for mixed text inputs and records the exact selection
without changing checkpoint identity or source rows. The step-4000 checkpoint
SHA-256 is `cbd1d607305579ac25bdc704b701d02c706222a9b64ef81be62d4cc44f8e9d82`;
its text source SHA-256 is
`27e40b8b0531b1c246fb13541301aa10afd4da0f376520ff01f71cdbd33c6353` (136,191
rows). It contains 126,241 explicitly declared program-header rows and 9,950
undeclared rows. The exporter proof preserves effective mode counts and the
undeclared count, and explicitly labels the choice
`explicit-serving-policy-on-mixed-training-input` with serving mode
`program-header`. Of the undeclared rows, 3,843 carry the `List of tools:`
schema marker; this is recorded as a training/serving audit debt, not as evidence
that those rows contain generic runtime prose. The selected step-4000 candidate
is now being evaluated against the same 48-row packet; results are pending.
