# Supervision status — 2026-11-01 morning

Latest observation: 08:54 CET. Check live processes before acting. The goal is
still capable Natlang/Neuralese models, not completion of this document.

## Earned model and numerical repair

Clean SD-R152 completed one real JOLT+ update to global53 from ten authenticated
whole episodes (eight students and two privileged attempts). Actual update:
83.58 seconds,243234 logical forward tokens /78670 packed physical tokens,
12560 supervised tokens, about2910 logical forward tokens/sec. Student OPSD and
teacher KL terms were nonzero; outcome terms were zero. This is actual learning,
not demonstrated capability lift or native-writer readiness. Full source SFT
obligations remain unearned.

Saved full53: `/home/werg/data/lfm3-rl-dgx-20261029/sd-r1/round-000010/learner/checkpoint.pt`,
SHA `1fdbacfb2cfd55cde8d6695bb666d820b41682d5c4d3299872512c267499ff77`,
15330800291 bytes. Controller is collect/round11/trained11. The unit
`lfm3-onpolicy52-amended-jolt-9520-1101` finished normally, before an attempted
operator signal was sent. No stop signal reached it. Its finite state is saved;
its original native numerical certificate failed. Do not treat that certificate
as passed or automatically continue Native SFT on the old arithmetic.

Exact no-SGD replay reproduced three gradient failures: layer0 convolution
in-projection, input embeddings and the rank8 interface adapter up-weight.
Forward/incremental/consumer CE differences were zero. The [2048,8] parameter
is the adapter, not a convolution kernel. Layer-local captures showed the first
difference in FP32 convolution-history adjoint fan-in on imported producer rows;
a2.98e-8 difference could change later BF16 rounding. A resident A/B proved the
invocation-split FP32 gather correction passes the complete53 gradient law with
unchanged forward values and unchanged tolerances. Commit `a9cf65487` implements
that shared correction, retaining one packed model forward. Five focused tests
passed, including rejection of old arithmetic and a disconnected-import control.
A16384-token single-convolution forward/backward timing was0.0809s old /0.0843s
new (three measurements; this is not end-to-end throughput). A real frozen
committed-runtime continuation/certificate still must be verified before Native
SFT. The successful diagnostic used an unstepped optimizer only as an inventory;
it does not replace actual optimizer/recovery custody.

Step52 remains the independently recovered fallback. Branch74 remains diagnostic;
no SFT53–74 exposure from that branch was credited to this new53.

## Actual source and scheduling repairs

The clean52 collection accidentally reused the old materialized ToolACE dataset,
although its converter and1500-case qualified successor were already ready. The
actual task exposed `sequences:any` but expected hidden object keys. A live-read
learner/seed guard stopped that invalid group before any update. Another group
had three successes and one real Native export infrastructure failure, not a
semantic negative. Original20 student/privileged rows are preserved.

Commit `9520ad202` supports authenticated completed-collection withdrawals and
rejects stale ToolACE schema contracts at admission. Actual proof
`ops/toolace-round10-withdrawal-v3/proof.json` SHA
`1e3a0418c1a82c8332be763e7ce347956a57382085e3908727cbaa320cd7be20`
retained only groups0/3 (ten rows), preserving their privileged contexts. The
zero-update transition was actually applied before53 training. Runtime9520 ran
this one round only; its9b actor/Node bindings authenticate historical recording
and must not be used for a new collection. New requests use the corrected runtime.

Withdrawn groups still polluted selector history. Commit `b957280ce` prevents
recurrence; root applied the reviewed exact round10/policy52 cleanup under the
controller lock, retaining old observations in an immutable correction record.
No weights, optimizer, exposure or counters changed. Receipt:
`ops/onpolicy52-current-20261101/selector-cleanup-applied-1101.json`.

The proposed two-graph SFT focus was NOT approved. Semantic review found a bad
qwen15 graph and invalid missing-until targets in qwen36. The combined candidate
at `ops/qwen15-qwen36-successor-v1/receipt.json` (SHA
`28f0e55e97963325ba128e29c97767af9cdf41aed3b63d0df2e29cb1a89ae8d4`)
withdraws qwen15's complete25-node graph and masks only763 confirmed bad action
targets in qwen36, preserving all12 nodes and producer/consumer edge. Current
candidate source is190153 rows, SHA
`b6ec0bbd4026afcf7e25f845b33af6e2e51de004f00f96ff4187e21880a3f7d2`.
CE/family map/geometry/recipe are bound in that receipt; actual `load_recipe`
passed under4GB LOW. It remains unpublished/unadmitted. Remove the stale LogHub
focus role in the actual next SFT controls. Reusable quality tools/tests landed
in `83edad9eb`.

IMPORTANT: original unearned SFT plan is591181 rows (519 leaves), not190153.
Only36158 old ID+binding pairs match the current candidate exactly;555023 old
IDs are absent and153995 candidate IDs are new. This does not establish data
loss: transformation/withdrawal/variant lineage must be audited. Do not silently
mark the old debt fulfilled. `/root/luna_diverse_codebases_a` is running
`sft-obligation-map-remaining-v1` (LOW5GB), reusing/writing source-bound indexes
and preparing exact source-group/lineage/protection evidence. Shared groups alone
are not quota fulfillment. Original checkpoint52 has zero SFT visits; no old
exposure is being erased.

## Structured teacher text now reaches actual Native capture

The teacher-text readout fix is real nested typed model execution. Later fixes
support historical typed-JSON eval-finish writers through separately bound
current conversion code, without relabeling the original Luna/ac519 actor.
Materializer, Python corpus and converter preserve the original sampled producer
eval target; host-encoded JSON is context-only, not fabricated writer-body CE.
Strict source/writer/body/raw-response/protection guards and focused tests passed.

Repeated loader validation exposed three real throughput defects: global hash-log
churn, complete runtime/bound-source rescans, and repeated protected-world loads.
Shared verified caches now invalidate file/dir/symlink and nested protection
changes. A cold audit load reads about1.49GB once; subsequent loads read about
177KB, with no content rehashes. Source pins include transitive custody/hash/IO
helpers. Do not edit those pinned bytes during an active capture without stopping
and rebinding. Earlier audits v7–v11 are historical after necessary fixes.

Current stable audit:
`/mnt/external/natlang-luna-campaigns/tatqa-v16-case0-typed-json-conversion-v12-current/native-conversion/ingestion-audit.json`,
SHA `5ff0957f5bf9a9e5ebbae6b9285af004d7f956789272d6944ac95b1685b563db`.
Actual capture unit `capture-tatqa-case0-native-v12-pop52-1101` uses remote
`pop-native52-capture-json-v2-1101` (10GB host,7.5GB discrete GPU,1GB reserve),
with source encoder checkpoint52/runtime9b. The teacher remains Luna/ac519.
Output: `/mnt/external/natlang-luna-campaigns/tatqa-v16-case0-native-capture52-v2`.
At last check it passed through index17, original eval targets supervised;
actual reader states exist as5.6MB safetensors, source positions[1,37), with
exact writer/typed-body receipts. Sampled `native_writes=0` counters exclude
these separate host encoder bindings; do not infer a missing source from that
counter alone. C must still verify all19 turns, all six import edges and full
Native graph assembly/admission. Wrapper automatically stops the actor/tunnel
and releases the ended lease. Prior v1 failed after a source-helper change;
its partial capture remains diagnostic and its actor/lease were actually closed.

ContractNLI final2 generation was actually launched on the correct source after
rejecting stale final/final4 plans. All three Luna episodes completed but Sol
judged them false; no positive admission. A drain marker left the fourth row
unclaimed so its new requests can use fixed seed/custody/runtime code. TATQA18
corrects the currency task and has three accepted scripted references; its
unsupported-USD negative and fresh teacher plan still need full qualification.
Use actual source/child evidence and repair feedback; do not count broken task
contracts as model negatives. Up to six Luna generation workers are authorized.

## Recovery, storage and active supervision

Step53 independent file export is active as `export-earned53-independent-1101`,
PID589782 at last check. Destination:
`/mnt/external/natlang-independent-recovery-20261101/jolt53/1fdbacfb2cfd55cde8d6695bb666d820b41682d5c4d3299872512c267499ff77/`.
It has copied a few GB, not a verified recovery. Preserve its source until
complete independent subprocess readback. The complete runtime/source53 draft
`ops/recovery-onpolicy53-v2-draft.json` SHA
`5a06d6104f79e626c6e9eda7156c2e5c329a152c5e2a99cb6fe5f35317b74456`
binds730 entries and separately retains future SFT obligations. No strong queue
activation or complete53 restore qualification is claimed.

The supplemental diagnostic69 restore uses NVMe for tiny runtime/Python/assets
and external hardlinks for source objects/checkpoint. Its bounded pause for53
backup ended and the exact invocation thawed. Root's three-converter180-second
window `checkpoint-external-write-window4-1101` must also be checked for actual
thaw. Diagnostic-only pause gave <1MB/s for53; pausing cold converters improved
it to roughly14MB/s, identifying disk contention. Internal NVMe was mostly idle;
the external HDD had about1s read latency. Do not call this an internal SSD fault.

MegaToken runs are deleted on both drives. Disposable recovery staging/caches and
merged worktrees were removed; unique work/recovery refs remain externally.
Internal free space was89GiB, external339GiB before ongoing exports.40GiB internal
floor remains enforced; only52/53 full checkpoints remain in SD-R1 learner slots.
Pop SSH briefly timed out, then was actually reachable again; never infer terminal
failure from one timeout. CoreWeave monitor's latest request was not entitled;
its stale-launch hold remains.

Root supervises operations. Existing helpers: Luna relational application (C)
finishes Native teacher data, Luna diverse_codebases_a audits source obligations,
Terra joint_sft_v4_execution fixes numerical law and recovery. Do not spawn more
strands by default. Targeted tests passed; broad check-main is not fully green.
Continue actual source admission, corrected-runtime qualification and comprehensive
joint SFT/JOLT+, then program adaptation/metalearner when the backbone is suitable.
