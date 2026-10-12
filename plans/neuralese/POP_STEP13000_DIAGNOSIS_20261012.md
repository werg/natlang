# Pop direct Muon step 13,000: held behaviour and source response (2026-10-12)

The Pop LFM2.5-350M direct Muon lineage reached a full-state checkpoint at update
13,000. Held next-token CE is 1.505 versus 2.598 for the untrained base on the
same held documents (step 12,000: 1.574). The task cadence uses the disjoint
64-case held packet v3, sampling one case from each of its 16 shapes. Its first
step-13,000 pass solved 0/16: four completed with wrong exact answers, eleven
exhausted the 48-request collection allowance, and one exhausted the 600-second
case allowance after repeated long completions. The required repeat also
solved 0/16: the cadence recorded `eligible=true`, `repeat_noise=0`,
`regressed=false`, with 12 budget exhaustions, three semantic mismatches, one
invalid action, and zero infrastructure errors. The trainer resumed productive
updates after the evaluation (step 13,024 observed). These are collection limits, not a language deadline or
an automatic conclusion that the model cannot learn the task.

The failure traces show distinct problems. One folder case chose `bash` where
the current call expected a folder action; other cases repeated `eval`,
`world.receipt()`, or file reads after error or observation text, and one
generated near-8192-token repetitive text. The model request hashes changed
between turns and the prompt grew, so this is not simply a replay of one cached
request. The frozen program-header server drops native tool schemas, as does
the matching program-header training renderer; a separate serving fix on main
retains *external* tool schemas for pi. Test any change to native schema
visibility live before adopting it. The base reference for held v3 is 0/16
under full rendering, so compare the student with that exact rendering and
the larger LFM reference before attributing failure to optimization.

The active r3 program-header union contains supervised `bash` targets in
native-sft-v13-r2 (1,846 turns) and harness-bench (3,769 turns), among other
sources. The inspected native row originally offered `bash` in a directory
reducer; the inspected harness rows carry their own tool instructions. These
are genuine source actions, not yet proof of a corrupt conversion. Audit
tool-availability context by task surface before changing admission. Several
held shapes are thin in the sampled training union: `logic_proof_verifier` 41
turns, `relational_late_argmax` 60, `contract_diagnosis` 70; source-group turn
counts overstate independent cases. The 350M base also never called `eval` in
its held-v3 reference, so learning tool use is progress even when exact task
success has not yet moved.

To collect targeted evidence and training candidates, Pop built 318 statically
verified **train-split** cases with fresh seed 52005 across eight shapes:
`contract_diagnosis`, `workbench_tickets`, `edit_stream`,
`logic_proof_verifier`, `actor_route_replan`, `folder_criteria_reducer`,
`iterate_schedule_repair`, and `relational_late_argmax`. The immutable source
at `/srv/storage/natlang-artifacts/free-provider-generation-20261012-v1/provider-luna-held-shape-remediation-v1/source/held-shapes-s52005-n64.ir.jsonl`
has SHA-256 `faeb0963e6c0678558543b2bae3575db5a4a63e531eb1f5fc9c4627a05014f7f`.
A 32-case Luna pilot (four per family) and two additional 16-case lanes
(two more per family in each lane) are running in the same campaign folder;
the manifest records their selected indices and queue hashes. The first six
reviewed cases were exact accepted. This is **not training
admission**: inspect exact results, semantic fidelity, failure clusters, and
source-group deduplication before promoting any of its rows. The held seed
61012 and its evaluation-only packet remain excluded from training.

Next: complete the repeat, report the eligible cadence result, inspect failure
traces against source/oracle and rendering, review the pilot, and add only
quality-approved, source-disjoint examples through the shared corpus registry
and continuation recipe. A fresh start uses the pinned QAD base under the
separate owner decision; this checkpoint continuation keeps its existing
weights and optimizer.
