# Pin comparison: Pop V8 vs DGX run-v9 (2026-10-11)

Reply to coord 2026-10-11T06:32:11.620960Z-pop-5dde. Facts read from runs/mellum-foundation-qat-20261010/run-v9/{recipe-plan.json,core_text_warmup/plan.json,code-handoffs.jsonl} and git.

## Recipe chain (resolved for core_text_warmup)
raw-recurrence-mellum-v12 <- v11 <- v10 <- v9 <- v8 <- v7 <- v6 <- v5 <- raw-recurrence-v6 <- v5 <- v4 <- v3 <- v2 <- v1 (extends).
Overrides on the Mellum line: v10 latent_step_in_backward; v11 moe_kernel grouped (text and recurrence stages); v12 steps 6144 with quantization_steps 4096 (ramp held at stage-end mixes past 4096) and report_column_fraction 0.25.
Resolved core_text_warmup: steps 6144 (ramp span 4096), context tokens 1024 (prefix 32), batch 2 (eval batch 4), cutoff 4, passes ramp to 3 sequence passes (pass_ramp_evals 2, backbone_ramp_evals 4), eval_every 256, checkpoint_every 2048, held_documents 8 (16 held windows), backbone_training latent, optimizer muon, lr 7.5e-6 (non-latent parameters), sketch_lr 3e-4 (heads), qat_latent_lr 3e-4 (LionSR, units of per-row ternary scale), latent_step_in_backward true, moe_kernel grouped, read_adapter true, input_map kernel 4 rank 64, weights embedding 1.0 sketch 1.0 text 0.25, context_weight 1.0, feedback_weight 0.25, gates max_ce_delta 0.1, max_relative_mse 0.25, min_agreement 0.9, 2 consecutive.
Trainable: backbone 11,696,935,936; heads 10,927,872.
Data: native records (luna-v17 ... assembled-v3/native-records.jsonl + pieces), 4480 documents, windows 5860 train / 3702 test; sampler cohort_weights {native: 1.0}, 2776 native train documents. The harness-bench cohort is declared (mix weight 0.25 of text windows) but admission_granted=false in this stage's sampler: run-v9 core_text_warmup trained native 100%.
Quantization: shared component natlang.neuralese-quantization/1, stage progress [0, 0.35]; points q4 (ramp .15-.30, w .5), ternary-experts (.30-.70, w 1.0, required), ternary (.55-.95, w .5, report). Multi-precision loss sampled per update, BF16 teacher. Column held docs 8, held_rows 48.
Preserve: teacher mellum21-teacher-v3-top64-20261010, weight 0.5, prompt_weight 0.25, max_tokens 4096.
Init: conversion v3 latents (init-gated, else BF16). Fresh optimizer.
AR feedback: not in core_text_warmup (ar_feedback_fixup false); the separate autoregressive_text_fixup stage (1024 steps, 16384 tokens) follows.

## Code identity
Lineage run-v9 froze runtime/ at launch and was handed off 7 times (recipe-plan.json handoffs); after handoff 7 (2026-10-11T05:08) the frozen runtime/natlang_neuralese is byte-identical to the snapshot /home/werg/data/code-snapshots/natlang-f6976b83 (diff -rq clean). Children ran `python -m` from training/neuralese with PYTHONPATH=training/neuralese, so the live snapshot's code was imported (snapshots before it: 8c2a551a, 085713c7, ...).
Python files: 146 under natlang_neuralese at both b3b7dec188 and f6976b83. Import closure of train.text_warmup: 74 modules (artifacts, common.{artifact_paths,hashing,jsonio,paths}, data.{context_refs,text_corpus}, eval.projected_history, maple.{distill_data,family,fused_moe,generation_gate,maple_port,model,nested_train,precision_kernels,qat_convert,student,ternary}, model.{capacity,causal_feedback,dialect,heads,hf_port,input_map,isolated_sequence,layer_staging,lfm2_port,tiny_adapters}, prompt_bank, serve.{batch,chat,engine,grad,guidance,recurrence_checkpoint,store,token_cache}, text_init, train.{adapters,backbone_policy,channel_objective,checkpoint_policy,checkpoint_safety,execution,foundation_schedule,inline_instructions,joint,joint_kd,loop,losses,memory,memory_estimator,memory_policy,optim,optim_restore,output_embedding_projection,projection_anchor,quantization,recipe,record_cohorts,recurrence,staging,text_supervision,text_warmup,trajectories,trajectory_probe,trajectory_state,warmup_export}, view, write).
Diff b3b7dec188..f6976b83 under training/neuralese: 2 commits (d11dfd84, f6976b83), 8 files, +493/-20. Python: 4 files differ (of 146): maple/generation_gate.py (reentrant deployed_cache), train/quantization.py (Behaviour.report fraction, gate_columns points=), train/recipe.py (quantization_steps, stage_extensions, handoff of extensions), train/text_warmup.py (--report-column-fraction, column cost: one deployed cache per column, required columns first, report columns only at stage end). Non-code: recipes raw-recurrence-mellum-v12.json, one luna-ar-native-harness-alignment-refine json, HISTORY.md, scripts/bench_mellum_qat_column.py. None changes the trained update (only precision-column evaluation cost and extension plumbing).
DGX has uncommitted working-tree changes in the shared checkout (optim.py, optim_restore.py, quantization.py, recipe.py, text_warmup.py, input_map.py, memory_policy.py, serve/engine.py, artifacts.py ...) from the current bug hunt; they are NOT in the f6976b83 snapshot and not pinned.

## Shared vs backbone-specific
Shared (same files, same trainer): train.text_warmup, recipe runner, text_supervision, record_cohorts, optim.PortMuonAdamW, optim_restore, memory_*, checkpoint_*, quantization component and gate (train/quantization.py), model.input_map, model.heads, model.layer_staging, serve.grad, data loaders.
Backbone-specific (Mellum, DGX-only in practice): maple/* (model.py dense MoE experts, fused_moe.py grouped GEMMs, precision_kernels.py, ternary.py per-row rule, qat_convert.py, generation_gate.py, family.py nested members), LionSR latent partition (optim.LionSR, latent_partition; used only with backbone_training latent), layer_staging lockstep backward with in-backward latent steps (model/layer_staging.py is shared code, used on the latent path), moe_kernel, row-scale ternary. LFM path uses model/lfm2_port.py + hf_port and the dense AdamW/Muon path; it does not touch maple/*.
Not yet in code: MuonSR (BF16 momentum + stochastic rounding for backbone) is being written; it is not in f6976b83.

## Differences table
| Item | Pop V8 | DGX run-v9 |
|---|---|---|
| Backbone | dense LFM-350M | Mellum2.1 12B-A2.5B (MoE, 11.7B trainable) |
| Package | 146 py @ b3b7dec188 | 146 py @ f6976b83 (4 py files differ, quantization/eval cost only) |
| Recipe | V8 | raw-recurrence-mellum-v12 (chain of 14) |
| Stage | shared core_text_warmup | same, steps 6144 (ramp 4096) |
| Context / batch | 32K / 1 | 1024 tokens (prefix 32) / 2 |
| Data | native 75 / harness 25 | native 100 % (harness declared 0.25, not admitted in this stage) |
| Held docs | 16 | 8 (16 held windows) |
| Passes / feedback | AR feedback 3 passes, fixed gold embeddings and readout | sequence passes ramp to 3; gold text-input map + trained read adapter + heads (trained, not fixed); AR feedback in a later fixup stage |
| Optimizer | Muon/AdamW | Muon/AdamW (lr 7.5e-6, heads 3e-4) + LionSR latents (3e-4 x row scale) |
| Quantization / preserve | none | QAT ladder q4 -> ternary-experts -> ternary; preserve KL to teacher, weight 0.5 |
| Eval / checkpoint | per Pop | eval 256, checkpoint 2048 |

## Follow-up status at 2026-10-11 08:20 UTC

This comparison describes the pinned runs, not all subsequent changes on main.
Both compared runs have stopped; Mellum stopped at its failed step-4096 behavior
check, and Pop V8 completed at selected step 39936. There is no active production
training run represented by this table.

Shared fixes on main now include MuonSR (4d044692), response-position behavior
drift thresholds (aa639a38), and a declared container launcher (38b026a0).
They are not retroactively part of the compared checkpoints. MuonSR remains a
shared selectable latent optimizer, not the optimizer used by Pop V8.

The shared audit in WARMUP_AUDIT_20261011.md found a genuine architectural gap:
preservation and behavior monitoring were initialized only through quantization.
Thus BF16 Pop had no preservation path. Making preservation a stage-level
component in the existing trainer is in progress; the Pop and DGX owners are
coordinating edits. This should not create a separate dense-model trainer.

Model-specific loaders, embedding/control conventions, templates, MoE kernels,
and quantization operations remain necessary shared-repository adapters.
Ordinary evaluation, checkpoint restoration, recipe execution, channel objectives,
preservation, and drift monitoring should use common implementations, with model
adapters providing only the model-specific behavior. Dirty worktrees and new main
commits are not proof that a frozen running package has adopted an implementation.

Pop's selected weights passed independent runtime transport and gradient-replay
qualification, but failed ordinary behavior checks. The earliest retained full
alignment checkpoint (V5, selected step 31744) already has the same math/tool
regression. The final V8 refinement did not introduce the entire regression.
Production recurrence optimization remains paused while the shared preservation
path and applied-update diagnostics are developed. Alignment agreement against
the current ordinary backbone alone is insufficient evidence of preserved task
capability.

### Canonical development checkout adoption

Read-only SSH inspection at 08:19 UTC found DGX canonical HEAD 32262369 while its
origin/main reference was f0fc39ee. Ten training-package files were modified;
some already matched published main content. A comparison of the whole live
package against that origin/main reference showed ten differing files, including
three newer shared evaluation modules absent from the older checkout and pending
changes in generation_gate, engine, recipe and text_warmup. This is an adoption
and publication gap, not ten machine-private implementations. Coord pop-87a6
asks the DGX owner to reconcile the canonical checkout safely before freezing
another run. Pop does not alter DGX dirty work or launch DGX jobs.

The container-launcher hashing ratchet and direct-stage fake-child output-dir
regressions were corrected in b80bb66b42 (AST checks only).

## Superseding owner decision (08:26 UTC)

Coord dgx-8f98 supersedes the preservation follow-up above. Both lines will restart
from base models with one shared identity-initialized multilayer causal-convolution
projection, ordinary token-embedding inputs, and direct target SFT. The input-map,
double-pass and preservation proposal is dropped. DGX owns the shared module and
recipe; Pop will use that implementation for LFM. Exact semantics and fresh
identity/runtime qualification are still pending. The historical comparison and
old failed behavior evidence remain relevant diagnostics, not active run settings.

## Live package cohesion check (09:27 UTC)

Read-only SHA-256 comparison against the canonical DGX working tree, with Pop at
54843abc and DGX HEAD still 32262369, found 149 Python modules on Pop and 147 on
DGX. Of these, 143 were identical; four differed (`serve/__init__.py`,
`serve/chat.py`, `serve/engine.py`, `train/adapters.py`). Two published evaluators
were present only on Pop (`eval/plain_behavior.py`, `eval/plain_weight_drift.py`);
there were no DGX-only Python modules. This is distinct from the historical
142-of-146 comparison above. The shared trainer is identical in this live check;
the remaining differences concern serving/restoration and evaluation adoption.

DGX retains uncommitted work, so its old HEAD alone does not describe its actual
implementation. Coord pop-dc2b asks its owner to reconcile published changes
safely before freezing another run. Pop does not overwrite that working tree.
The new direct-v1 trainer and projection were not yet present in either inspected
working tree; implementation remains assigned to DGX, for both models to consume.

### Updated canonical adoption audit — 2026-10-11 10:37 UTC

After publishing the new direct implementation, Pop has151 Python modules and
DGX canonical147;138 are byte-identical. Nine differ: maple/distill_data.py,
maple/generation_gate.py, serve/__init__.py, serve/chat.py, serve/engine.py,
train/adapters.py, train/quantization.py, train/recipe.py, train/text_supervision.py.
Four published Pop modules are absent from DGX canonical: eval/plain_behavior.py,
eval/plain_weight_drift.py, model/neuralese_projection.py, train/direct_sft.py.
DGX canonical HEAD remains32262369; Pop HEAD at this audit e9b5f9a417.
These are live working-tree hashes, not the active Mellum package inventory.
All four additions are on origin/main, so this is an unresolved canonical
adoption gap. Coord pop-69f6 asks DGX owner to reconcile dirty work safely and
provide the actual active frozen package path/hash inventory. Do not conflate
published shared code with its adoption by a running process.

The new Pop direct lineage V6 adopts the reported shared LionSR tiebreak
(coord dgx-0440), unlike historical V8. Backbone matrices use LionSR3e-6,
norms AdamW7.5e-6; projection3e-4. Native4962 text only,32K capacity,target256,
batch1; 1000steps,eval20/checkpoint100 for early diagnosis. Shared architecture,
ordinary token input, detached projection loss and monitor-only drift follow
neuralese-direct-v1; no input map/sketch/preservation or old weights. V3 aggressive
MuonSR7.5e-6 stopped20 on behavior26/48→18/48 despite improvedCE/value. V4 gentle
MuonSR5e-7 passed guard20 (CE5.795,value delta−.09954,responseKL.01213), was safely
checkpointed40 when the new optimizer decision arrived, and remains evidence.
V5 launch refused a changing live documentation hash before model work; V6
pins immutable lineage-doc snapshots. Old failed/stopped lineages remain held.
