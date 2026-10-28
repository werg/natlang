# JOLT+ privileged context and whole-graph teacher scoring

Shared by the whole-graph RL learner (OPSD term, J_T teacher term) and the
teacher-free program-tuning supply. Source: `TRAINING_INSIGHTS_20261028.md` §7,
`LFM3_RECIPE_20261028.md` "RL default: JOLT+".

| Part | Form | Why |
| --- | --- | --- |
| Protected world custody (`ProtectedWorlds`) | crisp verifier | Exact identity check (ids, source ids, source/split/pair groups, world fingerprint) against every registered `evaluation-only-never-training` packet; fails closed when a packet is unreadable. Only `split == 'train'` worlds teach. |
| Fact extraction (`case_facts`) | crisp helper | Mechanical reading of authored reference root/children, oracle data (expected result, stop, files, effects, reference failures) as sentences, minimum sequence, decisive evidence, verified same-world teacher trajectories, earlier-attempt runtime feedback. |
| Applicability (`applicable`) | crisp mechanism | A child sees only its own matched reference value; the root sees world-level facts; attempt feedback may be bound to one invocation. Prevents a policy from routing another call's answer. |
| Fact choice and order (`selectPrivilegedFacts`) | policy: own NL function + crisp default | `applications/curriculum-advisor/selectPrivilegedFacts.nl`, run by an optional command with one JSON request on stdin (as `select_tasks`). The crisp default orders by kind. Output is validated to distinct applicable ids. |
| Budget packing | crisp mechanism | Whole facts in policy order while the rendered turn fits `budget_tokens`; no mid-fact truncation; empty block inserts nothing. |
| Rendering (`PrivilegedRendering`) | crisp, hash-bound declaration | One ChatML turn (`role`, `header`) per backbone template (BOS/im_start/im_end + the template's content tokenizer, bound by tokenizer fingerprint), placed after the leading system turn, else after BOS. |
| Receipt (`PrivilegedBlock.receipt`) | crisp durability | Binds rendering hash, policy implementation hash, custody receipt, fact ids/hashes, omissions, insertion index and exact token-ID hash. `verify_block` rechecks. |
| Actor insertion (`insert_into_prompt_segments`) | crisp plumbing | The sampler of teacher attempts inserts the same block at the same own-token index. |
| Graph edit (`teacher_form`, `student_form`) | crisp mechanism | Rebases token IDs, producer spans, import boundaries and cross-invocation import spans (physical and logical) of the actual compiled `NativeReplay`; `compile_graph` assigns shifted consumer positions. Refuses edits that cross spans or share an import boundary. |
| Teacher scoring (`teacher_scores`) | crisp mechanism | Stop-gradient one-pass forward of the edited whole graph; per-action fp32 raw logits (or top-k processed logits) plus processed log-normalizer and sampled-token log-prob, aligned 1:1 with student actions. Optional check against `forward_reference` (explicit cached execution). |
| Teacher role (`teacher_role_logps`) | crisp mechanism | Log-probs with gradients for attempts sampled with the blocks, plus their exact student-form replay for the beta_T KL. |

The losses (score-centered outcome terms, OPSD reverse KL, J_T) belong to the
RL learner; this unit supplies their inputs only.

Open: semantic-codec graphs (`semantic_dependencies`) are refused until the
codec executor rebases its source spans; LFM3 has no native `SharedStateModel`
port yet, so tests cover LFM2 and Mellum; the privileged turn text and the NL
policy are unmeasured model-facing debt (`plans/MODEL_FACING_CHANGES.md`).
