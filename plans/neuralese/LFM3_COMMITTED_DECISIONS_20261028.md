# LFM3: standing decisions (owner, 2026-10-28; revised 2026-11-12)

The 2026-10-28 first-run decisions were made for a "YOLO run with our best foot forward" without experiments. The owner
withdrew that premise on 2026-11-12 (`APPROACH_REVIEW_20261111.md`): training changes are now decided by bounded
controlled experiments on one pinned commit, one change at a time, judged by paired held-out eval (AGENTS.md "How we
train now"). The decisions that set the first run's optimizer (MuonSR, BF16 everywhere), learning rate, data
allocation, reasoning share, shakedown without a before/after comparison, DiLoCo learners and outer optimizer, the
asynchronous NVFP4 actor fleet, JOLT+ as the RL loss from round 1, and three-precision stage monitoring are deleted; git
history keeps them. Each of those questions is now an experiment. The decisions below still stand.

| # | Decision | Fallback, and when |
|---|---|---|
| 5 | No teacher generation (owner). Behaviour gaps come from oracle-built training-split tasks and on-policy RL against exact oracles | None needed |
| 12 | Tuning fits use normal per-matrix Muon (no per-head Muon) | none |
| 13 | Metalearner mixture: the component-token distribution (null token, gain per target group, masked softmax, top-p execution) is the only parametrization; the old grid/head path is deleted | Restore it from git only if fits under the new form cannot reach the old fits' quality and no bug is found |
| 14 | Harness ergonomics changes are measured live, about 48 samples per variant (AGENTS.md "Measure model-facing changes live"); inside a training experiment a model-facing change is that experiment's one change | Bisect a bundle of fixes only on a regression |
| 16 | The Mellum line is retired. Its final exported checkpoint (step 39257) and all data are kept | Revive from git history |
| 17 | The RL actor engine is the SGLang fork (`werg/sglang-natlang`, pin `training/neuralese/sglang-fork.json`), built on Liquid's upstream LFM3 implementation (sgl-project/sglang PR #46858) for dependability of upstream-maintained code as much as speed; the vLLM port is dropped. The llama.cpp fork stays for local/browser serving (owner, 2026-10-28/29) | Keep llama.cpp actors if SGLang shows no benefit |
| 18 | Generic coding data enters supervised mixtures as a small, diverse, verified share in its own lowest tier (about 10% of supervised tokens, owner 2026-10-28) | Change the share through an experiment |
