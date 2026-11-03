# Weight-space editing of LFM3 full67 (2026-11-03)

Status: **time-boxed investigation; no edit candidate. The single causal edit test was negative.** The supervisor capped this work at 15:45 CET.
The findings, tools and negative results below are the deliverable. No checkpoint was modified. Nothing here claims
capability lift. The protected dev packet was never opened; its outcomes are known only from the brief.

Parent: full67 `71c3eb45…d9a83b` (`ops/bf95-quality-source-zero-sgd-v2-1102/checkpoint.pt`, 12/16 on dev16).
Comparison: step179 `239c8ca0d4f49a3c4414580c587890fd036c07bf9076df87fbf354ef487ca6dd` (`earned84-…/learner/checkpoint.pt`,
11/16). The SHA was verified while reading it.

Method model: the Exp 8 weight-space-programming post. There, looping traced to one direction written by two MLPs;
ablating it from 64 output columns each fixed looping. A model that did not loop gained nothing.

## 1. Answer to the supervisor's question

**Does full67 have an obstacle that weight editing removes well, or should the method target a looping checkpoint?**

- **Step179's looping is not an editable direction.** The full67→step179 SFT window changed the backbone only at the
  scale of BF16 stochastic rounding (section 2). With the same context, the two models pick the same next action
  token 99.6–99.7% of the time. Case-88 looping arises because tiny, diffuse per-token differences compound over long
  greedy trajectories. There is no low-rank "loop" component in the delta to remove. The only edit that undoes it is
  reverting to full67, which is already the parent.
- **The method is better aimed at a looping checkpoint only if that looping comes from a structured direction**, as in
  the post, where the base model looped on 89% of answers. The practice data shows nothing like that in either
  checkpoint. Exact repeated actions are 5–10% of requests in both, and step179's extra failures look like
  divergence, not a new attractor.
- **full67 does have one concrete, shared, near-deterministic obstacle**: eval code that starts with `import`
  (section 3). It is localized to two attention heads in layer 22 plus late FFNs. It is the closest analogue to the
  post's single derailing decision. Whether removing it improves task outcomes is **not established**: the causal
  test below was negative. The obvious direct-path edit did not remove the habit, so no candidate checkpoint was
  written.
- **SGD and edits.** The backbone learning rate is about 1e-6 (updates at BF16 stochastic-rounding scale), so further
  SGD would barely move an edited direction directly. The `import` habit, however, comes from data (Python/agentic
  tool sessions and the escaped-relay templates). If an edit is adopted, the durable fix is in the data:
  - remove or repair supervised targets whose eval code starts with imports or prose;
  - add recovery targets in which an `invalid-binding` or "failed the same way" observation is followed by a
    different action.

  Re-applying the operator at window boundaries is possible, because the operator is recomputed on the target
  weights (section 5), but it treats the symptom.

## 2. What SFT changed between full67 and step179 (`diff_ckpts.py`)

153 backbone matrices:

| Measure | Value |
|---|---|
| Median relative change ‖Δ‖/‖W‖ | 2.97e-4 (max 4.40e-4) |
| Median share of entries changed | 7.1% (BF16 bit flips) |
| Energy in the top 8 singular values of Δ | median 1.7% (flat spectrum) |
| Top singular value / random-matrix edge rms·(√m+√n) | median 1.18, max 3.24 (`layers.18.self_attn.o_proj`) |
| Embedding/LM head | unchanged |
| Interface: marker input rows | relative change 1.3% |
| Interface: adapter `up` | relative change 43%, but on a near-zero tensor |

Interpretation:

- Each backbone delta is close to a random sparse matrix of last-bit flips, with slight structure. Layer-18
  attention shows the most.
- The 112-step SFT window did not write a direction that could be surgically reverted.
- Cross-scoring confirms this:
  - full67's own 120 practice actions under step179: 99.72% argmax agreement on action tokens; mean summed
    log-prob change −0.12 nats per action.
  - step179's own 117 actions under full67: 99.57% agreement.
  - The largest per-action differences fall on the `import { x } from '...'? // prose` actions (−0.9 to −1.9 nats),
    which both models still produce.

## 3. Shared obstacle: eval code that starts with `import`

**Practice set.** TRAIN-split worlds of the 16 dev shapes, generated fresh:

- `build.mjs --split train --no-hints --seed 51103`, 266 verified cases;
- selection `practice-v1.jsonl`, 68 cases (one per world), sha `ae66ed88…`.

**Observed actions.** Across 4 runs (full67 and step179) there are 264 eval actions with a code start:

| Code starts with | Count | Example |
|---|---|---|
| JS (`const`, …) | 227 | — |
| Python | 21 | `import os, json, shutil / from pathlib import Path` inside `eval` |
| JS import plus prose | 16 | `import { facts } from '...'? Not sure. Let's just call…` |

- The two bad kinds (37 actions, 14% of all eval actions) always fail, with `invalid-binding`, `ReferenceError` or
  `SyntaxError`. They often trigger "failed the same way" repeats.
- They are the biggest single contributor to the repeated failing evals in folder reducers.
- In folder reducers, both models also switch to the `python`/`bash` tools to do semantic judgement with keyword
  matching, where the task asks for inline `nl`.

**The choice is near-deterministic in context.**

| Action kind | log p(`import`) | log p(`const`) |
|---|---|---|
| Python | −0.17 | −5.81 |
| JS import | −0.29 | −1.89 |
| Good JS | −7.29 | −0.31 |

**Localization.** Direct logit attribution of logit(import) − logit(const) at the first code token, bad minus good,
per block write:

- Attention layer 22: +7.56. All 11 attention layers were decomposed per head. Layer 22 head 0 gives +3.63 and
  head 3 gives +3.40; these share a KV group under GQA. Heads 6 (+1.06) and 10 (+0.82) add more.
- FFNs: layer 18 +2.37, layer 19 +2.10, layer 21 +1.34, layer 17 +1.14.
- Attention layer 14: +1.84, mostly head 5 (+1.06).
- Early layers contribute nothing.

**Not simple copying.** 19 of the 37 bad starts have no earlier import-style code in context. Conversely, 67 good
starts had import-style code earlier and did not copy it.

**Causal test.** Greedy regeneration from the exact code-start prefix, inside the native engine:

- Baseline reproduces all 37/37 bad starts and 0/30 imports on a random good sample.
- V1 projects the residual direction g_final ⊙ (W_U[import] − W_U[const]) out of layer-22 heads 0 and 3: 256
  `o_proj` columns, 524,288 weights, using the gain-aware operator.
- V1b applies the same projection to heads 0, 3, 6 and 10.
- Result: no fix and some collateral damage. Details in section 6.

## 4. Negative and process findings

- **The serving path is slow on the shared GB10.** Native greedy decode reached about 35–40 tok/s per process, with
  the GPU 89% busy and other agents' benches running. One 68-case practice pass therefore took many hours.
  - At the dev-protocol 1200 s per-case wall limit, a third of cases timed out, and timeouts are unresolved rather
    than scored.
  - Even at 3600 s, 10 of 14 folder-reducer cases timed out (4 for full67, all 6 started for step179).
  - Practice outcomes are therefore thin. Of the cases that completed with a scored outcome:

    | Family | full67 | step179 |
    |---|---|---|
    | logic entailment | 2/2 | 3/3 |
    | workbench tickets | 2/2 | 1/2 |
    | edit stream | 0/2 | 0/1 |
    | folder criteria reducer | 2/6 | 0/0 (all timed out) |

  - These numbers are not a capability comparison.
- **Two supervisor resource actions.** The work was cut to 2 GPU units, then my units were stopped for the
  throughput benchmarks; the time box followed. Paired before/after practice outcomes for an edit were therefore
  never obtained.
- **Meta-learning probe was prepared but not run.** The Neuralese-reader probe has 48 TRAIN tasks from 8 families
  with `Neuralese<string>` producers, in `neuralese-readers-v1.jsonl`, sha `a4490837…`. The harness's
  `--import-mode native|empty|text` swaps the consumer's exact KV splice for no import, or for text re-encoding.
  - The native arm was stopped under the resource cap before producing results.
  - **No claim** is made about whether consumers use imported state.
  - The weight diff shows the only learned Neuralese operators that moved in SFT: marker input rows (1.3%) and an
    adapter still near zero. That is consistent with the recipe review's finding that the import path is barely
    trained, but it is not a behavioural measurement.

## 5. Tools (re-applicable operators)

- `training/neuralese/natlang_neuralese/model/weight_space_edit.py`: edit spec `natlang.weight-space-edit/1`.
  - Operators:
    - `project_out_columns` (gain-aware, so it is exact under LFM3's sandwich post-norms);
    - `project_out_rows`;
    - `rank_one`.
  - Selections such as top-k columns are recomputed on whichever checkpoint the edit is applied to, so an edit
    re-applies to successors.
  - `write` produces a zero-SGD full candidate. It keeps optimizer, RNG and exposure byte-identical, appends
    `training.weight_space_edit_history`, writes an exact bit patch (`exact-patch.npz`), uses the 40 GiB-floor
    reservation, and makes the file read-only.
  - `verify` rederives the candidate from the parent and checks the exact patch.
- Analysis harness, in session scratch, not committed:
  - practice evaluation through the real native serving path, with an edit applied at decode and token capture of
    every request;
  - a persistent ledger-admitted analysis kernel with exact residual pre-hooks (decomposition error about 1e-3);
  - per-head attention capture through the shared `_linear`;
  - direct logit attribution;
  - in-place reversible edits;
  - greedy regeneration.

## 6. Causal probe result (negative)

Greedy regeneration, 60 tokens from each exact code-start prefix. The 37 bad starts come from both models; the 30
good starts are a random sample.

| Variant | Weights changed | Bad starts still `import` | Good starts identical to baseline | Good starts turned to `import` |
|---|---|---|---|---|
| Baseline (repeat is bit-identical) | 0 | 37/37 | 30/30 | 0/30 |
| V1: L22 heads 0, 3 (direct logit direction removed) | 446,209 | **37/37** | 19/30 | 3/30 |
| V1b: L22 heads 0, 3, 6, 10 | 868,849 | **37/37** | 19/30 | 3/30 |

**Control.** The edit really is live in the native forward: the parameter changed and repeated forwards are exact.
On the first bad case, however, V1b *raised* logit(import) − logit(const) from 2.125 to 2.625.

**Why.** Removing the head's component along the logit direction shrinks the pre-norm output. LFM3's sandwich
`post_operator_norm` then rescales the remaining components up, and downstream layers re-derive the decision.
Direct logit attribution in this model is not a causal handle on the decision.

**Conclusion.** The `import`/Python-in-eval habit is distributed and robust. A small direct-path ablation does not
remove it and does perturb unrelated code. This matches the post's own control: edits remove a concentrated
obstacle, and do not add or reshape a skill.

**Untested next steps, in order of promise:**

1. Ablate a residual *mean-difference* "Python-mode" direction across its top writers (mode-level, as in Exp 8).
   It should be tested in the same regeneration harness first, then in paired practice outcomes.
2. Fix the cause in data. Withdraw or repair supervised targets whose eval code starts with `import` or prose.
   Add error-recovery targets.
3. Use JOLT+ RL on folder-reducer and logic worlds. Exact oracles there already penalise the failed evals.

A weight edit of this habit should not be adopted without paired practice outcomes. The dev16 evaluation decides.

## 7. Artifacts

All under `/tmp/claude-1000/-home-werg-natlang/4c9bd489-afe4-4847-b74c-378caa27432c/scratchpad/wsp/`, which is
session scratch and disposable:

- `practice/`: practice packets and selection orders.
- `out/diff67_179.{json,factors.npz}`: per-tensor diff and top-16 SVD factors.
- `out/prac-*`: practice runs with captured requests (`requests.jsonl.gz`).
- `k1b/`: DLA features (`codefeats.pkl`), per-head DLA (`heads.pkl`) and regenerations (`regen_*.pkl`).

No checkpoint, actor view or ops directory was written, so no rejected candidate needs deletion.
