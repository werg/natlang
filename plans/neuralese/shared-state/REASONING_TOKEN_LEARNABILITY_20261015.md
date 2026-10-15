# Frozen reasoning-token rows prevent reliable delimiter learning

The LFM QAD native training lineage needs learned input and output rows for its
existing thinking delimiters. Do not remove teacher reasoning or discard the
useful backbone training. Continue from the latest exact parent after an explicit,
qualified interface and optimizer extension. Apply the shared mechanism to both
students; keep each tokenizer's actual token identities and initialization.

## Evidence

The exact QAD inventory is
`f40d7a5bf651a03618ae8c06bda24a180e22fa79de964e55862c2c4984249da6`.
Its `<think>` and `</think>` tokens are IDs 64400 and 64401. Their output rows have
cosine similarity 1.0 and maximum element difference 0.00006103515625. The current
full-layer training policy freezes both vocabulary/readout and final RMS norm;
only the separate four Neuralese role rows have learned vocabulary overrides.
The teacher's exact positive targets require these thinking tokens.

This is a fixed learnability limit, independent of the backbone weights. For
output rows `a` and `b`, final RMS norm weights `w`, and hidden width `d`,
Cauchy–Schwarz bounds the absolute pair log-odds by
`sqrt(d) * norm(w * (a - b))`. The exact QAD tensors give
**0.007926277764878096**. Either delimiter's probability is therefore at most
**0.5019815590667979**, even if every other vocabulary token is suppressed.
The source LFM RMS normalization multiplies directly by `w`, without an offset.
More backbone updates cannot repair the frozen output distinction.

Actual checkpoint 2000 (SHA
`40191820e14014522dccb4ad15a54b63b08a7fe423f15790083a326c59286be3`)
has high first-reply entropy on four held natlang contexts: approximately 7.69–7.72,
with `<|fim_mid|>` the top token at probabilities around 0.0008. On three ordinary
coding contexts, first-token probabilities remain useful and entropy is 1.50–2.64.
New positive Qwen examples expecting `<think>` also expose the problematic reply
boundary. Header-token predictions immediately before it remain confident.

## Alternatives checked

- Incremental inference and a flat graph forward agree on the first held prompt
  within 0.00009 in logits, with the same top token.
- The independent stock HF implementation over the restored checkpoint agrees
  on ordinary vocabulary logits within 0.000081 and has the same first argmax.
  The four learned Neuralese output overrides are compared separately from the
  stock head. Its first probability is also approximately 0.00085.
- The final normalization weights are exactly unchanged from QAD; hidden and
  normalized hidden magnitudes have not collapsed. The HF readout remains tied.
- The decoded 53,223-row old full source and the 4,691-turn raw Qwen V4 source
  contain no `<|fim_mid|>` in either targets or contexts.
- An independent optimizer audit reports exact full gradients, model weights and
  optimizer state after two updates for both tiny ports, checkpointing on/off,
  and multiple readout chunks. Its RFT/GRPO checks also agree exactly. These
  focused CPU checks do not qualify all production CUDA kernels.

The fixed thinking-row limit is established. Its complete contribution to every
held task failure is not yet established; continue examining actual actions and
task quality after repair.

## Repair and acceptance

Independent review also reproduced a precision defect at the production learning
rate, 0.00001: after 16 nonzero-gradient updates on the tiny BF16 Mellum path,
none of 128 new output-row elements moved, and only one new input-row element
moved. The existing BF16 role rows also lost most small updates. The sparse-row
repair therefore includes FP32 master parameters for both reasoning and
Neuralese role rows, with native activation casts. An explicit upgrade preserves
old values and moment values exactly by upcasting, retains counters and RNG, and
records the dtype transition. Legacy loading must preserve the saved contract
until that upgrade. The independent review also requires rejecting malformed
row shapes before mutation; broadcastable tensors are not valid checkpoints.

Extend the existing sparse learned input/readout override mechanism. Keep the
four Neuralese role IDs and their existing parameter tensors unchanged. Preserve
old checkpoint behavior exactly until an explicit upgrade; initialize new rows
from the exact restored vocabulary, and preserve all existing optimizer tensors,
RNG, valid-row credits and continuation frames. Only new parameters start with
fresh optimizer state. Include the learned rows in checkpoint/composite restore,
serving, shared semantic training, RL, and native embedding identity.

Before cloud continuation, verify exact old-parent logits at migration, exact
old optimizer preservation, actual learning of distinct delimiters, save/resume,
and both backbone paths. Measure the resulting task behavior on held data; never
admit this diagnostic's held contexts or outputs into training.

The actual LF2000 CUDA qualification of the first sparse-row repair passed:
149 existing optimizer parameter histories, all existing parameter values,
RNG, input embeddings and initial full-vocabulary logits were preserved.
Sixteen real production-optimizer updates on one admitted Qwen V5 training graph
reduced its loss from 0.625005 to 0.268292. The learned row pair's log-odds capacity
bound increased from 0.007926 to 0.043462; full checkpoint and updated optimizer
reload were exact. This confirms the mechanism, not general task quality.
The row's final first-delimiter probability was still only 0.000549, so continue
measuring actual learning and task behavior rather than declaring the model
capable from the small diagnostic.

The initial diagnostic incorrectly required the old fixed hidden vector's
margin to reach the theoretical maximum within four, then sixteen updates.
That condition did not measure the fixed-capacity defect. The corrected test
checks the actual learned row-pair capacity, nonzero parameter updates, loss,
and exact reload; the fixed-vector margin remains telemetry. Numerical parity
tolerances were not relaxed.

Private evidence and ledger logs live under
`/home/werg/data/dgxops/lfm-native-task2000-held-diagnosis-20261015/`:
`think-token-learnability-bound.json`, `think-token-embedding-audit.json`,
`readout-weight-hidden-diagnosis.json`, and
`prompt-boundary-reference-diagnosis.json`. The source scripts and failed initial
probe attempts are retained there for this active investigation; failed probe
setup is not a production model failure.
