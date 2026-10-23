# Native model-line parity

Mellum and LFM share one native production recipe contract. The executable
checker in `natlang_neuralese.train.line_parity` compares the resolved
`native_state_sft` stage settings in `training/neuralese/line-parity.json`.
Every difference needs a concrete, current reason; obsolete direct, recurrence,
projected, and end-to-end recipe stages are not part of this check.

The native source corpus is rendered separately for each model tokenizer. The
registry's immutable `native-state-full-graphs-20261015-v1` entry binds both
renders to the same admitted full-turn source membership and exact train/test
split. The checker permits this model-specific tokenization only while both
content hashes remain in that same registered source declaration. Mellum's
backbone, dtype and expert backend differences are listed separately because
they describe its architecture and runtime.

Run the normal quick checks to verify both production recipes. A new native
training parameter or changed data identity fails parity until it is shared or
has an explicit justified native exception. The full recipe and exact input
bindings remain recorded by each learner's own plan and exposure receipts.
