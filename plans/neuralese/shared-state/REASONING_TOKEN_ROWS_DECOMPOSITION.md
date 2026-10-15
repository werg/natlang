# Existing reasoning-boundary row repair

The training policy remains the owner's declared comprehensive recipe. This
repair is exact token/optimizer/checkpoint plumbing; it introduces no prompt,
new token spelling, tokenizer change or separate training algorithm.

1. Inspect the actual pinned tokenizer for existing `<think>` and `</think>`
   vocabulary entries. Preserve the four distinct Neuralese role IDs.
2. Extend the existing sparse input/readout override mechanism with separately
   named learned rows for those reasoning boundaries. Initialize from the exact
   pinned input/readout rows; an upgrade initially changes no model outputs.
3. Fresh models enable detected rows. Old checkpoints retain their old exact
   architecture and semantic identity unless an explicit declared trainer
   upgrade is requested. Record that upgrade and preserve old tensors, moments,
   FP8/RNG and every exposure credit. Only newly added optimizer groups start
   without history. Reject removals, shape changes or unexplained parameters.
4. Bind new row IDs/content in native identity, checkpoints, portable composite
   artifacts and inference restoration. Semantic and RL use the same model and
   optimizer mechanisms; there is no per-model fork or vocabulary-sized copy.
5. Verify exact old-parent logits and unchanged old moments at upgrade, actual
   learned boundary-row updates for both ports, durable roundtrips and normal
   shared trainer resume. Record actual full-size outcome separately.

The root's QAD proof bounds open-versus-close log-odds by 0.0079262778 for ANY
backbone weights with the frozen final norm/rows. Backbone training cannot
remove that restriction. This repair removes the frozen-target obstruction;
no task-quality improvement is claimed until measured.
