# Native full-graph training throughput

Policy is a typed natural-language function:
`chooseUsefulWork(actualProgress, admittedShapes, exactMechanisms) -> Continuation`.
First distinguish disabled mechanisms, ineligible graph shapes, and actual
hardware-bound work. Preserve every remaining recipe/data obligation and earned
per-row credit. Do not use repeated qualifications instead of training.

1. `inspectExecution(progress, recipe) -> ExecutionEvidence` reads actual
   checkpoint/runtime/CLI bindings and measured update work. Crisp I/O only.
2. `inspectReuse(admittedRows, batches) -> ReuseEvidence` measures exact token
   prefixes and eligibility on bounded retained TRAIN rows, never held content
   repurposed as training. Reuse existing plans, scheduling and packing.
3. `chooseContinuation(evidence) -> DeclaredOptions` is policy. Enable an
   existing qualified mechanism when supported; extend it only for a concrete
   demonstrated gap. No per-model trainer forks or fabricated throughput.
4. `packExactWork(examples, options) -> PlanAndRowMaps` is crisp identity/mask
   plumbing: preserve each example's original query rows, node/import/state
   semantics, loss normalization, multiplicity and gradients. Communication
   edges never become detached external inputs for physical reuse.
5. `verifyEquivalent(reference, packed) -> Evidence` checks actual outputs,
   whole-graph losses and all relevant gradients on both canonical model ports.
   GPU fit/throughput claims require actual measured qualified hardware; a CPU
   shape audit or tiny derivative check is explicitly narrower evidence.

No cloud signals, new sandbox, sealed-runtime mutation, source/exposure rewrite,
or model-facing text change is authorized by this inspection. Existing trained
actors and quality work retain their owners and resource claims.

## Concrete repeated-readout mechanism

The actual LF18754 ordinary batch already reused 53,748 backbone rows out of
61,844 original tokens. Its old CE path still projected every original loss row
through the vocabulary head independently. Extend that same prefix mechanism:
group only identical `(physical predecessor, actual target token)` pairs, and
sum `original_weight / original_graph_weight_sum / number_of_graphs` for each.
Compute each resulting CE once through the unchanged checkpointed kernel.
Different branch labels remain distinct. Never normalize the combined scalar a
second time. Preserve original per-graph metadata, loss multiplicity and source
credits. No new policy, text rendering, learned state or source identity appears.
Assemble coefficients in fixed original-use order with FP32 segmented reduction,
not CUDA atomic fan-in. Actual BF16 qualification showed that folding moves a
real adjoint cast boundary; retain its original per-use readouts. Only the
qualified FP32 hidden/readout path coalesces terms. This is a shared precision
guard, not an architecture fork or an implicit change to the BF16 profile.

This reuse applies only when exact ordinary-prefix packing actually succeeded
and training requested no per-graph accuracy. Held/accuracy measurements retain
their existing per-graph readouts. Explicit GRPO/DPO objectives return through
their original path before this optimization. Progress reports original and
executed readout-row counts; fewer rows alone are not a throughput measurement.
