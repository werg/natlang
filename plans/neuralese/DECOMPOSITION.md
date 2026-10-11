# Native trajectory writer-site extraction

## Scope and review

Owner review for this narrow shared-helper change is recorded in the 2026-10-11 root task: use one pure-data write-site extractor for trajectory preparation and donor selection, preserve inline-sidecar validation and all donor checks, and do not broaden selector eligibility. This change does not alter model prompts, task policy, training admission, or frozen runtime inputs.

## Parts

| Part | Form | Reason |
| --- | --- | --- |
| Parse target JSON writer markers and compiler-attested `neuralese_code` sites | Crisp pure-data helper | Exact paths, JSON schemas, and source spans are provenance facts. The inline validator must accept or reject each sidecar as a whole. |
| Resolve legacy path/type/prefix/argument views for trajectory training and evaluation | Crisp adapters over the shared site | Existing training/render call sites depend on these views; they should not reparse the target independently. |
| Find matching donor candidates and enforce schema/body/group constraints | Crisp verifier in `trajectory_probe` | Matching requires exact structural equality and split/source-group protection. This patch centralizes write facts only and does not infer semantic compatibility. |
| Interpret inline natural-language body meaning or decide task quality | Not part of this helper | Preserve the authored source string; this layer performs no semantic judgment. |

## Shared record contract

A parsed site carries its tool name, top-level argument, exact value path, writer name, raw descriptor, source text, declared value type, source kind, and full/native call arguments. JSON-marker paths include the top-level argument. Sidecar sites bind the exact original `function.arguments` JSON to the validated code hash and compiler sidecar source span. The parsed schema is resolved at that same argument/value path. Existing APIs remain compatibility views over this shared record.

Donor matching still requires a real function-body soft ID, exact schema and value path, distinct writer source text, source-group containment, and reciprocal disjoint factual groups. Missing proof remains ineligible with a specific reason.

## Direct writer with a known token sequence — 2026-10-11

The direct-v1 producer reads gold token embeddings during text-sized training;
its emitted values are read by consumers. Known producer tokens can therefore
be processed as a causal sequence instead of repeating a full backbone call
for each token. This is execution plumbing, not a model policy change.

| Part | Form | Reason |
| --- | --- | --- |
| Select a known complete token sequence versus incremental greedy writing | Exact crisp dispatch in shared `write_direct` | The supplied token count must cover the forced write length; greedy decisions remain sequential. |
| Advance the producer backbone over detached ordinary token embeddings | Shared causal backbone sequence call | Same cache, positional shift and discrete token inputs as incremental writing. |
| Emit values from prefix state plus all but the last advanced state | Shared Neuralese projection | Each value describes the next token; no new projection or private trainer. |
| Record stop scores and final states | Shared token-choice readout | Scores align to post-token states, preserving the incremental writer's diagnostics. |
| Qualify outputs, cache and gradients | GPU mechanism diagnostic against the incremental writer | Sequence and decode kernels may differ numerically; measure before adopting the execution change. |
