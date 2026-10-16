# Newly acquired agent sources

The training policy is to preserve complete useful sessions and their original
actor, task and outcome facts, then reuse the existing SessionHistory producer
and consumer conversion. Source-reported success is not a local execution proof.
Known failed sessions remain explicitly available as failure data; positive
conversion uses the existing failure filter. Missing sessions are counted rather
than terminating conversion of the remaining source.

1. Verify each pinned acquisition receipt and stream its bound Parquet files.
2. Open-SWE and SWE-rebench sessions retain structured calls, observations,
   reasoning, repository/task grouping and reported outcome.
3. Nemotron terminal sessions retain every conversation turn and original
   teacher/run/task identifiers. No execution certificate is invented.
4. Cooper sessions retain each original agent trajectory, sharing the parent
   task group and original communication log. Missing trajectories are explicit
   conversion omissions; the raw source remains available for further views.
5. Existing trajectory normalization, port-record rendering, split protection
   and native graph conversion remain the only implementations of those concepts.

The receipt verifier, bounded source readers and field mappings are crisp
mechanisms: they preserve source facts without selecting a teacher or admission
policy. Admission and full-recipe exposure remain separate operational steps.

The same receipt-bound reader now covers JSONL/gzip/zstd and seven further
structured sources: Toucan, ToolMind, Nemotron-Agentic, UltraData, SWE-ZERO,
CoderForge raw trajectories and Orchard SWE. Source messages/tools may be JSON
strings; decode them without trimming. Preserve actual metadata task/repository
identities, upstream teacher and reported rewards/status. Known failures are
non-positive; no upstream reward is a local execution certificate. CoderForge's
pretokenized view duplicates the raw sessions and is explicitly omitted in favor
of those original messages. Image-bearing Orchard sessions remain explicitly
pending supported multimodal input; do not pretend missing images were observed.
