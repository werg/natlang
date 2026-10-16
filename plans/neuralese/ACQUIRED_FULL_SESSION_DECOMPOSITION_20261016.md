# Use complete acquired sessions in the shared native recipe

The existing trajectory adapters own source normalization and factual provenance.
`prepareAcquiredSession` applies the existing source protection policy: failed,
quality-held or pending sessions are not positive SFT; evaluation groups stay held.
This is static source-curated third-party supervision, never local execution or
observed actor-state exchange. Raw originals and exact export receipts stay bound.

Crisp plumbing streams each hash-bound full-session export through the existing
serving renderer and native text graph converters. Keep the complete conversation,
all tool feedback and all assistant turns, with existing role loss weights. Do not
use the old four-window or character limits. Graphs larger than current memory
capacity remain complete and pending capacity qualification, never truncated.

Earlier reasoning follows each backbone's serving template. Supply an additional
full-prefix, single-target view for every reasoned assistant turn, so a
backbone dropping history reasoning still receives its complete target when that
turn is current, including conversations ending in observations or unanswered
user messages. These are declared auxiliary views, not inferred communication
edges. Only actual producer/consumer graphs count toward communication readiness.

The builder records every source session as graphs or an explicit omission, joins
current protected source groups and decision holds, and binds both tokenizer and
source receipt. Existing corpus registration/whole-union split validation and
resident continuation own final admission and actual training exposure. Conversion
alone does not grant admission.

The pinned source-ref to complete-session adapter lookup is shared in
`acquired_agents.FULL_TRAJECTORY_CORPORA`; the exporter resolves that lookup
through the existing `agents.CORPORA` registry. The lookup is crisp plumbing
because a ref must select exactly one already-authored adapter. Session
normalization and source-quality interpretation remain in the shared adapters.

## Observed numeric settings and shared compressed input

D12 source qualification isolated a thread-sensitive FP32 reduction in the
existing segmented role weights. Tokens, masks, provenance and all non-weight
fields were exact; replaying the captured calculation at20 threads reproduced
the original entire vector exactly. The historical worker environment was not
recorded and remains unknown. No data was rewritten and no tolerance changed.

Future static-text and complete-session receipts record the actual Torch/Python
versions, Torch thread counts, default dtype and OMP/MKL environment. Each
complete-session shard records its own observed settings, including after a
resume. These observations do not constrain continuation or reset row credits.
Source sample qualification uses the selected shard's recorded Torch threads,
restores the caller's setting even after failure, and distinguishes an explicit
legacy reference override from known historical metadata.

The static text converter also reuses common.jsonio's existing compressed
reader, including symlink resolution and long-window handling. A file symlink
previously skipped by the direct zstd wrapper must never produce empty success.
Focused source-reader, numeric-reference, full-session and preference tests
passed27 cases; the integrated offline-source/optimizer suite passed34 cases.
