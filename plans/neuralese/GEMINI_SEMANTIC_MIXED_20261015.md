# Gemini source-disjoint semantic decision tranche (2026-10-15)

The prepared current local decision catalog contained 3,200 train cases across
AG News, BoolQ, Boardgame-QA and PubMedQA. Exact source/world decontamination
against the prior provider source left 482 new cases: 231 Boardgame-QA and 251
PubMedQA. The two pinned batches have 256 and 226 cases, with no source-group
overlap against the protected held packet. The local catalog source references
carry file SHA-256, physical row coordinates and an explicit unknown-upstream
metadata status. `teacher-decision-labels.mjs` validates these receipts without
claiming a pinned upstream revision.

Gemini batch01 closed 256/256 with 141 exact source matches: 56 Boardgame-QA
and 85 PubMedQA. Batch02 closed 226/226 with 142 exact source matches. The
static adapter converted all 283 matching labels into 37 authored multi-item
Natlang programs and 640 native turns; all 37 reference replays passed. The
199 mismatching provider choices were not converted. I inspected 140 unique
source-matching question/answer pairs, 70 from each batch. One PubMedQA item
asks whether biofeedback improves athletic performance, but its visible
abstract reports only psychophysiological control; its whole authored program
is held. The review and exact source/label/replay/turn freeze receipt are in
`gemini-semantic-mixed-review-20261015-v1` and
`gemini-semantic-mixed-freeze-20261015-v1` under the campaign root. No
provider-authored program or reasoning trace is claimed.

The exact static freeze verifier accepts source and adapter directory templates;
it verified both mixed batches and compressed native/replay intermediates. The
source corpus `gemini-semantic-mixed-static-reviewed-20261015-v1` is registered
with immutable manifest `c04c6c59` and verified on DGX. Its training admission
is for scripted static SFT only, with the one held program excluded at render
and global held/source-quality filters still required. The mixed tranche was a
temporary source-diversity window. The CLINC replenisher timer resumed after
batch02 closed and has started batch47 at its one-minute cadence.

The LFM/Mellum twin render is registered as
`gemini-semantic-mixed-static-reviewed-render-20261015-v1` (manifest
`e4966936`) and verified on DGX. It selects 622 native decisions and excludes
all 18 decisions from the one held program, with exact program-header parity.
The DGX protected union still makes the final cross-corpus admission decision.
