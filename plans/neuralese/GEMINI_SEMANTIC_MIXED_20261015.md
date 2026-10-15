# Gemini source-disjoint semantic decision tranche (2026-10-15)

The prepared current local decision catalog contained 3,200 train cases across
AG News, BoolQ, Boardgame-QA and PubMedQA. Exact source/world decontamination
against the prior provider source left 482 new cases: 231 Boardgame-QA and 251
PubMedQA. The two pinned batches have 256 and 226 cases, with no source-group
overlap against the protected held packet. The local catalog source references
carry file SHA-256, physical row coordinates and an explicit unknown-upstream
metadata status. `teacher-decision-labels.mjs` validates these receipts without
claiming a pinned upstream revision.

Gemini batch01 closed 256/256 with 256 typed labels and 141 exact source
matches: 56 Boardgame-QA and 85 PubMedQA. Its static adapter converted all 141
matching labels into 18 authored multi-item Natlang programs and 318 native
turns; all 18 reference replays passed. The remaining 115 provider labels were
not converted. Batch02 is running behind the same project quota lock. Both
batches remain held for semantic/source review and explicit static SFT admission.
No provider-authored program or reasoning trace is claimed.

The existing exact static freeze verifier now accepts source and adapter
directory templates so that the same source/label/replay/turn checks can be
applied to this tranche after batch02 closes. The mixed tranche is a temporary
source-diversity window; the CLINC replenisher timer resumes automatically after
both batches complete. Its one-minute cadence will then keep the older source
available while new source families are prepared.
