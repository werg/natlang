# Gemini CLINC static batches 45–46 (2026-10-15)

The two 256-item source-disjoint batches closed with 394 exact source-matching
typed Gemini labels. The static adapter produced 50 authored multi-item Natlang
programs, 888 native decisions, and 50/50 exact reference replays with visible
child captures. Nonmatching and invalid labels remain outside this source.

I inspected 146 unique, independently shuffled source-matching requests and
intent labels: 30 from each batch and 100 across both, with overlap removed.
All sampled labels fit the visible request and source intent. No clear
mislabeled item was found. The semantic review records the sampling and an
empty item-hold list. `freeze_gemini_clinc_static.py` now accepts an inclusive
start and end batch range and validates the declared sampling scope, so this
tranche uses the same exact freeze/verifier as batches 1–44. The large native
and replay intermediates are compacted to zstd with decoded hashes pinned.

The static source is authorized for SFT as scripted reference trajectories,
not provider-authored reasoning or program traces. Its immutable source
registry ID is `gemini-clinc-static-b45-b46-reviewed-20261015-v1` (manifest
`16b887b4`, verified on DGX). The twin render
`gemini-clinc-static-b45-b46-reviewed-render-20261015-v1` selected all 888
native turns with zero omissions and exact LFM/Mellum token-boundary parity.
It has 888 full and program-header documents and 444 whole-trajectory
documents per tokenizer; its manifest `51170f39` is verified on DGX. The DGX
owner was asked to include the render in the next protected union with global
held/source-quality filtering.
