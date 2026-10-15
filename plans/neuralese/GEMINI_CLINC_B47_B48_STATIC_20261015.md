# Gemini CLINC static batches 47–48 (2026-10-15)

Two 256-item source-disjoint batches closed with 440 typed labels that exactly
match pinned CLINC train labels. The shared static adapter authored 56 multi-item
Natlang programs, 992 pending native decisions, and 56/56 exact reference
replays with visible child captures. Source-mismatching and invalid typed
outputs were not converted.

I inspected 134 distinct independently shuffled source-matching request/intent
pairs, including 30 from each batch. One request asks how to *hurt* a credit
score while the pinned label says `improve_credit_score`; its whole 18-decision
program is held. Other sampled labels fit the visible request or are harmlessly
underspecified. The independent semantic decision is bound in
`gemini-clinc-static-v2-review-b47-b48-20261015-v1/semantic-review.json`;
`gemini-clinc-static-v2-freeze-b47-b48-20261015-v1/freeze-receipt.json`
checks source, labels, authored programs, replay and native turns and compacts
large intermediates to zstd with decoded hashes.

The source corpus `gemini-clinc-static-b47-b48-reviewed-20261015-v1` (manifest
`9ac19310`) and twin render
`gemini-clinc-static-b47-b48-reviewed-render-20261015-v1` (manifest
`0ea15c8d`) are registered and verified on DGX. The render selects 974
approved static turns and excludes the held program's 18 turns. All program
header strips passed exact serving-boundary parity. This is scripted static
SFT, not provider-authored program/reasoning data; the DGX protected union
still makes the final cross-corpus admission decision.

The batch47 static conversion first hit its 0.7 GB memory claim and left
partial output. `b9478a3d` raises the claim to 1.3 GB and lets the replenisher
list and prune only an OOM-failed partial output before relaunching. The rerun
completed with 29/29 exact reference replays; batch48 completed with 27/27.

A later pass found two source-matching `sync_device` requests that actually ask
to *unsync* a phone. Their direction remains ambiguous under the intent label,
so both source items and the previously excluded inverse credit-score item are
pending in `source-reviews-20261015-v2`. `67675804` lets source reviews bind
their opaque rendered group IDs. The registered render remains immutable;
the protected union must omit both additional programs. Its program-header
view contains 36 affected documents. DGX was asked to apply the updated holds
before publishing its next union.
