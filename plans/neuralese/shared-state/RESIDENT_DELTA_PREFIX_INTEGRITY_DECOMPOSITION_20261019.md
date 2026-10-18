# Resident delta prefix integrity

## Natlang-native contract

The resident continuation has a natural-language decision: keep every row
already in the learner's ordered train prefix, admit only source rows that are
new to that prefix, and retain held and pending protection. Exact code is used
only to authenticate indexed row identities, compare ordered source metadata,
and atomically commit the request pointer.

## Stages

1. Bind the current source plan, prior train and held manifests, explicit delta,
   recipe, and runtime using their existing immutable references.
2. Load the prior train and held metadata only through the exact warm graph
   indexes already keyed by source SHA, parser policy, tokenizer, split, model,
   and role IDs. A missing or mismatched index fails closed; this preflight does
   not fall back to parsing the prior compressed graph source.
3. Load and authenticate the delta. Compare each delta ID and graph binding to
   the full prior train map. Reject an exact repeated identity separately from
   a same-ID changed binding before computing geometry or publishing. Also
   reject every ID already present in held, even if a malformed row claims
   disjoint source groups.
4. Assemble the proposed full train manifest from those loaded prior and delta
   sources. Preserve source order, verify the ordered IDs/bindings, assert train
   and held groups are disjoint, and run the same pending/withdrawal obligations
   used by the learner. The delta count and witness list come from this checked
   union, never from blind arithmetic.
5. When the learner polls, perform the existing full-source validation. At the
   single commit lock, re-read the pending pointer and require it to still name
   the exact request being adopted, then require the current generation to
   remain the exact predecessor. A withdrawn or replaced request cannot commit.

## Verification

Fixtures cover a new row, exact duplicate ID/binding, conflicting same-ID
binding, held-ID and train/held-group overlap, absent prior index, cache removal
between preflight and use, and withdrawn or replaced requests after validation
but before the commit lock. Race tests prove no generation is committed and no
exposure is credited. Existing source bytes and optimizer/RNG/exposure state
remain outside the metadata transaction.
