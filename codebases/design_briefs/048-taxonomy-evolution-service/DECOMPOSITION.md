# Taxonomy evolution service

Membership is tested against the exact current definition and positive and negative examples, preserving overlaps and uncertainty. Repeated boundary failures are separated from inadequate annotation guidance; rare concepts can justify a revision when their decision importance is evidenced. Revision design retains stable IDs for semantic renames and creates new identities where meaning changes, with explicit historical relations and cross-steward boundary conflict checks.

Only records intersecting changed boundaries are replayed under both pinned definition versions. Historical split membership stays coarse when the source cannot reconstruct it. Consumer compatibility checks the actual label cardinality, so single-label systems cannot silently accept a multi-label rule. Promotion remains pending steward approval, migration scope, and dependent cache/version alignment; old definitions remain queryable.

`reconcileTaxonomyPromotion.nl` records the publisher's observed active version only after the definition digest, all invalidated record replays, approvals, and compatible consumer acknowledgements match. Partial or unknown publication leaves the active version indeterminate until the index is reconciled; historical definitions remain intact.
