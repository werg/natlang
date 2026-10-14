---
description: Construct supported reading options from authority, dependency and explicit ambiguity.
args:
  readings: WrittenEvent[]
  claims: EditorialClaim[]
  targets: TargetDecision[]
  families: FamilyIndex
  protocol: NotationProtocol
returns: ReadingBundle
---
1. Bind originalEventId from the actual readings and verify shared location identity. Inventory every
   distinct diplomatic symbol tuple and the source families supporting it; preserve original reading IDs.
2. Read protocol.editorialPolicy to establish which source governs an uncorrected location. Use actual
   source-family evidence to distinguish a repeated inherited reading from an independent alternative.
   A family with unresolved scope cannot count as an established independent endorsement.
3. Initialize baseOptions from supported underlying readings with their actual event fields and
   witnesses. Keep rejected reading IDs with concrete authority/dependence reasons; rejection concerns
   the reading's use in the edition, not erasure of its documentary existence.
4. Visit actual claims in their source-established precedence. For each join its TargetDecision for
   this location. basis=excluded leaves baseOptions unchanged with its actual reason. basis=unknown-authority or
   unresolved-premise retains an unresolved reading obligation; those states cannot create finite options.
5. For authorized basis=established-target with applicable=true, interpret the stated operation. A duration operation replaces only
   durationGlyph; a pitch-shift changes only written rank by the supplied shift. Copy voice, section,
   order, kind and accidental unchanged unless separate original authority explicitly governs them.
6. For actual basis=finite-exclusive-candidate under authenticated exclusive-target ambiguity, fork each base option once per supported
   original target assignment. A branch assigned here applies the operation; a branch assigned elsewhere
   retains the local event. Both branches carry that same claimId→targetEventId assignment and the actual TargetDecision in
   targetDecisions, including its typed basis and exact evidence.
7. If an earlier option already contains this claim assignment, require agreement before extension.
   A second mutually exclusive target is a rejected extension, not another correction added to this note.
   Preserve the reason so later global assembly can expose the same source conflict consistently.
8. If multiple authorized claims change one field, read explicit supersession/scope policy. Apply an
   established precedence, preserve compatible operations or retain unresolved alternatives. Neither
   document date nor traversal order silently supplies missing editorial authority.
9. Compare any copyist expansion with the actual claim scope. A tentative expansion choosing one
   ambiguous target is derivative interpretation; it cannot eliminate the other supported assignment.
   An unsigned fingering aid remains advisory even if its proposed pitch fits an ensemble constraint.
10. Deduplicate only options equal in event fields and source-level claim assignments, combining exact
    supporting witnesses. Numerically equal duration totals do not equate different pitch/target readings.
11. Retain all actual governing targetDecisions on each option, including established/excluded decisions.
    Return original eventId, actual options, rejectedReadingIds, reason and witnesses. Empty supported
    options retain a concrete source obstacle. Global timing/sonority checks will consume these options;
    this stage must not preselect their expected successful timeline.
