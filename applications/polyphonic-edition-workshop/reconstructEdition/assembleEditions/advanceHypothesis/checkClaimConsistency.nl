---
description: Judge whether one local reading agrees with the edition’s existing editorial assignments.
args:
  hypothesis: EditionHypothesis
  option: ReadingOption
returns: ConsistencyDecision
---
1. Bind priorAssignments to the actual hypothesis.claimAssignments and incomingAssignments to this
   option.claimAssignments. Preserve the original claim and target IDs from both sources of evidence.
2. For each incoming assignment find all actual prior rows with the same claimId. If none exists, it
   is a new source-level decision; if all select the same target, it agrees with the prior edition.
3. If prior and incoming targets differ for the same exclusive claim, return compatible=false with
   both actual option/assignment witnesses and name the conflicting targets. This rejects this extension,
   not the original editorial slip or every other hypothesis using it.
4. Consume option.targetDecisions. Branch assignments require actual established-target or
   finite-exclusive-candidate support; unknown-authority/unresolved-premise cannot authorize an extension.
   Inspect the option's appliedClaimIds. For each applied claim find its actual assignment where the
   source represents an exclusive target. The target must be this option's original eventId. A claim
   assigned to another event cannot be applied here merely because its numerical shift would fit.
5. Read the option reason and correction witnesses to verify the changed field is within the actual
   operation scope. Distinct authorized claims affecting different fields can coexist; equality or
   inequality of resulting pitches alone does not decide editorial compatibility.
6. Preserve a correction with a uniquely named target even when it needs no branching: its assignment
   records that actual source identity. A missing decisive target/operation witness returns null with
   the exact premise; generated option IDs are not evidence replacing that premise.
7. Check this event location has not already been selected in the partial hypothesis. If it has an
   equal actual option, the duplicate extension is a coverage defect for the parent to repair; if its
   selected meaning differs, preserve the source conflict rather than overwrite an earlier choice.
8. For fully supported agreement return compatible=true with the actual merged supporting witnesses
   and a reason naming consistent shared claims or independent operations. Return actual unknowns as
   null. The parent alone copies this result into accepted/rejected/unresolved frontier state.
