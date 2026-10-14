---
description: Judge one original framed landmark without assigning its physical counterpart.
args:
  unit: SurveyReadingUnit
  primaryClauses: SourceClause[]
  supportingClauses: SourceClause[]
returns: LandmarkItemMeaning
---
1. Require this original unit to be a landmark reading. Preserve its sourceFrame and source clause
   identities and exact originalLocalName. Bind the actual literal input as
   {unit,primaryClauses,supportingClauses}; it contains original names/scope rather than prepared geometry.
2. In eval create a scoped NL literal returning LandmarkItemMeaning and directly await it on
   {unit,primaryClauses,supportingClauses}. Copy unit.originalLocalName unchanged into meaning.localId;
   judge frame-owned position, descriptive features and uncertainty
   in this one source scope; it does not decide which reflection or global landing is actual.
3. A stated point belongs to its stated frame. A local origin is local zero, while its globally named
   physical anchor keeps a separate global point. A missing point stays null. Preserve original source
   names and the parent's consistent frame key; never form a hybrid local/global frame label.
4. Distinguish an actual sample niche from a painted imitation, and a reused surface mark from a
   unique physical anchor. Record those descriptions as source claims, leaving same-place judgment
   to the landmark reconciliation stage rather than silently adding a physical equality here.
5. Return one nullable meaning and clause-keyed exact quoted evidence. Every established meaning
   needs at least one nonempty supporting quote; missing decisive identity or frame information is
   unresolved with its actual premise, not a plausible replacement name or coordinate.
6. Consume and return that actual literal result. The parent validates quote bytes and owns original
   source ID/path binding before inserting this meaning into the FrameSurvey consumed downstream.
