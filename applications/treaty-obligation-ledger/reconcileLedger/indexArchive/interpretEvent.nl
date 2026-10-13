---
description: Interpret trips, evidence content and evidence arrival without changing event time.
args:
  document: SourceDocument
  clauses: Clause[]
returns: EventInterpretation
---
1. Identify explicit transit events and record tripId,time,reach,operator,destination with exact witnesses.
   A report about an event is not a second transit; an amendment date is not the trip's date.
2. For every relevant factual assertion create EvidenceFact with source-derived factId, proposition,
   affected tripId, eventTime, availableFrom, availabilityPremiseIds and truth. Preserve open rescue claims as null.
3. Treat an affirmative ordinary/non-rescue statement as false rescue status when the source gives
   that statement authority. A missing certification in an open docket remains unknown.
4. For a signed certification retain issuer, named trip and validity proposition in the evidence text.
   Keep its content fact separate from its archive-arrival fact so delta explanations can cite both.
5. Return only document-supported trips and facts. If this document has no events, return empty
   lists and no invented unresolved issue; document category absence is not missing task evidence.
6. The archive caller owns cross-document joins and visibility. Preserve exact source identities so
   the same trip can acquire new evidence without becoming a different business obligation.
