---
description: Rank program promises by supplied authority and downstream dependency.
args:{event:EventProductionState;sourceRecord:LiveEvent;sourceEvidence:EvidenceItem[]}
returns:PromiseLedgerItem[]
---
Bind the event identity and promise set to the LiveEvent top-level eventId, agendaIds, venueId, producerId and commitmentIds. Each public, contractual and accessibility promise must match one of these commitment IDs and current evidence. A generated event template cannot replace the source event, its venue or responsible producer.

Preserve contract, public and accessibility promises with their source references. Internal preferences do not outrank an external promise. Bind dependencies and owner roles; never drop a commitment because its resource is unavailable.
