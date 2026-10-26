---
args:{prior:TenderCheckpoint|null;request:BidPreparationRequest;room:TenderSourceRoom;obligations:TenderRequirement[];sections:ResponseSection[];bundle:TenderBundleProposal|null;timeout:TenderGuardrail}
returns:TenderCheckpoint
---
Bind history to this tender and its ordered addendum edition. Preserve prior response sections and exact hashes. If a changed addendum affects a requirement, mark only that section hash stale and invalidate its approval binding. Retain portal submission and readback receipts by exact bundle hash; an unknown timeout is not a failure receipt and cannot be retried until readback establishes the state. Append each new receipt once and treat identical response/addendum evidence as a no-op.
