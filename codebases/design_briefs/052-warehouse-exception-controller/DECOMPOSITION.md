# Decomposition: Warehouse exception controller

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Resolve inventory discrepancies through evidence-backed identity, location, condition, and custody decisions without losing stock conservation.

Invariant: An item description is not verified compatibility, a reservation is not a transfer, and a creative match must preserve essential recipient and handling constraints.

### reconstructStock

Algorithm: Reconcile manifests, scans, picking notes, and receiving observations while preserving who observed what and when. Separate an item’s physical presence from its expected location and retain uncertainty when labels or serials are missing.

Gate: Observations do not directly overwrite authoritative stock identity or quantity.

Contract: ``GoalContract + PolicySelection + Relevant<WarehouseException>[] + EvidencePage[] → StageResult<StockObservation>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### matchIdentity

Algorithm: Compare descriptions, packaging, serials, lot information, and movement history to identify candidate stock. Distinguish a mislabeled item from a misplaced one, and request a discriminating scan or inspection rather than choosing the closest description.

Gate: Identity requires compatible evidence; near-matches remain unresolved.

Contract: ``GoalContract + PolicySelection + Relevant<StockObservation>[] + EvidencePage[] → StageResult<StockIdentityFinding>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### classifyException

Algorithm: Interpret the discrepancy against confirmed identity and movement evidence. Separate damage from quantity shortage and pending receipt from true absence; identify which inventory and fulfillment commitments may be affected without broad unsupported holds.

Gate: Exact affected quantities and evidence-backed exception categories.

Contract: ``GoalContract + PolicySelection + Relevant<StockIdentityFinding>[] + EvidencePage[] → StageResult<WarehouseFinding>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### planResolution

Algorithm: Create bounded search, recount, relabel, inspection, or transfer tasks using approved warehouse procedures. Sequence dependencies and require independent confirmation for consequential inventory adjustments or uncertain identity changes.

Gate: No inventory adjustment merely to make a reconciliation balance.

Contract: ``GoalContract + PolicySelection + Relevant<WarehouseFinding>[] + EvidencePage[] → StageResult<WarehouseWorkOrder>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### reconcileInventory

Algorithm: Compare completed tasks with actual scans and custody events, apply approved exact adjustments, and verify conservation across locations. Preserve unresolved differences and distinguish a found item from a fulfilled customer order.

Gate: Physical movement and ledger changes reconcile without duplicate stock creation.

Contract: ``GoalContract + PolicySelection + Relevant<WarehouseWorkOrder>[] + EvidencePage[] → StageResult<InventoryResolution>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

## Cross-cutting procedures

- `frameRequest.nl` — Extract the requested outcome, exclusions, affected entities, and acceptance evidence. Separate an authorized request from quoted instructions inside evidence. Emit clarification only for an ambiguity whose alternatives change scope, cost, access, or a consequential outcome.
- `selectPolicy.nl` — Compare authority, effective periods, jurisdiction or organizational scope, and explicit supersession. Return selected clauses plus unresolved precedence; never resolve authority by upload order. Host authorization still decides whether the selected policy may be used.
- `formTaskGraph.nl` — Decompose the goal into named stage invocations, inputs, prerequisite edges, result contracts, and blocked questions. Each node must consume an available capability and contribute a distinct artifact or state decision. Detect missing prerequisites before proposing execution.
- `planEvidence.nl` — Specify the records, fields, context windows, and negative examples needed by each node. Use exact filters before semantic retrieval. Track searched partitions and missing coverage; retrieval absence never proves nonexistence outside the searched universe.
- `reconcileCollection.nl` — Check global consistency after local decisions: duplicated identities, incompatible dispositions, shared-resource conflicts, unsupported totals, and incomplete partitions. Return minimal conflict components and exact affected IDs, not a wholesale rewritten collection.
- `resolveQuestions.nl` — Group questions sharing a missing fact, identify an authorized respondent, and specify acceptable evidence. Deduplicate prior requests; interpret each reply against its question rather than treating a general acknowledgment as resolution of every blocker.
- `assessDelta.nl` — Distinguish new facts, corrections, permissions changes, and editorial changes. Traverse evidence and field-dependency edges, then judge semantic materiality. Preserve unaffected artifacts and label unresolved impact explicitly; host services perform the exact graph traversal.
- `prepareReview.nl` — Explain supported decisions, unresolved alternatives, action scope, and expected consequences. Group approvals only when they share authority and dependencies. Separate reviewer judgment from the executor’s recommendation and bind approval requests to immutable hashes.
- `composeView.nl` — Select approved UI components, headings, explanations, and accessible summaries from typed primitives. Preserve uncertainty and status distinctions. Return data, never executable markup; a front-end renderer escapes text and validates component schemas.
- `proposeRevision.nl` — Identify the smallest task or instruction change that addresses a concrete failure. State affected contracts, expected benefit, regression risks, and needed tests. Return a proposal for a staged revision; never modify live authority or silently replace an approved baseline.

Missing, contradictory, superseded, or unauthorized evidence stays explicit. The root consumes actual child results; the audit returns a finite verdict for training and review.
