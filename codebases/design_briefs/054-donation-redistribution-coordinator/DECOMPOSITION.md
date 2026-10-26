# Decomposition: Donation redistribution coordinator

The program owns a case-local, evidence-backed proposal workflow. The root coordinates typed cross-cutting procedures and five domain stages; each domain stage calls four children (`prepare`, `decide`, `construct`, and `audit`), with the audit consuming a finite `qualify` verdict. No child writes to production state. Exact authorization, persistence, effect dispatch, rendering, and external verification remain host responsibilities.

## Domain design

Intent: Match donations to recipient needs and coordinate fair, feasible bundles, substitutions, and transfers under supplied allocation principles.

Invariant: An item description is not verified compatibility, a reservation is not a transfer, and a creative match must preserve essential recipient and handling constraints.

### normalizeDonations

Algorithm: Interpret offers and condition evidence, distinguishing available items from tentative promises. Record quantities, collection constraints, and any permitted-use restrictions; request missing information that changes suitability rather than accepting vague offers into inventory.

Gate: Only confirmed quantities enter allocatable stock.

Contract: ``GoalContract + PolicySelection + Relevant<DonationRound>[] + EvidencePage[] → StageResult<DonationOffer>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### interpretNeeds

Algorithm: Recover recipient needs from descriptions and prior confirmations without inferring sensitive circumstances. Distinguish necessary features from preferences, record accessibility and timing constraints explicitly supplied by recipients, and preserve consent for contact or delivery.

Gate: Suitability criteria are recipient-grounded and not based on stereotypes.

Contract: ``GoalContract + PolicySelection + Relevant<DonationOffer>[] + EvidencePage[] → StageResult<RecipientNeed>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### constructBundles

Algorithm: Assemble useful combinations rather than allocating isolated items that cannot serve the recipient’s purpose. Propose acceptable substitutions, identify missing components, and ensure the bundle’s condition and practical requirements match the confirmed need.

Gate: Bundle is usable under the recipient’s stated constraints.

Contract: ``GoalContract + PolicySelection + Relevant<RecipientNeed>[] + EvidencePage[] → StageResult<DonationBundle>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### applyAllocation

Algorithm: Apply supplied fairness and priority principles to feasible bundles, using exact inventory and transport constraints. Explain tradeoffs and preserve unmet needs without inventing deservingness criteria or optimizing only the easiest deliveries.

Gate: Exact conservation and consistent application of the approved allocation policy.

Contract: ``GoalContract + PolicySelection + Relevant<DonationBundle>[] + EvidencePage[] → StageResult<DonationAllocation>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

### coordinateDelivery

Algorithm: Arrange consented collection or delivery, verify availability before dispatch, and track actual recipient acceptance. Reallocate declined or unavailable items through the same policy rather than informal side agreements, and preserve privacy in donor communications.

Gate: Delivery and acceptance are distinct; recipient details are shared only as needed.

Contract: ``GoalContract + PolicySelection + Relevant<DonationAllocation>[] + EvidencePage[] → StageResult<DonationDelivery>`. Later stages may consume earlier evidence-backed records as declared in the work graph; only authorized, versioned records are visible.`

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
