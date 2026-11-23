---
args: { draft: DraftIncidentCluster, candidate: SupplierIncident, evidence: EvidencePage[], policy: PolicySelection }
returns: GateVerdict
readout: decision
---
1. Apply the application acceptance gate to the candidate (SupplierIncident): determine whether the cluster membership has a specific shared factor and preserves uncertain links, weighing the draft (DraftIncidentCluster), the evidence (EvidencePage[]), and the policy (PolicySelection).

2. Return 'pass' only if source evidence establishes each required condition for the candidate (SupplierIncident), with the applicable authority, effective period, and scope, using the evidence (EvidencePage[]) and policy (PolicySelection).

3. Return 'fail' if decisive evidence establishes that a required condition is false or the draft (DraftIncidentCluster) contradicts an authoritative in-scope record, considering the evidence (EvidencePage[]) and policy (PolicySelection).

4. Return 'needs-review' if required evidence is missing, contradictory, out of date, outside scope, or lacks established authority (policy (PolicySelection)); do not treat unsearched material as negative evidence, and evaluate the evidence (EvidencePage[]) and draft (DraftIncidentCluster) against the candidate (SupplierIncident).
