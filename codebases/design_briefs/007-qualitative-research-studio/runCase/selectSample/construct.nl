---
args: { candidate: InquiryFrame, decisions: DecisionsSamplingPlan, evidence: EvidencePage[] }
returns: DraftSamplingPlan
---
1. Construct an evidence-backed SamplingPlan record from established atoms, using the fields declared for SamplingPlan in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Define strata from the inquiry, not from convenient examples, then request exact selection within each stratum. Include negative and atypical cases, record nonresponse, and freeze a sampling manifest before coding the main set.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
