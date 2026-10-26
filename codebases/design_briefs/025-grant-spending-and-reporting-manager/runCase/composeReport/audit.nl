---
args: { draft: DraftGrantNarrative, candidate: ActivityEvidence, evidence: EvidencePage[], policy: PolicySelection }
returns: AuditGrantNarrative
---
1. Compare every populated field to the exact cited source span, authority, date, and scope. Check that no relevant conflicting evidence was omitted.
2. Test a plausible counterinterpretation against the same source set.
3. Obtain a finite gate verdict by calling qualify(draft, candidate, evidence, policy).
4. Return the child's verdict with concrete evidence findings and the smallest repair target. Passing does not authorize external commitment.
