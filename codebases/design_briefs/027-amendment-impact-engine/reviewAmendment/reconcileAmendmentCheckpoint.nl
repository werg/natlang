---
args: { prior: AmendmentCheckpoint|null, request: AmendmentReviewRequest, executed: ExecutedAgreementBundle, baseline: GoalContract, targets: AmendmentTargetMap, scope: AmendmentScopeMap|null, amended: ObligationComposition|null, impacts: AmendmentImpactMap|null, preservation: AmendmentGuardrail, cutover: AmendmentGuardrail, release: AmendmentReleasePlan|null }
returns: AmendmentCheckpoint
---
Preserve the exact executed agreement edition and each amendment’s mapped clause target and effective interval. Append changed obligation editions without rewriting completed acts. Walk impact only along declared dependency edges and revalidate tasks crossing cutover. Preserve every exception that the edited baseline would otherwise remove. For retroactive billing, retain the original invoice and link a reviewed correction; a proposed correction is not a billing receipt. Duplicate cutover receipts do not create another revision.
