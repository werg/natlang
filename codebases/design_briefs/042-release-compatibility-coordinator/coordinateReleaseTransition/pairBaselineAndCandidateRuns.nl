---
args: { consumers: BehaviorConsumerMap[], changes: ReleaseBehaviorDelta[], release: CompatibilityReleaseRequest }
returns: PairedCompatibilityReplay[]
---
For each affected consumer bind one scenario, input, baseline revision, candidate revision, and independent acceptance criterion. Keep the input and environment constant across both runs. Judge outcomes against intended consumer behavior; report acceptable differences, breaking differences, and unresolved contract changes separately.
