---
args: { request: HarmonizationRequest, measures: MeasureDefinition[], evidence: EvidencePage[] }
returns: MeasureBridge
---
Compare construct definition, instrument, population, timepoint, response options, and direction before aligning values. A shared label does not prove equivalent meaning. Keep incompatible measures separate and record every semantic assumption as unresolved pending protocol-owner review.

For each left and right MeasureDefinition pair that could match the target, bind pairReview = compareOneConstructPair(request, leftDefinition, rightDefinition, evidence). Assemble the returned pair-specific meaning, scope and unresolved assumptions into the bridge. Never infer a positive bridge from shared terminology alone.
