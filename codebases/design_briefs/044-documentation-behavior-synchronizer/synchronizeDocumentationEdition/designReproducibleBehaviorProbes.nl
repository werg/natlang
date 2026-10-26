---
args: { promises: VersionedDocumentationPromise[], documentation: DocumentationEditionRequest }
returns: BehaviorProbePlan[]
---
For each checkable claim define bounded inputs, expected outcome, environment identity, and target release. Join existing observations only when their environment and release match. Missing execution evidence becomes a prepared probe, never an observation. Keep a deprecated service example in an explicitly labeled environment until an owner approves its replacement.
