---
args: { dependencies: ConfigurationDependency[], evidence: Evidence[] }
returns: ID[]
---
Follow only supplied dependency edges whose source configuration is in the approved change scope. Return affected downstream identifiers and preserve all nodes outside that traversed impact; uncertain edge evidence stays unresolved rather than expanding scope by guess.
