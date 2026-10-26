---
args: { change: DesignDelta, configurations: ConfigurationState[], edges: ConfigurationDependency[] }
returns: ApplicabilityPartition
---
Partition only by an explicit effective rule, approved revision, serial range, and configuration evidence. Preserve grandfathered units under their approved configuration and maintenance guidance. Detect a reference-number-only drawing update as documentationOnly when function and interface are unchanged; return unresolved units where evidence cannot establish applicability.
