---
args: { change: DesignDelta, configurations: ConfigurationState[], edges: ConfigurationDependency[] }
returns: ApplicabilityPartition
---
Bind seeds = identifyDirectlyChangedConfigurations(change, configurations). Start the impact set only from explicitly changed functions or interfaces, and preserve grandfathered units on their approved revision. Bind closure = closeConfigurationDependencySet(seeds, edges). Traverse configuration dependencies using their supplied direction and evidence; do not turn a shared drawing or supplier into blanket applicability.
Return separate changed, preserved, and unresolved configuration IDs. Set documentationOnly only when the evidence shows reference renumbering with unchanged function and interface. Record traversal evidence and any missing dependency evidence so downstream work can identify exactly where scope ends.
