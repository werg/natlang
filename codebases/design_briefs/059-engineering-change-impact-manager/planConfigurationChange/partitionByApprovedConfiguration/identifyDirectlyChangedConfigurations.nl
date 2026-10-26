---
args: { change: DesignDelta, configurations: ConfigurationState[] }
returns: ConfigSeedSet
---
Match changed design IDs, function, interface, serial range, and approved baseline revision against each configuration record. Seed only configurations with direct evidence of the affected interface. Keep a grandfathered configuration outside the changed set unless the approved change explicitly names it. Missing revision or serial coverage remains unresolved.
