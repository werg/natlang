---
args: { observations: SignalObservationSet }
returns: CustomerSignalDossier
---
Preserve each supplied partition in its matching evidence role: conversations as episode evidence, exposure frames as denominator coverage, identity events as versioned mappings, consent events as withdrawal evidence, and channel coverage as comparability evidence. No supplied aggregate receipt or raw episode abstraction is available in this input shape, so leave those collections empty; do not manufacture counts or identities.
