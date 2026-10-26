---
description: Classify an explicit performance event and bind its scene, person, prop and stop-signal identity.
args: { show: LiveShowState; event: TheatreEvent }
returns: ShowIncident
---
Interpret the event evidence in the context of the exact production, checkpoint version, and current safety rules. Classify as participant-decline, cast-absence, prop-failure, knowledge-divergence, safety-stop or ordinary. If a stop signal is evidenced, it is the primary incident. Extract affected scene, participant/cast IDs, prop IDs, exact source sentence, and signal ID only where the source identifies them. Preserve stale or conflicting status in unresolved; do not assert a live condition from an older source. An ambiguous event scope remains unresolved.
