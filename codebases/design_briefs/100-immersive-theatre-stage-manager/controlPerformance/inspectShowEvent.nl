---
description: Identify the concrete live-show incident from the current show state and new event evidence.
args: { show: LiveShowState; event: TheatreEvent }
returns: ShowIncident
---
Bind incident = classifyShowSignals(show, event). Check that a current safety-stop signal takes precedence over dramatic handling and that concurrent cast, prop, consent and knowledge changes remain in incident.unresolved. Return that complete ShowIncident with the original event statement and exact supporting evidence IDs. Do not discard a concurrent refusal or resource failure because a stop currently blocks execution.
