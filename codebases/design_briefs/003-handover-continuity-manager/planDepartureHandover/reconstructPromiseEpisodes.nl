---
args: {"docket": "DepartureDocket", "capture": "CaptureBoundary"}
returns: "PromiseEpisode[]"
---
Read only permitted conversation histories with replies and amendments in chronological order. For each supported promise episode bind episode = classifyPromiseAct(source, docket.departures). Preserve intention vs explicit promise and conditional activation. Unknown dates remain null. For a recurring task without a written deadline, propose cadence from actual recurrence evidence and mark cadenceConfirmed false until an authorized owner confirms. Return unresolved beneficiary, deliverable or attribution rather than completing those facts by guesswork.
