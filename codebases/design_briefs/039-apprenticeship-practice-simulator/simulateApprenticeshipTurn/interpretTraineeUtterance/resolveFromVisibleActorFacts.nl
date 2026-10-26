---
args: { question: string, actor: SimulatedActor, world: TrainingWorld }
returns: ActorAnswer
---
Search the named actor's visible facts and authorized reference materials for the requested information. Return an answer only when those facts support it. If the fact is hidden from this actor, say what must be checked through the simulated procedure rather than leaking it. Record the fact references used.
