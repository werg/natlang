/*---
engine: typescript-host
args:
  state: State
  event: UiEvent
  decision: Decision
returns: Step
---*/
return await host.studio.apply(args.state, args.event, args.decision);
