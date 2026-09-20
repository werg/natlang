/*---
engine: typescript-host
args:
  current: WorkflowState
  item: WorkflowEvent
  decision: Decision
returns: WorkflowState
---*/
return await host.workflow.apply(args.current.order_id, args.current.revision,
  args.item, args.decision);
