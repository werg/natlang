/*---
engine: typescript-host
args:
  order_id: Text
returns: WorkflowState
---*/
return await host.workflow.read(args.order_id);
