/*---
engine: typescript-host
args:
  order_id: string
returns: WorkflowState
---*/
return await host.workflow.read(args.order_id);
