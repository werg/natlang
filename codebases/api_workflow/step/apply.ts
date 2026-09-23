export default async function apply(current: WorkflowState, item: WorkflowEvent, decision: Decision): Promise<WorkflowState> {
return await host.workflow.apply(current.order_id, current.revision,
  item, decision);
}
