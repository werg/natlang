export default async function inspect(order_id: string): Promise<WorkflowState> {
return await host.workflow.read(order_id);
}
