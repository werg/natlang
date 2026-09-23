import type { WorkflowEvent, Operation, WorkflowState, Decision } from "../types.js";
import { host } from "natlang:runtime";

export default async function inspect(order_id: string): Promise<WorkflowState> {
return await host.workflow.read(order_id);
}
