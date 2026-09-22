export type WorkflowEvent = { kind: string, fault?: string };
export type Operation = { key: string, action: string, status: string, detail: string };
export type WorkflowState = { order_id: string, amount: number, revision: number,
  phase: string, pending: string, obligations: string[], history: Operation[] };
export type Decision = { action: string, reason: string };
