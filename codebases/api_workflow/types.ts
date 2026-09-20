export type WorkflowEvent = { kind: Text, fault?: Text };
export type Operation = { key: Text, action: Text, status: Text, detail: Text };
export type WorkflowState = { order_id: Text, amount: Num, revision: Num,
  phase: Text, pending: Text, obligations: Text[], history: Operation[] };
export type Decision = { action: Text, reason: Text };
