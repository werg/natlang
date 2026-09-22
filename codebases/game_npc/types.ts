export type NpcEvent = { from: string, text: string };
export type Memory = { id: string, from: string, text: string };
export type Commitment = { to: string, detail: string, evidence_id: string };
export type NpcObservation = { actor: string, event_id: string, event: NpcEvent,
  inventory: Record<string, number>, memory: Memory[], commitments: Commitment[] };
export type NpcPlan = { say: string, action: string, item?: string,
  target?: string, detail?: string };
export type NpcResult = { actor: string, said: string, action: string,
  evidence_id: string, inventory: Record<string, number>, commitments: Commitment[] };
