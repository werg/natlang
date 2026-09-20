export type NpcEvent = { from: Text, text: Text };
export type Memory = { id: Text, from: Text, text: Text };
export type Commitment = { to: Text, detail: Text, evidence_id: Text };
export type NpcObservation = { actor: Text, event_id: Text, event: NpcEvent,
  inventory: Dict<Num>, memory: Memory[], commitments: Commitment[] };
export type NpcPlan = { say: Text, action: Text, item?: Text,
  target?: Text, detail?: Text };
export type NpcResult = { actor: Text, said: Text, action: Text,
  evidence_id: Text, inventory: Dict<Num>, commitments: Commitment[] };
