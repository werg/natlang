export type Task = { id: string, minutes: number, earliest: number, latest: number,
  after: string[], preference?: string };
export type Slot = { id: string, start: number, end: number };
export type ScheduleSnapshot = { revision: number, identity: string, tasks: Task[],
  fixed: Slot[], plan: Slot[] };
export type Candidate = { id: string, slots: Slot[] };
export type Alternatives = { revision: number, options: Candidate[], truncated: boolean,
  searched: number, detail: string };
export type ScheduleResult = { status: string, revision: number, plan: Slot[], detail: string };
