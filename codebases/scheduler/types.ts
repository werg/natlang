export type Task = { id: Text, minutes: Num, earliest: Num, latest: Num,
  after: Text[], preference?: Text };
export type Slot = { id: Text, start: Num, end: Num };
export type ScheduleSnapshot = { revision: Num, identity: Text, tasks: Task[],
  fixed: Slot[], plan: Slot[] };
export type Candidate = { id: Text, slots: Slot[] };
export type Alternatives = { revision: Num, options: Candidate[], truncated: Bool,
  searched: Num, detail: Text };
export type ScheduleResult = { status: Text, revision: Num, plan: Slot[], detail: Text };
