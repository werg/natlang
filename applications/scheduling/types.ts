// The scheduler's stages pass these values to each other. Every time is a whole number of minutes after
// DayView.origin, so no stage handles an epoch number, a date library or a time zone.

/** A span of time. Minutes after the origin; start is before end. */
export type Span = { start: number, end: number };

/** Where a task is placed. Minutes after the origin; end - start is the task's duration; start is a multiple of the slot. */
export type Placement = { id: string, start: number, end: number };

/** A task to place. */
export type TaskView = {
  /** Starts with a letter; letters, digits, underscore and hyphen only. */
  id: string,
  /** Duration in minutes, at least 1. */
  minutes: number,
  /** The task starts no earlier than this minute. */
  earliest: number,
  /** The task ends no later than this minute. */
  latest: number,
  /** Ids of the tasks that end before this one starts. */
  after: string[],
  /** The user's own words about when it suits them, if they said any. */
  preference?: string,
};

/** A commitment that occupies time: minutes after the origin. */
export type Commitment = { id: string, start: number, end: number };

/** The day as the scheduler sees it: a snapshot at one revision. */
export type DayView = {
  revision: number,
  /** Minute 0 as an ISO time with its UTC offset, e.g. "2026-09-21T09:00+02:00". A multiple of the slot. */
  origin: string,
  /** Minutes between allowed start times: every start is a multiple of slot. */
  slot: number,
  /** Where work may happen, ascending. */
  windows: Span[],
  tasks: TaskView[],
  /** Commitments that already occupy time. No task overlaps one. */
  fixed: Commitment[],
  /** The plan committed now, empty before the first. */
  plan: Placement[],
  /** The windows and commitments as clock times, for reading, e.g. "window 09:00-12:00", "meeting 10:00-10:30". */
  clock: string[],
};

/** A commitment the request states ("I'm at the dentist 11:00-11:30"). It occupies time like any fixed commitment. */
export type Block = { id: string, start: number, end: number, reason: string };

/** A hard limit the request puts on one task. Only the fields that narrow the task are set. */
export type Limit = {
  task: string,
  /** The task starts no earlier than this minute. */
  notBefore?: number,
  /** The task ends no later than this minute. */
  endBy?: number,
  /** Ids of tasks that must end before this one starts, in addition to its own. */
  after?: string[],
  /** The words of the request this limit restates. */
  reason: string,
};

/** What the request fixes: it holds in every acceptable plan. */
export type Hard = { tasks: TaskView[], limits: Limit[], blocks: Block[] };

/** A wish the request states. It is judged, not enforced. */
export type Preference = {
  /** "p1", "p2", ... in the order the request states them. */
  id: string,
  /** The wish in one sentence. */
  text: string,
  /** Ids of the tasks it concerns; empty when it concerns the whole plan. */
  tasks: string[],
  /** 1 for a mild wish, 2 for a clear one, 3 for one stated as important. */
  weight: number,
};

/** What `read-tasks` returns: tasks the request introduces, and questions it cannot settle itself. */
export type TaskReading = { tasks: TaskView[], questions: string[] };
/** What `read-limits` returns: hard limits and commitments, and questions it cannot settle itself. */
export type LimitReading = { limits: Limit[], blocks: Block[], questions: string[] };
/** What `read-preferences` returns. */
export type PreferenceReading = { preferences: Preference[], questions: string[] };

/** Everything the request requires and wishes, ready for the stages that build candidates. */
export type Requirements = { hard: Hard, preferences: Preference[], questions: string[] };

/** The places one task can go. */
export type Domain = {
  task: string,
  minutes: number,
  /** The task's own dependencies joined with the limits' dependencies. */
  after: string[],
  /**
   * Free time inside the task's earliest and latest, ascending and disjoint, each at least `minutes` long and starting
   * at a multiple of the slot. The task goes inside one span: start from span.start or later, end by span.end.
   */
  spans: Span[],
};

/** Tasks in an order where every task comes after the tasks it depends on; or the problem with the dependencies. */
export type Ordering = { order: string[], problem: string };

/** One complete schedule. */
export type Offer = { id: string, placements: Placement[] };
/** Complete schedules for the day. */
export type Offered = { options: Offer[], truncated: boolean, detail: string };

/** A hard constraint a schedule breaks. */
export type Violation = {
  rule: 'unknown-task' | 'missing-task' | 'duplicate-task' | 'duration' | 'bounds' | 'window' | 'grid' | 'fixed' | 'overlap' | 'dependency' | 'requirements',
  task: string,
  /** What is wrong and what would make it right, in minutes after the origin. */
  detail: string,
};
/** The verifier's answer. */
export type Verdict = { ok: boolean, violations: Violation[] };

/** How well a schedule meets one preference. */
export type Fit = 'met' | 'partly' | 'missed';
export type Assessment = { candidate: string, preference: string, fit: Fit };
/** A candidate with its weighted total. */
export type Score = { candidate: string, total: number };

/** Which of two schedules serves the request better. */
export type Choice = 'first' | 'second';

/** Why no schedule exists. */
export type Diagnosis = {
  /** The tasks or commitments that cannot all hold, with clock times. */
  conflict: string,
  /** What the user could relax to get a schedule, most promising first. */
  relaxations: string[],
};

export type Settings = {
  /** How many complete schedules to build and compare at most. */
  candidates: number,
};

/** The scheduler's answer for one request. */
export type Proposal = {
  status: 'chosen' | 'infeasible' | 'unclear',
  /** The chosen schedule; empty unless chosen. */
  placements: Placement[],
  /** What the request fixes; the host admits it with the plan. */
  hard: Hard,
  /** Plain words: why this schedule, what it gives up; or why none exists; or what to ask. */
  explanation: string,
  /** Questions for the user; non-empty only when unclear. */
  questions: string[],
  /** The scores of the compared candidates, best first. */
  ranking: Score[],
  /** How many valid candidates were compared. */
  considered: number,
  /** True when more schedules existed than were compared. */
  truncated: boolean,
};
