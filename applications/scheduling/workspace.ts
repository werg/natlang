/**
 * The scheduler's exact side. `Problem` is a frozen set of facts (windows, tasks, commitments) over integer UTC
 * minutes: it checks any set of placements against every hard constraint and names each violation, and it enumerates
 * complete feasible schedules by depth-first search. `ScheduleWorkspace` owns the revision and the committed plan; a
 * decision made on a snapshot is applied by `commit` or `commitPlan` only if the revision is unchanged and the
 * verifier accepts it. Nothing here judges preferences: that is the natural-language scheduler (`scheduler.nl`).
 */
import { createHash } from 'node:crypto';
import type { Block, DayView, Hard, Offered, Placement, TaskView, Verdict, Violation } from './types.js';

export type Task = { id: string, minutes: number, earliest: number, latest: number, after: string[], preference?: string };
export type TaskInput = Omit<Task, 'earliest' | 'latest'> & { earliest: string, latest: string };
export type Slot = { id: string, start: number, end: number };
export type Interval = { start: string, end: string };
export type ScheduleSnapshot = { revision: number, identity: string, tasks: Task[], fixed: Slot[], plan: Slot[] };
export type Candidate = { id: string, slots: Slot[] };
export type Alternatives = { revision: number, options: Candidate[], truncated: boolean, searched: number, detail: string };
export type ScheduleResult = { status: 'committed' | 'stale' | 'rejected' | 'infeasible' | 'unclear', revision: number, plan: Slot[],
  detail: string, explanation?: string };
export type BlockEvent = { kind: 'block', id: string, start: string, end: string };

const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const ID = /^[A-Za-z][A-Za-z0-9_-]*$/;
const overlap = (a: { start: number, end: number }, b: { start: number, end: number }) => a.start < b.end && b.start < a.end;
const whole = (value: unknown): value is number => Number.isSafeInteger(value);
const MAX_VIOLATIONS = 40;

function minute(value: string): number {
  if (typeof value !== 'string' || !/(?:Z|[+-]\d\d:\d\d)$/.test(value)) throw new Error('time requires explicit UTC offset');
  const milliseconds = Date.parse(value);
  if (!Number.isFinite(milliseconds) || milliseconds % 60000 !== 0) throw new Error('invalid minute time');
  return milliseconds / 60000;
}
/** The UTC offset of an ISO time, in minutes. */
function offsetOf(value: string): number {
  const match = /([+-])(\d\d):(\d\d)$/.exec(value);
  return match ? (match[1] === '-' ? -1 : 1) * (Number(match[2]) * 60 + Number(match[3])) : 0;
}
const two = (n: number) => String(n).padStart(2, '0');
function offsetText(offset: number): string {
  return offset === 0 ? 'Z' : `${offset < 0 ? '-' : '+'}${two(Math.floor(Math.abs(offset) / 60))}:${two(Math.abs(offset) % 60)}`;
}
/** An epoch minute as an ISO time at the given offset. */
export function isoAt(epoch: number, offset: number): string {
  return `${new Date((epoch + offset) * 60000).toISOString().slice(0, 16)}${offsetText(offset)}`;
}

/** The facts of one day, frozen: the verifier and the exact enumeration. */
export class Problem {
  constructor(readonly windows: { start: number, end: number }[], readonly tasks: Map<string, Task>, readonly fixed: Slot[],
    readonly slot: number, readonly origin: number, readonly offset: number) {}

  /** Dependency order; throws "task dependency cycle". */
  order(): string[] {
    const done = new Set<string>(), active = new Set<string>(), sorted: string[] = [];
    const visit = (id: string) => {
      if (active.has(id)) throw new Error('task dependency cycle');
      if (done.has(id)) return;
      active.add(id);
      for (const dependency of this.tasks.get(id)!.after) visit(dependency);
      active.delete(id); done.add(id); sorted.push(id);
    };
    for (const id of this.tasks.keys()) visit(id);
    return sorted;
  }

  private rel(epoch: number): number { return epoch - this.origin; }
  /** A clock time: HH:MM on the origin's day, with the date when it is another day. */
  clock(epoch: number): string {
    const text = isoAt(epoch, this.offset);
    return text.slice(0, 10) === isoAt(this.origin, this.offset).slice(0, 10) ? text.slice(11, 16) : `${text.slice(5, 10)} ${text.slice(11, 16)}`;
  }
  private span(row: { start: number, end: number }): string { return `${this.clock(row.start)}-${this.clock(row.end)}`; }

  toEpoch(placements: Placement[]): Slot[] {
    return placements.map(row => ({ id: row?.id, start: this.origin + row?.start, end: this.origin + row?.end }));
  }
  toOffsets(slots: Slot[]): Placement[] {
    return slots.map(row => ({ id: row.id, start: this.rel(row.start), end: this.rel(row.end) }));
  }

  view(revision: number, plan: Slot[]): DayView {
    const taskView = (task: Task): TaskView => ({ id: task.id, minutes: task.minutes, earliest: this.rel(task.earliest),
      latest: this.rel(task.latest), after: [...task.after], ...task.preference ? { preference: task.preference } : {} });
    const windows = this.windows.map(row => ({ start: this.rel(row.start), end: this.rel(row.end) })).sort((a, b) => a.start - b.start);
    return { revision, origin: isoAt(this.origin, this.offset), slot: this.slot, windows,
      tasks: [...this.tasks.values()].map(taskView),
      fixed: this.fixed.map(row => ({ id: row.id, start: this.rel(row.start), end: this.rel(row.end) })),
      plan: this.toOffsets(plan),
      clock: [...this.windows.map(row => `window ${this.span(row)}`), ...this.fixed.map(row => `${row.id} ${this.span(row)}`)] };
  }

  /** The placements as lines of clock times, earliest first: "draft 09:00-09:30". */
  describe(placements: Placement[]): string[] {
    return [...placements].sort((a, b) => a.start - b.start || a.end - b.end)
      .map(row => `${row.id} ${this.span({ start: this.origin + row.start, end: this.origin + row.end })}`);
  }

  private clone(): { windows: { start: number, end: number }[], tasks: Map<string, Task>, fixed: Slot[] } {
    return { windows: this.windows.map(row => ({ ...row })), fixed: this.fixed.map(row => ({ ...row })),
      tasks: new Map([...this.tasks].map(([id, task]) => [id, { ...task, after: [...task.after] }])) };
  }

  /** The problem with a request's requirements added: new tasks, limits narrowing tasks, commitments. */
  with(hard: Hard): { problem?: Problem, violations: Violation[] } {
    const violations: Violation[] = [];
    const bad = (task: string, detail: string) => violations.push({ rule: 'requirements', task, detail });
    const { windows, tasks, fixed } = this.clone();
    for (const row of hard.tasks ?? []) {
      if (!row || typeof row.id !== 'string' || !ID.test(row.id)) { bad(String(row?.id), 'a task id starts with a letter and has only letters, digits, underscores and hyphens'); continue; }
      if (tasks.has(row.id)) { bad(row.id, `${row.id} is already a task of the day; give a new task its own id`); continue; }
      if (!whole(row.minutes) || row.minutes < 1) { bad(row.id, `${row.id} needs a whole number of minutes, at least 1`); continue; }
      if (!whole(row.earliest) || !whole(row.latest) || row.earliest >= row.latest) { bad(row.id, `${row.id} needs earliest before latest, in whole minutes`); continue; }
      tasks.set(row.id, { id: row.id, minutes: row.minutes, earliest: this.origin + row.earliest, latest: this.origin + row.latest,
        after: [...row.after ?? []], ...row.preference ? { preference: row.preference } : {} });
    }
    for (const limit of hard.limits ?? []) {
      const task = tasks.get(limit?.task);
      if (!task) { bad(String(limit?.task), `a limit names ${limit?.task}, which is not a task; name a task of the day or one the request adds`); continue; }
      if (limit.notBefore !== undefined) {
        if (!whole(limit.notBefore)) { bad(task.id, 'notBefore is a whole number of minutes after the origin'); continue; }
        task.earliest = Math.max(task.earliest, this.origin + limit.notBefore);
      }
      if (limit.endBy !== undefined) {
        if (!whole(limit.endBy)) { bad(task.id, 'endBy is a whole number of minutes after the origin'); continue; }
        task.latest = Math.min(task.latest, this.origin + limit.endBy);
      }
      for (const dependency of limit.after ?? []) if (!task.after.includes(dependency)) task.after.push(dependency);
    }
    for (const block of hard.blocks ?? []) {
      if (!block || typeof block.id !== 'string' || !ID.test(block.id)) { bad(String(block?.id), 'a commitment id starts with a letter and has only letters, digits, underscores and hyphens'); continue; }
      if (!whole(block.start) || !whole(block.end) || block.start >= block.end) { bad(block.id, `${block.id} needs start before end, in whole minutes after the origin`); continue; }
      const row = { id: block.id, start: this.origin + block.start, end: this.origin + block.end };
      const prior = fixed.find(item => item.id === row.id);
      if (!prior) fixed.push(row);
      else if (prior.start !== row.start || prior.end !== row.end) bad(row.id, `${row.id} is already a commitment at ${this.span(prior)}; give a different commitment its own id`);
    }
    for (const task of tasks.values()) {
      const unknown = task.after.find(id => !tasks.has(id));
      if (unknown) bad(task.id, `${task.id} depends on ${unknown}, which is not a task`);
      if (task.after.includes(task.id)) bad(task.id, `${task.id} depends on itself`);
    }
    if (violations.length) return { violations };
    const problem = new Problem(windows, tasks, fixed, this.slot, this.origin, this.offset);
    try { problem.order(); } catch { return { violations: [{ rule: 'requirements', task: '', detail: 'the tasks depend on each other in a cycle; remove one dependency' }] }; }
    return { problem, violations };
  }

  /** Every hard constraint a set of placements breaks, in epoch minutes. */
  violations(slots: Slot[]): Violation[] {
    const found: Violation[] = [];
    const say = (rule: Violation['rule'], task: string, detail: string) => { if (found.length < MAX_VIOLATIONS) found.push({ rule, task, detail }); };
    const seen = new Map<string, Slot>();
    for (const row of slots) {
      if (!row || typeof row.id !== 'string') { say('unknown-task', String(row?.id), 'a placement names a task by id'); continue; }
      if (!this.tasks.has(row.id)) { say('unknown-task', row.id, `${row.id} is not a task of the day`); continue; }
      if (seen.has(row.id)) { say('duplicate-task', row.id, `${row.id} is placed more than once; place each task once`); continue; }
      seen.set(row.id, row);
    }
    for (const id of this.tasks.keys()) if (!seen.has(id)) say('missing-task', id, `${id} has no placement; place every task`);
    const placed = [...seen.values()];
    for (const row of placed) {
      const task = this.tasks.get(row.id)!;
      if (!whole(row.start) || !whole(row.end)) { say('duration', row.id, `${row.id} needs whole-minute start and end`); continue; }
      const at = `minutes ${this.rel(row.start)}-${this.rel(row.end)}`;
      if (row.end - row.start !== task.minutes) say('duration', row.id, `${row.id} lasts ${row.end - row.start} minutes (${at}); it needs ${task.minutes}, so end = start + ${task.minutes}`);
      if (row.start < task.earliest) say('bounds', row.id, `${row.id} starts at minute ${this.rel(row.start)}; it may start from minute ${this.rel(task.earliest)}`);
      if (row.end > task.latest) say('bounds', row.id, `${row.id} ends at minute ${this.rel(row.end)}; it must end by minute ${this.rel(task.latest)}`);
      if ((((row.start - this.origin) % this.slot) + this.slot) % this.slot !== 0) say('grid', row.id, `${row.id} starts at minute ${this.rel(row.start)}; a start is a multiple of ${this.slot}`);
      if (!this.windows.some(window => row.start >= window.start && row.end <= window.end))
        say('window', row.id, `${row.id} (${at}) is not inside one working window; the windows are ${this.windows.map(window => `${this.rel(window.start)}-${this.rel(window.end)}`).join(', ')}`);
      for (const fixed of this.fixed) if (overlap(row, fixed))
        say('fixed', row.id, `${row.id} (${at}) overlaps ${fixed.id} (minutes ${this.rel(fixed.start)}-${this.rel(fixed.end)}); move it before ${this.rel(fixed.start)} or after ${this.rel(fixed.end)}`);
      for (const dependency of task.after) {
        const before = seen.get(dependency);
        if (before && whole(before.end) && before.end > row.start)
          say('dependency', row.id, `${row.id} starts at minute ${this.rel(row.start)} but ${dependency} ends at minute ${this.rel(before.end)}; start ${row.id} at ${this.rel(before.end)} or later`);
      }
    }
    for (let i = 0; i < placed.length; i++) for (let j = i + 1; j < placed.length; j++)
      if (overlap(placed[i]!, placed[j]!)) say('overlap', placed[j]!.id,
        `${placed[j]!.id} (minutes ${this.rel(placed[j]!.start)}-${this.rel(placed[j]!.end)}) overlaps ${placed[i]!.id} (minutes ${this.rel(placed[i]!.start)}-${this.rel(placed[i]!.end)})`);
    return found;
  }

  /** The exact verifier: placements in minutes after the origin, with the request's requirements added. */
  check(placements: Placement[], hard?: Hard): Verdict {
    let problem: Problem = this;
    if (hard) {
      const extended = this.with(hard);
      if (!extended.problem) return { ok: false, violations: extended.violations };
      problem = extended.problem;
    }
    if (!Array.isArray(placements)) return { ok: false, violations: [{ rule: 'missing-task', task: '', detail: 'placements is a list of { id, start, end }' }] };
    const violations = problem.violations(problem.toEpoch(placements));
    return { ok: violations.length === 0, violations };
  }

  /** Complete feasible schedules in epoch minutes, up to `limit`: windows in order, starts ascending. */
  enumerateSlots(limit: number): { options: Slot[][], truncated: boolean, searched: number } {
    const order = this.order(), options: Slot[][] = [], placement = new Map<string, { start: number, end: number }>();
    let searched = 0, truncated = false;
    const visit = (index: number) => {
      if (index === order.length) {
        searched++;
        const slots = order.map(id => ({ id, ...placement.get(id)! }));
        if (options.length < limit) options.push(slots);
        else truncated = true;
        return;
      }
      if (truncated) return;
      const task = this.tasks.get(order[index]!)!;
      const dependencyEnd = Math.max(task.earliest, ...task.after.map(id => placement.get(id)!.end));
      for (const window of this.windows) {
        let start = Math.ceil(Math.max(window.start, dependencyEnd) / this.slot) * this.slot;
        const endLimit = Math.min(window.end, task.latest);
        for (; start + task.minutes <= endLimit; start += this.slot) {
          const slot = { start, end: start + task.minutes };
          if (this.fixed.some(row => overlap(row, slot)) || [...placement.values()].some(row => overlap(row, slot))) continue;
          placement.set(task.id, slot); visit(index + 1); placement.delete(task.id);
          if (truncated) return;
        }
      }
    };
    visit(0);
    return { options, truncated, searched };
  }

  /** Complete feasible schedules in minutes after the origin, with the request's requirements added. */
  enumerate(hard: Hard | undefined, limit: number): Offered {
    if (!whole(limit) || limit < 1) throw new Error('invalid limit');
    let problem: Problem = this;
    if (hard) {
      const extended = this.with(hard);
      if (!extended.problem) return { options: [], truncated: false, detail: extended.violations.map(row => row.detail).join('; ') };
      problem = extended.problem;
    }
    const found = problem.enumerateSlots(limit);
    return { options: found.options.map((slots, index) => ({ id: `c${index + 1}`, placements: problem.toOffsets(slots) })),
      truncated: found.truncated, detail: found.options.length ? '' : 'no feasible complete schedule' };
  }
}

/** Revisioned local scheduling over exact facts; a decision is applied only to the revision it was made on. */
export class ScheduleWorkspace {
  revision = 0;
  private readonly windows: { start: number, end: number }[];
  private readonly tasks: Map<string, Task>;
  private readonly fixed: Slot[];
  private readonly slot: number;
  private readonly origin: number;
  private readonly offset: number;
  private plan: Slot[] = [];
  private readonly events: Record<string, unknown>[] = [];

  constructor({ windows, tasks, fixed = [], slotMinutes = 15 }: { windows: Interval[], tasks: TaskInput[],
    fixed?: (Interval & { id: string })[], slotMinutes?: number }) {
    if (!Number.isSafeInteger(slotMinutes) || slotMinutes < 1) throw new Error('invalid slot');
    this.windows = windows.map(row => ({ start: minute(row.start), end: minute(row.end) }));
    this.tasks = new Map(tasks.map(task => [task.id, { ...task, earliest: minute(task.earliest), latest: minute(task.latest) }]));
    this.fixed = fixed.map(row => ({ id: row.id, start: minute(row.start), end: minute(row.end) }));
    this.slot = slotMinutes;
    if (this.tasks.size !== tasks.length || this.windows.some(row => row.start >= row.end) ||
        this.fixed.some(row => row.start >= row.end)) throw new Error('invalid schedule input');
    for (const task of this.tasks.values()) {
      if (!ID.test(task.id) || !Number.isSafeInteger(task.minutes) || task.minutes < 1 ||
          task.earliest >= task.latest || !Array.isArray(task.after) || task.after.some(id => !this.tasks.has(id)))
        throw new Error(`invalid task: ${task.id}`);
    }
    const earliest = this.windows.reduce((best, row, index) => row.start < this.windows[best]!.start ? index : best, 0);
    this.offset = windows.length ? offsetOf(windows[earliest]!.start) : 0;
    this.origin = Math.floor(Math.min(...this.windows.map(row => row.start)) / slotMinutes) * slotMinutes;
    this.problem().order();
  }

  /** A frozen copy of the facts: what a decision is made on. */
  problem(): Problem {
    return new Problem(this.windows.map(row => ({ ...row })), new Map([...this.tasks].map(([id, task]) => [id, { ...task, after: [...task.after] }])),
      structuredClone(this.fixed), this.slot, this.origin, this.offset);
  }

  snapshot(): ScheduleSnapshot {
    return { revision: this.revision,
      identity: hash({ windows: this.windows, tasks: [...this.tasks.values()], fixed: this.fixed, plan: this.plan }),
      tasks: [...this.tasks.values()].map(task => ({ ...task })), fixed: structuredClone(this.fixed), plan: structuredClone(this.plan) };
  }

  /** The day as the scheduler's stages read it: minutes after the origin. */
  view(): DayView { return this.problem().view(this.revision, this.plan); }

  /** Complete feasible schedules, up to `limit`. */
  alternatives({ limit = 64 } = {}): Alternatives {
    if (!Number.isSafeInteger(limit) || limit < 1) throw new Error('invalid limit');
    const found = this.problem().enumerateSlots(limit);
    const options = found.options.map(slots => ({ id: hash(slots), slots }));
    this.events.push({ operation: 'schedule.alternatives', revision: this.revision, returned: options.length, truncated: found.truncated });
    return { revision: this.revision, options, truncated: found.truncated, searched: found.searched, detail: options.length ? '' : 'no feasible complete schedule' };
  }

  commit(candidate: Candidate, expectedRevision: number): ScheduleResult {
    if (expectedRevision !== this.revision)
      return { status: 'stale', revision: this.revision, plan: structuredClone(this.plan), detail: 'schedule changed' };
    if (!this.validCandidate(candidate))
      return { status: 'rejected', revision: this.revision, plan: structuredClone(this.plan), detail: 'candidate is not feasible' };
    this.plan = structuredClone(candidate.slots);
    this.revision++;
    this.events.push({ operation: 'schedule.commit', revision: this.revision, candidate_id: candidate.id });
    return { status: 'committed', revision: this.revision, plan: structuredClone(this.plan), detail: '' };
  }

  private validCandidate(candidate: Candidate): boolean {
    if (!candidate || !Array.isArray(candidate.slots) || candidate.slots.length !== this.tasks.size ||
        candidate.id !== hash(candidate.slots)) return false;
    return this.problem().violations(candidate.slots).length === 0;
  }

  /**
   * Commit placements (minutes after the origin) decided on revision `expectedRevision`. The verifier checks them
   * against the day plus the request's requirements; on success the request's new tasks and commitments join the day
   * (its limits apply to this plan only), the plan is replaced and the revision advances once.
   */
  commitPlan(placements: Placement[], expectedRevision: number, hard?: Hard): ScheduleResult {
    if (expectedRevision !== this.revision)
      return { status: 'stale', revision: this.revision, plan: structuredClone(this.plan), detail: 'schedule changed' };
    const problem = this.problem();
    const verdict = problem.check(placements, hard);
    if (!verdict.ok) return { status: 'rejected', revision: this.revision, plan: structuredClone(this.plan),
      detail: `the plan breaks ${verdict.violations.length} constraint${verdict.violations.length === 1 ? '' : 's'}: ${verdict.violations.slice(0, 3).map(row => row.detail).join('; ')}` };
    const admitted = hard ? problem.with({ tasks: hard.tasks, limits: [], blocks: hard.blocks }).problem! : problem;
    for (const [id, task] of admitted.tasks) if (!this.tasks.has(id)) this.tasks.set(id, task);
    for (const row of admitted.fixed) if (!this.fixed.some(item => item.id === row.id)) this.fixed.push(row);
    this.plan = problem.toEpoch(placements);
    this.revision++;
    this.events.push({ operation: 'schedule.plan', revision: this.revision, tasks_added: hard?.tasks.map(row => row.id) ?? [],
      blocks_added: hard?.blocks.map((row: Block) => row.id) ?? [] });
    return { status: 'committed', revision: this.revision, plan: structuredClone(this.plan), detail: '' };
  }

  /** Record an external calendar block; it advances the revision and invalidates stale proposals. */
  observe(event: BlockEvent): ScheduleSnapshot {
    if (event.kind !== 'block' || !event.id || !event.start || !event.end) throw new Error('invalid observation');
    const block = { id: event.id, start: minute(event.start), end: minute(event.end) };
    if (block.start >= block.end) throw new Error('invalid block');
    const prior = this.fixed.find(row => row.id === block.id);
    if (prior) {
      if (JSON.stringify(prior) !== JSON.stringify(block)) throw new Error('changed event ID');
      return this.snapshot();
    }
    this.fixed.push(block); this.revision++;
    this.events.push({ operation: 'schedule.observe', revision: this.revision, id: block.id,
      conflicted: this.plan.filter(row => overlap(row, block)).map(row => row.id) });
    return this.snapshot();
  }

  /** Add a record to the event log (the planner records its proposals here). */
  note(event: Record<string, unknown>): void { this.events.push(event); }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
}
