/**
 * Revisioned local scheduling. The workspace enumerates exact feasible schedules over integer UTC
 * minutes (windows, durations, dependencies, fixed commitments, non-overlap); natlang only ranks the
 * offered candidates against the user's soft preferences, and the pick is re-checked before commit.
 */
import { createHash } from 'node:crypto';
import { nl } from '@natlang/node';

export type Task = { id: string, minutes: number, earliest: number, latest: number, after: string[], preference?: string };
export type TaskInput = Omit<Task, 'earliest' | 'latest'> & { earliest: string, latest: string };
export type Slot = { id: string, start: number, end: number };
export type Interval = { start: string, end: string };
export type ScheduleSnapshot = { revision: number, identity: string, tasks: Task[], fixed: Slot[], plan: Slot[] };
export type Candidate = { id: string, slots: Slot[] };
export type Alternatives = { revision: number, options: Candidate[], truncated: boolean, searched: number, detail: string };
export type ScheduleResult = { status: 'committed' | 'stale' | 'rejected' | 'infeasible', revision: number, plan: Slot[], detail: string };
export type BlockEvent = { kind: 'block', id: string, start: string, end: string };

const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
function minute(value: string): number {
  if (typeof value !== 'string' || !/(?:Z|[+-]\d\d:\d\d)$/.test(value)) throw new Error('time requires explicit UTC offset');
  const milliseconds = Date.parse(value);
  if (!Number.isFinite(milliseconds) || milliseconds % 60000 !== 0) throw new Error('invalid minute time');
  return milliseconds / 60000;
}
const overlap = (a: { start: number, end: number }, b: { start: number, end: number }) => a.start < b.end && b.start < a.end;

export class ScheduleWorkspace {
  revision = 0;
  private readonly windows: { start: number, end: number }[];
  private readonly tasks: Map<string, Task>;
  private readonly fixed: Slot[];
  private readonly slot: number;
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
      if (!/^[A-Za-z][A-Za-z0-9_-]*$/.test(task.id) || !Number.isSafeInteger(task.minutes) || task.minutes < 1 ||
          task.earliest >= task.latest || !Array.isArray(task.after) || task.after.some(id => !this.tasks.has(id)))
        throw new Error(`invalid task: ${task.id}`);
    }
    this.order();
  }

  private order(): string[] {
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

  snapshot(): ScheduleSnapshot {
    return { revision: this.revision,
      identity: hash({ windows: this.windows, tasks: [...this.tasks.values()], fixed: this.fixed, plan: this.plan }),
      tasks: [...this.tasks.values()].map(task => ({ ...task })), fixed: structuredClone(this.fixed), plan: structuredClone(this.plan) };
  }

  /** Complete feasible schedules, up to `limit`. */
  alternatives({ limit = 64 } = {}): Alternatives {
    if (!Number.isSafeInteger(limit) || limit < 1) throw new Error('invalid limit');
    const order = this.order(), options: Candidate[] = [], placement = new Map<string, { start: number, end: number }>();
    let searched = 0, truncated = false;
    const visit = (index: number) => {
      if (index === order.length) {
        searched++;
        const slots = order.map(id => ({ id, ...placement.get(id)! }));
        if (options.length < limit) options.push({ id: hash(slots), slots });
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
    this.events.push({ operation: 'schedule.alternatives', revision: this.revision, returned: options.length, truncated });
    return { revision: this.revision, options, truncated, searched, detail: options.length ? '' : 'no feasible complete schedule' };
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
    const byId = new Map(candidate.slots.map(row => [row.id, row]));
    if (byId.size !== this.tasks.size) return false;
    return candidate.slots.every(row => {
      const task = this.tasks.get(row.id);
      return task && Number.isSafeInteger(row.start) && Number.isSafeInteger(row.end) && row.end - row.start === task.minutes &&
        row.start >= task.earliest && row.end <= task.latest && row.start % this.slot === 0 &&
        this.windows.some(window => row.start >= window.start && row.end <= window.end) &&
        !this.fixed.some(fixed => overlap(row, fixed)) && !candidate.slots.some(other => other !== row && overlap(row, other)) &&
        task.after.every(id => byId.has(id) && byId.get(id)!.end <= row.start);
    });
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

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
}

/** Plan the day for a free-text request: enumerate exactly, rank in natural language, commit conditionally. */
export async function plan(scheduler: ScheduleWorkspace, request: string): Promise<ScheduleResult> {
  const snapshot = scheduler.snapshot();
  const alternatives = scheduler.alternatives();
  if (!alternatives.options.length)
    return { status: 'infeasible', revision: snapshot.revision, plan: snapshot.plan, detail: alternatives.detail };
  const chosen: Candidate = await nl`Choose one candidate from alternatives.options for the request, reading
the user's soft preferences; hard constraints are already checked. Return the offered candidate unchanged
(same id and slots). If alternatives.truncated is true, the ranking saw only a bounded subset.`(request, snapshot, alternatives);
  return scheduler.commit(chosen, snapshot.revision);
}
