/** Revisioned local scheduling with exact UTC intervals and conditional commit. */
import { createHash } from 'node:crypto';

const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
function minute(value) {
  if (typeof value !== 'string' || !/(?:Z|[+-]\d\d:\d\d)$/.test(value))
    throw new Error('time requires explicit UTC offset');
  const milliseconds = Date.parse(value);
  if (!Number.isFinite(milliseconds) || milliseconds % 60000 !== 0)
    throw new Error('invalid minute time');
  return milliseconds / 60000;
}
const overlap = (a, b) => a.start < b.end && b.start < a.end;

export class ScheduleWorkspace {
  constructor({ windows, tasks, fixed = [], slotMinutes = 15 }) {
    if (!Number.isSafeInteger(slotMinutes) || slotMinutes < 1) throw new Error('invalid slot');
    this.windows = windows.map(row => ({ start: minute(row.start), end: minute(row.end) }));
    this.tasks = new Map(tasks.map(task => [task.id, { ...task,
      earliest: minute(task.earliest), latest: minute(task.latest) }]));
    this.fixed = fixed.map(row => ({ id: row.id, start: minute(row.start), end: minute(row.end) }));
    this.slot = slotMinutes; this.revision = 0; this.events = [];
    this.plan = [];
    if (this.tasks.size !== tasks.length || this.windows.some(row => row.start >= row.end) ||
        this.fixed.some(row => row.start >= row.end)) throw new Error('invalid schedule input');
    for (const task of this.tasks.values()) {
      if (!/^[A-Za-z][A-Za-z0-9_-]*$/.test(task.id) ||
          !Number.isSafeInteger(task.minutes) || task.minutes < 1 ||
          task.earliest >= task.latest || !Array.isArray(task.after) ||
          task.after.some(id => !this.tasks.has(id))) throw new Error(`invalid task: ${task.id}`);
    }
    this.#order();
  }

  #order() {
    const done = new Set(), active = new Set(), sorted = [];
    const visit = id => {
      if (active.has(id)) throw new Error('task dependency cycle');
      if (done.has(id)) return;
      active.add(id);
      for (const dependency of this.tasks.get(id).after) visit(dependency);
      active.delete(id); done.add(id); sorted.push(id);
    };
    for (const id of this.tasks.keys()) visit(id);
    return sorted;
  }

  snapshot() {
    return { revision: this.revision, identity: hash({ windows: this.windows,
      tasks: [...this.tasks.values()], fixed: this.fixed, plan: this.plan }),
      tasks: [...this.tasks.values()].map(task => ({ ...task })),
      fixed: structuredClone(this.fixed), plan: structuredClone(this.plan) };
  }

  alternatives({ limit = 64 } = {}) {
    if (!Number.isSafeInteger(limit) || limit < 1) throw new Error('invalid limit');
    const order = this.#order(), options = [], placement = new Map();
    let searched = 0, truncated = false;
    const visit = index => {
      if (index === order.length) {
        searched++;
        const slots = order.map(id => ({ id, ...placement.get(id) }));
        if (options.length < limit) options.push({ id: hash(slots), slots });
        else truncated = true;
        return;
      }
      if (truncated) return;
      const task = this.tasks.get(order[index]);
      const dependencyEnd = Math.max(task.earliest,
        ...task.after.map(id => placement.get(id).end));
      for (const window of this.windows) {
        let start = Math.ceil(Math.max(window.start, dependencyEnd) / this.slot) * this.slot;
        const endLimit = Math.min(window.end, task.latest);
        for (; start + task.minutes <= endLimit; start += this.slot) {
          const slot = { start, end: start + task.minutes };
          if (this.fixed.some(row => overlap(row, slot)) ||
              [...placement.values()].some(row => overlap(row, slot))) continue;
          placement.set(task.id, slot); visit(index + 1); placement.delete(task.id);
          if (truncated) return;
        }
      }
    };
    visit(0);
    const result = { revision: this.revision, options, truncated,
      searched, detail: options.length ? '' : 'no feasible complete schedule' };
    this.events.push({ operation: 'schedule.alternatives', revision: this.revision,
      returned: options.length, truncated });
    return result;
  }

  commit(candidate, expectedRevision) {
    if (expectedRevision !== this.revision) return { status: 'stale',
      revision: this.revision, plan: structuredClone(this.plan), detail: 'schedule changed' };
    if (!this.#validCandidate(candidate)) return { status: 'rejected', revision: this.revision,
      plan: structuredClone(this.plan), detail: 'candidate is not feasible' };
    this.plan = structuredClone(candidate.slots);
    this.revision++;
    this.events.push({ operation: 'schedule.commit', revision: this.revision,
      candidate_id: candidate.id });
    return { status: 'committed', revision: this.revision,
      plan: structuredClone(this.plan), detail: '' };
  }

  #validCandidate(candidate) {
    if (!candidate || !Array.isArray(candidate.slots) ||
        candidate.slots.length !== this.tasks.size ||
        candidate.id !== hash(candidate.slots)) return false;
    const byId = new Map(candidate.slots.map(row => [row.id, row]));
    if (byId.size !== this.tasks.size) return false;
    for (const row of candidate.slots) {
      const task = this.tasks.get(row.id);
      if (!task || !Number.isSafeInteger(row.start) || !Number.isSafeInteger(row.end) ||
          row.end - row.start !== task.minutes || row.start < task.earliest ||
          row.end > task.latest || row.start % this.slot !== 0 ||
          !this.windows.some(window => row.start >= window.start && row.end <= window.end) ||
          this.fixed.some(fixed => overlap(row, fixed)) ||
          candidate.slots.some(other => other !== row && overlap(row, other)) ||
          task.after.some(id => !byId.has(id) || byId.get(id).end > row.start)) return false;
    }
    return true;
  }

  observe(event) {
    if (event.kind !== 'block' || !event.id || !event.start || !event.end)
      throw new Error('invalid observation');
    const block = { id: event.id, start: minute(event.start), end: minute(event.end) };
    if (block.start >= block.end) throw new Error('invalid block');
    const prior = this.fixed.find(row => row.id === block.id);
    if (prior) {
      if (JSON.stringify(prior) !== JSON.stringify(block)) throw new Error('changed event ID');
      return this.snapshot();
    }
    this.fixed.push(block); this.revision++;
    this.events.push({ operation: 'schedule.observe', revision: this.revision,
      id: block.id, conflicted: this.plan.filter(row => overlap(row, block)).map(row => row.id) });
    return this.snapshot();
  }

  drainEvents() { return this.events.splice(0); }
}
