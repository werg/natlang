import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import { ScheduleWorkspace, plan } from '../../applications/dist/scheduling/index.js';
import { scriptedModel } from './support/natlang.mjs';

const day = {
  windows: [{ start: '2026-09-21T09:00:00+02:00', end: '2026-09-21T12:00:00+02:00' }],
  fixed: [{ id: 'meeting', start: '2026-09-21T10:00:00+02:00',
    end: '2026-09-21T10:30:00+02:00' }],
  tasks: [
    { id: 'draft', minutes: 30, earliest: '2026-09-21T09:00:00+02:00',
      latest: '2026-09-21T11:30:00+02:00', after: [] },
    { id: 'review', minutes: 30, earliest: '2026-09-21T09:00:00+02:00',
      latest: '2026-09-21T12:00:00+02:00', after: ['draft'] },
  ],
};
const run = runtime => (fn, options) => runtime.run(fn, options);

// Stage implementations a model might write in eval, as real code: the scripted model returns them for the
// stage whose instructions it is shown. Each is a function of the stage's arguments.
const impl = {
  readTasks: (request) => /workout/.test(request)
    ? { tasks: [{ id: 'workout', minutes: 45, earliest: 0, latest: 180, after: [] }], questions: [] }
    : { tasks: [], questions: /\bsomething\b/.test(request) ? ['How long should it take?'] : [] },
  readLimits: (request, view) => /not before 11/.test(request)
    ? { limits: [{ task: 'review', notBefore: 120, reason: 'not before 11' }], blocks: [], questions: [] }
    : /dentist/.test(request)
      ? { limits: [], blocks: [{ id: 'dentist', start: 0, end: 30, reason: 'dentist' }], questions: [] }
      : { limits: [], blocks: [], questions: [] },
  readPreferences: (request) => /early/.test(request)
    ? { preferences: [{ id: 'p1', text: 'draft early', tasks: ['draft'], weight: 2 }], questions: [] }
    : { preferences: [], questions: [] },
  domains: (view, hard) => {
    const tasks = [...view.tasks, ...hard.tasks].map(t => ({ ...t, after: [...t.after] }));
    for (const l of hard.limits) {
      const t = tasks.find(x => x.id === l.task);
      if (l.notBefore !== undefined) t.earliest = Math.max(t.earliest, l.notBefore);
      if (l.endBy !== undefined) t.latest = Math.min(t.latest, l.endBy);
      for (const a of l.after ?? []) if (!t.after.includes(a)) t.after.push(a);
    }
    const busy = [...view.fixed, ...hard.blocks].map(b => ({ start: b.start, end: b.end })).sort((a, b) => a.start - b.start);
    const merged = [];
    for (const b of busy) { const last = merged.at(-1); if (last && b.start <= last.end) last.end = Math.max(last.end, b.end); else merged.push({ ...b }); }
    const free = [];
    for (const w of view.windows) {
      let from = w.start;
      for (const b of merged) if (b.end > from && b.start < w.end) { if (b.start > from) free.push({ start: from, end: b.start }); from = Math.max(from, b.end); }
      if (from < w.end) free.push({ start: from, end: w.end });
    }
    return tasks.map(t => ({ task: t.id, minutes: t.minutes, after: t.after, spans: free.flatMap(f => {
      const start = Math.ceil(Math.max(f.start, t.earliest) / view.slot) * view.slot, end = Math.min(f.end, t.latest);
      return end - start >= t.minutes ? [{ start, end }] : [];
    }) }));
  },
  construct: (domains, order, slot, limit) => {
    const placed = {}, options = []; let truncated = false;
    const place = remaining => {
      if (remaining === 0) { if (options.length < limit) options.push({ id: 'c' + (options.length + 1), placements: order.map(id => ({ id, ...placed[id] })) }); else truncated = true; return; }
      if (truncated) return;
      const d = domains.find(x => x.task === order[order.length - remaining]);
      const ready = Math.max(0, ...d.after.map(x => placed[x].end));
      for (const span of d.spans) {
        const first = Math.ceil(Math.max(span.start, ready) / slot) * slot;
        const count = Math.floor((span.end - d.minutes - first) / slot) + 1;
        for (let i = 0; i < count; i++) {
          const s = first + i * slot, e = s + d.minutes;
          if (Object.values(placed).some(p => p.start < e && p.end > s)) continue;
          placed[d.task] = { start: s, end: e }; place(remaining - 1); delete placed[d.task];
          if (truncated) return;
        }
      }
    };
    place(order.length);
    return { options, truncated, detail: options.length ? '' : 'no feasible complete schedule' };
  },
  assess: (schedule) => /^draft 09:00/.test(schedule.find(l => l.startsWith('draft')) ?? '') ? 'met' : 'missed',
  tiebreak: () => 'first',
  explain: (request, schedule) => `Plan: ${schedule[0]} first, ${schedule.at(-1)} last.`,
  diagnose: () => ({ conflict: 'The tasks do not fit.', relaxations: ['Widen the window.'] }),
};
const call = (name, args) => `return (${impl[name]})(${args});`;

/** The stages scripted by their instructions; the driver follows its own numbered steps. */
function plannerModel({ construct = null, corrupt = false } = {}) {
  const seen = [];
  const model = scriptedModel(opening => {
    const stage = [['Plan the day in view for request', 'scheduler'], ['list the tasks it introduces', 'readTasks'], ['list what it fixes', 'readLimits'],
      ['list the wishes it states', 'readPreferences'], ['Compute the places each task can go', 'domains'], ['Build complete schedules, up to limit', 'construct'],
      ['placements is a schedule that the verifier rejected', 'repair'], ['Decide how well the plan meets it', 'assess'], ['Both meet the request', 'tiebreak'],
      ['Write the explanation the user reads', 'explain'], ['No complete schedule satisfies', 'diagnose']].find(([marker]) => opening.includes(marker))?.[1];
    seen.push(stage);
    switch (stage) {
      case 'scheduler': return `
        const tasks = await readTasks(request, view);
        const all = [...view.tasks, ...tasks.tasks];
        const [limits, wishes] = await Promise.all([readLimits(request, view, all), readPreferences(request, all)]);
        const hard = { tasks: tasks.tasks, limits: limits.limits, blocks: limits.blocks };
        const questions = [...tasks.questions, ...limits.questions, ...wishes.questions];
        const empty = { placements: [], hard, questions: [], ranking: [], considered: 0, truncated: false };
        if (questions.length) return { ...empty, status: 'unclear', questions, explanation: 'To plan this I need to know: ' + questions.join(' ') };
        const doms = await domains(view, hard);
        const ordering = order(doms);
        if (ordering.problem) return { ...empty, status: 'unclear', questions: [ordering.problem], explanation: ordering.problem };
        const offered = await enumerate(doms, ordering.order, view.slot, hard, settings.candidates);
        let truncated = offered.truncated;
        let valid = offered.options.filter(o => calendar.check(o.placements, hard).ok);
        ${corrupt ? `if (!valid.length && offered.options.length) {
          let placements = offered.options[0].placements;
          for (let round = 0; round < 3 && !valid.length; round++) {
            const verdict = calendar.check(placements, hard);
            if (verdict.ok) { valid = [{ id: 'repaired', placements }]; break; }
            placements = await repair(placements, verdict.violations, doms, ordering.order, view.slot);
          }
        }` : ''}
        if (!valid.length) { const confirmed = calendar.exact(hard, settings.candidates); valid = confirmed.options; truncated = confirmed.truncated; }
        if (!valid.length) { const d = await diagnose(request, view, hard, doms); return { ...empty, status: 'infeasible', explanation: d.conflict + ' ' + d.relaxations.join(' ') }; }
        const described = valid.map(o => calendar.describe(o.placements));
        const pairs = valid.flatMap((o, i) => wishes.preferences.map(w => ({ o, i, w })));
        const answers = await Promise.all(pairs.map(pair => assess(described[pair.i], pair.w)));
        const fits = pairs.map((pair, k) => ({ candidate: pair.o.id, preference: pair.w.id, fit: answers[k] }));
        const points = { met: 2, partly: 1, missed: 0 };
        const scored = valid.map((o, i) => ({ o, i, total: fits.filter(a => a.candidate === o.id).reduce((sum, a) => sum + points[a.fit] * wishes.preferences.find(w => w.id === a.preference).weight, 0) }));
        const top = Math.max(...scored.map(s => s.total));
        let kept = scored.find(s => s.total === top);
        for (const s of scored.filter(s => s.total === top && s !== kept)) if (await tiebreak(request, described[kept.i], described[s.i]) === 'second') kept = s;
        const explanation = await explain(request, described[kept.i], wishes.preferences, fits.filter(a => a.candidate === kept.o.id), valid.length, truncated);
        return { status: 'chosen', placements: kept.o.placements, hard, explanation, questions: [], ranking: scored.map(s => ({ candidate: s.o.id, total: s.total })).sort((a, b) => b.total - a.total), considered: valid.length, truncated };`;
      case 'construct': return construct ?? call('construct', 'domains, order, slot, limit');
      case 'repair': return `
        // Move the violating task to the earliest free start in its domain.
        const kept = placements.filter(p => !violations.some(v => v.task === p.id));
        for (const id of order) {
          if (kept.some(p => p.id === id)) continue;
          const d = domains.find(x => x.task === id);
          const ready = Math.max(0, ...d.after.map(x => kept.find(p => p.id === x)?.end ?? 0));
          let found = null;
          for (const span of d.spans) {
            const first = Math.ceil(Math.max(span.start, ready) / slot) * slot;
            for (let i = 0; i < 100; i++) {
              const s = first + i * slot;
              if (found || s + d.minutes > span.end) break;
              if (!kept.some(p => p.start < s + d.minutes && p.end > s)) found = { id, start: s, end: s + d.minutes };
            }
          }
          kept.push(found ?? placements.find(p => p.id === id));
        }
        return order.map(id => kept.find(p => p.id === id));`;
      case 'readTasks': return call('readTasks', 'request');
      case 'readLimits': return call('readLimits', 'request, view');
      case 'readPreferences': return call('readPreferences', 'request');
      case 'domains': return call('domains', 'view, hard');
      case 'assess': return call('assess', 'schedule');
      case 'tiebreak': return call('tiebreak', '');
      case 'explain': return call('explain', 'request, schedule');
      case 'diagnose': return call('diagnose', '');
      default: return null;
    }
  });
  return { ...model, seen };
}

test('the scheduler reads the request, builds candidates, ranks them by the wish and the workspace commits', async () => {
  const workspace = new ScheduleWorkspace(day);
  const model = plannerModel();
  const result = await plan(workspace, 'Draft early, then review', { run: run(createNatlangRuntime({ model: model.driver })) });
  assert.equal(result.status, 'committed', result.detail);
  assert.equal(result.plan.length, 2);
  const draft = result.plan.find(row => row.id === 'draft'), review = result.plan.find(row => row.id === 'review');
  assert.ok(review.start >= draft.end);
  assert.match(result.explanation, /Plan: draft 09:00-09:30 first/);
  assert.ok(model.seen.includes('assess') && model.seen.includes('explain') && !model.seen.includes('construct'));
  assert.deepEqual(workspace.drainEvents().map(event => event.operation), ['schedule.proposal', 'schedule.plan']);
});

test('the request adds a task, a hard limit and a commitment, which the verifier enforces', async () => {
  const workspace = new ScheduleWorkspace(day);
  const runtime = createNatlangRuntime({ model: plannerModel().driver });
  const first = await plan(workspace, 'Add a workout, review not before 11', { run: run(runtime) });
  assert.equal(first.status, 'committed', first.detail);
  assert.equal(first.plan.length, 3);
  const view = workspace.view();
  assert.deepEqual(view.tasks.map(task => task.id), ['draft', 'review', 'workout']);
  const review = view.plan.find(row => row.id === 'review');
  // The limit holds for this plan only, and the task joined the day.
  assert.ok(review.start >= 120);
  const second = await plan(new ScheduleWorkspace(day), 'I am at the dentist, draft early', { run: run(runtime) });
  assert.equal(second.status, 'committed', second.detail);
  assert.ok(second.plan.every(row => row.start >= 0));
});

test('the natural-language enumeration is selected by a setting and gives the same plan as the exact one', async () => {
  const runtime = createNatlangRuntime({ model: plannerModel().driver });
  const exact = await plan(new ScheduleWorkspace(day), 'Draft early', { run: run(runtime), enumeration: 'crisp' });
  const model = plannerModel();
  const natural = await plan(new ScheduleWorkspace(day), 'Draft early',
    { run: run(createNatlangRuntime({ model: model.driver })), enumeration: 'natural-language' });
  assert.ok(model.seen.includes('construct'));
  assert.equal(natural.status, 'committed', natural.detail);
  assert.deepEqual(natural.plan, exact.plan);
});

test('a near-miss from natural-language construction is repaired, and the verifier names each violation', async () => {
  const workspace = new ScheduleWorkspace(day);
  // A construction that puts review on top of the meeting and draft; repair moves it.
  const wrong = `return { options: [{ id: 'c1', placements: [{ id: 'draft', start: 0, end: 30 }, { id: 'review', start: 60, end: 90 }] }], truncated: false, detail: '' };`;
  const model = plannerModel({ construct: wrong, corrupt: true });
  const result = await plan(workspace, 'Draft early', { run: run(createNatlangRuntime({ model: model.driver })), enumeration: 'natural-language' });
  assert.equal(result.status, 'committed', result.detail);
  assert.ok(model.seen.includes('repair'));
  const verdict = workspace.problem().check([{ id: 'draft', start: 0, end: 30 }, { id: 'review', start: 20, end: 50 }]);
  assert.equal(verdict.ok, false);
  assert.deepEqual(verdict.violations.map(row => row.rule).sort(), ['dependency', 'grid', 'overlap']);
  assert.match(verdict.violations.find(row => row.rule === 'dependency').detail, /start review at 30 or later/);
});

test('unclear requests return questions, impossible ones are diagnosed, and nothing is committed', async () => {
  const workspace = new ScheduleWorkspace(day);
  const runtime = createNatlangRuntime({ model: plannerModel().driver });
  const unclear = await plan(workspace, 'Fit in something', { run: run(runtime) });
  assert.equal(unclear.status, 'unclear');
  assert.match(unclear.detail, /How long/);
  const full = await plan(workspace, 'Add a workout, review not before 11, dentist', { run: run(runtime) });
  assert.equal(full.status, 'committed');
  const tight = new ScheduleWorkspace({ ...day, windows: [{ start: '2026-09-21T09:00:00+02:00', end: '2026-09-21T10:30:00+02:00' }] });
  const impossible = await plan(tight, 'Draft early, dentist', { run: run(runtime) });
  assert.equal(impossible.status, 'infeasible');
  assert.match(impossible.explanation, /do not fit/);
  assert.equal(tight.revision, 0);
});

test('a proposal that breaks a constraint is rejected by the commit', () => {
  const workspace = new ScheduleWorkspace(day);
  const result = workspace.commitPlan([{ id: 'draft', start: 60, end: 90 }, { id: 'review', start: 60, end: 90 }], 0);
  assert.equal(result.status, 'rejected');
  assert.match(result.detail, /fixed|overlaps/);
  assert.equal(workspace.revision, 0);
});

test('new observations invalidate stale proposals and impossible schedules remain explicit', () => {
  const scheduler = new ScheduleWorkspace(day);
  const initial = scheduler.alternatives();
  const candidate = initial.options[0];
  const block = { kind: 'block', id: 'urgent',
    start: new Date(candidate.slots[0].start * 60000).toISOString(),
    end: new Date(candidate.slots[0].end * 60000).toISOString() };
  scheduler.observe(block);
  assert.equal(scheduler.commit(candidate, initial.revision).status, 'stale');
  assert.equal(scheduler.commit(candidate, scheduler.revision).status, 'rejected');
  assert.equal(scheduler.observe(block).revision, 1);
  assert.throws(() => scheduler.observe({ ...block, end: '2026-09-21T12:00:00+02:00' }), /changed event ID/);
  assert.throws(() => new ScheduleWorkspace({ ...day,
    tasks: day.tasks.map(task => ({ ...task, after: ['review'] })) }), /dependency cycle/);
  assert.throws(() => new ScheduleWorkspace({ ...day,
    windows: [{ start: '2026-09-21T09:00:00', end: '2026-09-21T12:00:00' }] }),
  /explicit UTC offset/);
});

test('a calendar block observed during planning makes the plan stale', async () => {
  const workspace = new ScheduleWorkspace(day);
  const model = plannerModel();
  const inner = model.driver;
  let observed = false;
  const driver = async request => {
    if (!observed) { observed = true; workspace.observe({ kind: 'block', id: 'urgent', start: '2026-09-21T11:00:00+02:00', end: '2026-09-21T11:30:00+02:00' }); }
    return inner(request);
  };
  const result = await plan(workspace, 'Draft early', { run: run(createNatlangRuntime({ model: driver })) });
  assert.equal(result.status, 'stale');
  assert.equal(workspace.revision, 1);
});
