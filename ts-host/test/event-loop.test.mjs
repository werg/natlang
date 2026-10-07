import assert from 'node:assert/strict';
import { test } from 'node:test';
import { EventLoop, KeyedEventLoop } from '../dist/index.js';

const settle = ms => new Promise(resolve => setTimeout(resolve, ms));
const counterLoop = (options = {}) => new EventLoop({ initialState: { count: 0, log: [] },
  reduce: (state, event) => ({ ...state, count: state.count + 1, log: [...state.log, event.kind] }), view: state => state.count, ...options });

test('each step has one time, stored with its commit', async () => {
  let clock = 1000;
  const commits = [], seen = [];
  const loop = counterLoop({ clock: () => clock++, onCommit: commit => { commits.push(commit.now); },
    reduce: (state, event, context) => { seen.push(context.now, context.now); return { ...state, count: state.count + 1 }; } });
  await loop.start();
  await loop.dispatch({ id: 'a', kind: 'tick' });
  await loop.dispatch({ id: 'b', kind: 'tick' });
  assert.deepEqual(seen, [1001, 1001, 1003, 1003]);
  assert.deepEqual(commits, [1001, 1003]);
  await loop.close();
});

test('follow-up work runs after the commit and its event is applied next', async () => {
  const order = [];
  const loop = new EventLoop({ initialState: { answer: null },
    reduce: (state, event, context) => {
      if (event.kind === 'ask') context.after(async () => { order.push('work'); return { id: 'answer', kind: 'answered', value: 42 }; });
      return event.kind === 'answered' ? { answer: event.value } : state;
    },
    onCommit: commit => { order.push(`commit ${commit.event.kind}`); }, view: state => state.answer });
  await loop.start();
  const transition = await loop.dispatch({ id: 'q', kind: 'ask' });
  assert.equal(transition.view, null, 'the step does not wait for its follow-up');
  await settle(20);
  assert.deepEqual(order, ['commit ask', 'work', 'commit answered']);
  assert.equal(loop.state.answer, 42);
  await loop.close();
});

test('failed follow-up work is reported, and close stops what is still running', async () => {
  const failures = [];
  let aborted = false;
  const loop = new EventLoop({ initialState: {}, view: () => null, onFailure: failure => failures.push(failure),
    reduce: (state, event, context) => {
      if (event.kind === 'fail') context.after(() => { throw new Error('upstream down'); });
      if (event.kind === 'slow') context.after(signal => new Promise(resolve => signal.addEventListener('abort', () => {
        aborted = true; resolve({ id: 'late', kind: 'late' }); })));
      return state;
    } });
  await loop.start();
  await loop.dispatch({ id: 'f', kind: 'fail' });
  await settle(10);
  assert.equal(failures[0].stage, 'after'); assert.match(failures[0].error.message, /upstream down/);
  await loop.dispatch({ id: 's', kind: 'slow' });
  await loop.close();
  assert.ok(aborted);
  assert.ok(!loop.seenEventIds.includes('late'));
  const viewing = new EventLoop({ initialState: {}, reduce: state => state, view: (state, context) => context.after(() => null) });
  await assert.rejects(viewing.start(), /available while reducing/);
});

test('wakeAt delivers one wake event per time, re-arms on a later time, and survives a restart', async () => {
  const wakes = [];
  const options = { view: state => state.due,
    wakeAt: state => state.due,
    reduce: (state, event, context) => {
      if (event.kind === 'remind') return { due: context.now + 30, reminders: state.reminders };
      if (event.kind === 'wake') { wakes.push(event.at); return { due: state.reminders < 1 ? context.now + 30 : state.due, reminders: state.reminders + 1 }; }
      return state;
    } };
  const loop = new EventLoop({ ...options, initialState: { due: null, reminders: 0 } });
  await loop.start();
  await loop.dispatch({ id: 'r', kind: 'remind' });
  await settle(200);
  assert.equal(wakes.length, 2, 'woken at the first time and at the later one; the second time is not repeated');
  assert.ok(loop.seenEventIds.every(id => id === 'r' || id.startsWith('wake:')));
  await loop.close();
  const restarted = new EventLoop({ ...options, initialState: { due: Date.now() + 20, reminders: 1 }, initialRevision: 5 });
  await restarted.start();
  await settle(100);
  assert.equal(wakes.length, 3, 'a restored state arms its wake time');
  await restarted.close();
  const closed = new EventLoop({ ...options, initialState: { due: Date.now() + 30, reminders: 1 } });
  await closed.start(); await closed.close(); await settle(60);
  assert.equal(wakes.length, 3, 'close clears the wake timer');
});

test('keyed loops apply each key in order and keys in parallel', async () => {
  const commits = [];
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  const loops = new KeyedEventLoop({ key: event => event.user, initialState: user => ({ user, items: [] }),
    restore: user => user === 'carol' ? { state: { user, items: ['saved'] }, revision: 3, seenEventIds: ['old'] } : undefined,
    reduce: async (state, event) => { if (event.slow) await gate; return { ...state, items: [...state.items, event.item] }; },
    view: state => state.items.join(','), onCommit: (key, commit) => { commits.push(`${key}:${commit.revision}:${commit.event.item}`); } });
  const slow = loops.dispatch({ id: 'a1', kind: 'add', user: 'alice', item: 'x', slow: true });
  const queued = loops.dispatch({ id: 'a2', kind: 'add', user: 'alice', item: 'y' });
  const bob = await loops.dispatch({ id: 'b1', kind: 'add', user: 'bob', item: 'z' });
  assert.equal(bob.view, 'z', 'bob is not held up by alice');
  assert.deepEqual(commits, ['bob:1:z']);
  release();
  assert.equal((await slow).view, 'x'); assert.equal((await queued).view, 'x,y');
  assert.equal((await loops.dispatch({ id: 'old', kind: 'add', user: 'carol', item: 'again' })), null, 'restored event IDs are not reapplied');
  assert.equal((await loops.dispatch({ id: 'c1', kind: 'add', user: 'carol', item: 'new' })).view, 'saved,new');
  assert.deepEqual(commits, ['bob:1:z', 'alice:1:x', 'alice:2:y', 'carol:4:new']);
  assert.deepEqual(loops.keys().sort(), ['alice', 'bob', 'carol']);
  await loops.close();
});
