import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime, openFolder } from '../dist/index.js';
import { WikiWorkspace } from '../../applications/dist/wiki/index.js';
import { scriptedModel } from './support/natlang.mjs';

const profile = { model: 'fixture-model', source: 'merge-v2', seed: 17 };
const page = { id: 'guide', blocks: [
  { id: 'title', kind: 'prose', text: '# Install\nHow to install the engine.' },
  { id: 'intro', kind: 'prose', text: 'This guide explains the engine.' },
  { id: 'demo', kind: 'cell', text: 'return input.toUpperCase();', language: 'javascript', returns: 'string' },
] };

/**
 * The wiki's stages, scripted: each reply is the eval code a model might write for that stage's instructions. The
 * orchestrators (merge, block, maintain, cells) follow their instruction steps; the judgments are small closures over
 * what the stage shows. `script` overrides one stage.
 */
const STAGES = {
  merge: `
    const groups = {};
    for (const update of updates) (groups[update.block_id] ??= []).push(update);
    const outcomes = {};
    await Promise.all(base.blocks.filter(row => groups[row.id]).map(async row => { outcomes[row.id] = await block(row, groups[row.id], settings); }));
    return { blocks: base.blocks.map(row => outcomes[row.id]?.block ?? row),
      accounted: base.blocks.flatMap(row => outcomes[row.id]?.accounted ?? []),
      unresolved: base.blocks.flatMap(row => outcomes[row.id]?.unresolved ?? []) };`,
  block: `
    const ordered = [...updates].sort((a, b) => a.id.localeCompare(b.id));
    const list = await Promise.all(ordered.map(update => changes(block, update, settings)));
    const relations = [];
    for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++) {
      const a = list[i], b = list[j];
      let relation;
      if (['noop', 'format'].includes(a.kind) || ['noop', 'format'].includes(b.kind)) relation = 'compatible';
      else if (a.text === b.text) relation = 'redundant';
      else { const d = await decide(relate, a, b); relation = d.confidence >= 0.6 ? d.value : 'contradictory'; }
      relations.push({ a: a.update_id, b: b.update_id, relation });
    }
    const p = await plan(block.id, list, relations);
    const carriedIds = new Set(p.steps.filter(step => step.action !== 'conflict').flatMap(step => step.update_ids));
    const carried = list.filter(change => carriedIds.has(change.update_id));
    let attempt = { text: block.text, clear: true, lost: [] };
    if (p.steps.some(step => step.action === 'combine'))
      attempt = await refine.iterateOn({ text: '', clear: true, lost: null }, block, carried, p)
        .withLimit({ maxSteps: 3 }).until(a => !a.clear || (a.lost !== null && a.lost.length === 0)).catch(() => ({ text: block.text, clear: false, lost: [] }));
    const failed = !attempt.clear || attempt.lost.length > 0;
    const unresolved = [];
    for (const step of p.steps.filter(step => step.action === 'conflict'))
      for (const id of step.update_ids) unresolved.push({ update_id: id, block_id: block.id, alternatives: step.update_ids.map(other => ordered.find(u => u.id === other).text) });
    if (failed) for (const id of [...carriedIds].sort()) unresolved.push({ update_id: id, block_id: block.id, alternatives: [...carriedIds].sort().map(other => ordered.find(u => u.id === other).text) });
    return { block: { ...block, text: failed ? block.text : attempt.text }, accounted: ordered.map(u => u.id), unresolved };`,
  summarize: `
    const kind = update.text === block.text ? 'noop' : update.text.startsWith(block.text) ? 'extend' : 'rewrite';
    return { update_id: update.id, block_id: update.block_id, author: update.author, kind, intent: 'Make it read: ' + update.text,
      adds: kind === 'extend' ? [update.text.slice(block.text.length).trim()] : [update.text], removes: [], text: update.text };`,
  plan: `
    const parent = {};
    const find = x => { let root = x; for (let i = 0; i < changes.length; i++) if (parent[root] !== root) root = parent[root]; return root; };
    for (const change of changes) parent[change.update_id] = change.update_id;
    for (const r of relations) if (r.relation === 'contradictory') parent[find(r.a)] = find(r.b);
    const groups = {};
    for (const change of changes) (groups[find(change.update_id)] ??= []).push(change.update_id);
    const steps = [], conflicted = new Set(), covered = {};
    for (const ids of Object.values(groups)) if (ids.length > 1) { ids.sort(); ids.forEach(id => conflicted.add(id)); steps.push({ action: 'conflict', update_ids: ids, note: 'they disagree' }); }
    for (const r of relations) if (r.relation === 'redundant' && !conflicted.has(r.a) && !conflicted.has(r.b) && !covered[r.a] && !covered[r.b]) covered[r.b] = r.a;
    for (const [b, a] of Object.entries(covered)) steps.push({ action: 'covered', update_ids: [b], by: a, note: 'same meaning' });
    const combined = changes.map(change => change.update_id).filter(id => !conflicted.has(id) && !covered[id]);
    if (combined.length) steps.push({ action: 'combine', update_ids: combined, note: 'they fit together' });
    return { block_id, steps };`,
  refine: `
    let combined = changes.filter(change => plan.steps.find(step => step.action === 'combine').update_ids.includes(change.update_id));
    let problem = '';
    if (attempt.lost !== null) {
      if (!attempt.lost.length) return attempt;
      combined = changes.filter(change => combined.includes(change) || attempt.lost.includes(change.update_id));
      problem = 'The text does not carry: ' + changes.filter(change => attempt.lost.includes(change.update_id)).map(change => change.intent).join('; ');
    }
    const merged = block.kind === 'prose' ? await prose(block, combined, problem) : await code(block, combined, problem);
    if (!merged.clear) return { text: block.text, clear: false, lost: [] };
    const verdicts = await Promise.all(changes.map(async change => [change.update_id, await decide(honors, change, block.text, merged.text)]));
    return { text: merged.text, clear: true, lost: verdicts.filter(([, d]) => d.value !== 'honored').map(([id]) => id) };`,
  prose: `
    return { text: changes.length === 1 ? changes[0].text : [block.text, ...changes.map(change => change.adds.join(' '))].join(' ').trim(), clear: true, reason: '' };`,
  maintain: `
    const [sa, sb] = await Promise.all([scan(after), scan(before)]);
    const [oa, ob] = await Promise.all([outline(after, sa), outline(before, sb)]);
    const ids = after.blocks.map(row => row.id);
    const links = await Promise.all(sa.links.map(link => resolve(link, oa, ob, ids)));
    const byBlock = {};
    for (const link of links.filter(link => link.status === 'renamed')) (byBlock[link.block_id] ??= []).push(link);
    const repairs = Object.entries(byBlock).map(([id, renamed]) => {
      let text = after.blocks.find(row => row.id === id).text;
      for (const link of renamed) text = retarget(text, link.target, link.resolves_to);
      return { id: 'repair-' + id, block_id: id, base_revision: after.revision, author: 'wiki', text };
    });
    const verdicts = await staleness(after, delta(before, after), records, settings);
    return { structure: { revision: after.revision, outline: oa, links, cells: verdicts }, repairs };`,
  outline: `
    const sections = [];
    for (const row of page.blocks) {
      const heading = scan.headings.find(h => h.block_id === row.id);
      if (heading) sections.push({ id: heading.title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, ''), title: heading.title,
        level: heading.level, block_ids: [row.id], parent: null });
      else if (sections.length) sections.at(-1).block_ids.push(row.id);
    }
    return { sections, problems: [] };`,
  resolve: `
    const known = [...outline.sections.map(s => s.id), ...block_ids];
    if (known.includes(link.target)) return { block_id: link.block_id, target: link.target, status: 'ok', resolves_to: link.target, reason: 'it exists' };
    const old = previous.sections.find(s => s.id === link.target);
    const now = old && outline.sections.find(s => s.block_ids[0] === old.block_ids[0]);
    return now ? { block_id: link.block_id, target: link.target, status: 'renamed', resolves_to: now.id, reason: 'the section was renamed' }
      : { block_id: link.block_id, target: link.target, status: 'broken', resolves_to: null, reason: 'nothing has that name' };`,
  judge: `
    const deps = await Promise.all(records.map(record => depends(page.blocks.find(row => row.id === record.block_id), page)));
    return records.map((record, i) => {
      const hit = deps[i].blocks.find(id => delta.changed.includes(id) || delta.removed.includes(id));
      return delta.changed.includes(record.block_id) ? { block_id: record.block_id, status: 'stale', reason: 'its source changed' }
        : hit ? { block_id: record.block_id, status: 'stale', reason: hit + ' changed' }
        : { block_id: record.block_id, status: 'fresh', reason: 'nothing it reads changed' };
    });`,
  depends: `
    const mentioned = [...cell.text.matchAll(/\\[\\[([A-Za-z0-9_-]+)\\]\\]/g)].map(m => m[1]);
    return { cell: cell.id, blocks: mentioned.filter(id => page.blocks.some(row => row.id === id && row.kind === 'prose')), cells: mentioned.filter(id => page.blocks.some(row => row.id === id && row.kind === 'cell')), files: false };`,
  cells: `
    const refused = [], runs = new Map();
    for (const request of requests) {
      const cell = page.blocks.find(row => row.id === request.block_id && row.kind === 'cell');
      const reason = !cell ? 'no such cell' : (page.unresolved ?? []).some(c => c.block_id === cell.id) ? 'its source is an unresolved conflict'
        : request.origin === 'auto' && cell.language === 'natlang' ? 'automatic runs use javascript cells only'
        : request.origin === 'auto' && fresh.includes(cell.id) ? 'its result is fresh' : null;
      if (reason) refused.push({ block_id: request.block_id, reason });
      else if (!runs.has(request.block_id) || request.origin === 'user') runs.set(request.block_id, { block_id: request.block_id, input: request.input });
    }
    const list = [...runs.values()];
    const deps = await Promise.all(list.map(run => depends(page.blocks.find(row => row.id === run.block_id), page)));
    const level = {};
    for (let round = 0; round < list.length; round++) list.forEach((run, i) => {
      const among = deps[i].cells.filter(id => runs.has(id));
      if (level[run.block_id] === undefined && among.every(id => level[id] !== undefined)) level[run.block_id] = among.length ? Math.max(...among.map(id => level[id])) + 1 : 0;
    });
    const batches = [];
    for (const run of list) { if (level[run.block_id] === undefined) refused.push({ block_id: run.block_id, reason: 'the cells read each other' }); else (batches[level[run.block_id]] ??= []).push(run); }
    return { batches: batches.filter(Boolean), refused };`,
};

/** A scripted wiki model. `overrides` replaces stage replies; `verdict(change, merged)` answers honors; `decisions` counts decisions. */
function wikiModel({ overrides = {}, verdict = () => 'honored' } = {}) {
  const seen = [];
  const model = scriptedModel(opening => {
    const stage = /inside this call: (\w+)\(/.exec(opening)?.[1];
    seen.push(stage);
    return overrides[stage] ?? STAGES[stage] ?? null;
  });
  const decide = async ({ messages, options }) => {
    const text = JSON.stringify(messages).replace(/\\"/g, '"').replace(/\\\\/g, '\\');
    const stage = /inside this call: (\w+)\(/.exec(text)?.[1];
    seen.push(stage);
    let winner;
    if (stage === 'relate') winner = /tabs/.test(text) && /spaces/.test(text) ? 'contradictory' : 'compatible';
    else {
      const id = /update_id: "([^"]*)"/.exec(text)?.[1], merged = /merged: string = "([^"]*)"/.exec(text)?.[1] ?? '';
      winner = verdict(id, merged);
    }
    return { log_probs: options.map(option => Math.log(option === JSON.stringify(winner) ? 0.9 : 0.1 / (options.length - 1))) };
  };
  return { driver: Object.assign(model.driver, { decide }), seen, openings: model.openings };
}
const wikiRuntime = model => createNatlangRuntime({ model: { driver: model.driver, maxTurns: 6 }, seed: { mode: 'derived', root: profile.seed } });
const update = (base, id, block_id, text, author = id) => ({ id, block_id, base_revision: base.revision, author, text });
const count = (seen, name) => seen.filter(stage => stage === name).length;

test('natlang reconciles two edits: summaries, relations, a plan, a merged text checked against both intents', async () => {
  const model = wikiModel();
  const wiki = new WikiWorkspace(page, { profile, runtime: wikiRuntime(model) });
  const base = wiki.snapshot();
  const report = await wiki.merge(base, [
    update(base, 'b', 'intro', 'This guide explains the engine. Add a concise example.'),
    update(base, 'a', 'intro', 'This guide explains the engine. Clarify that it is small.')]);
  assert.equal(report.status, 'merged', report.detail);
  assert.equal(report.detail, '');
  assert.match(report.page.blocks[1].text, /concise example/);
  assert.match(report.page.blocks[1].text, /Clarify that it is small/);
  assert.deepEqual(model.seen.filter(stage => stage !== 'honors'),
    ['merge', 'block', 'summarize', 'summarize', 'relate', 'plan', 'refine', 'prose', 'maintain', 'outline', 'outline'],
    'with no recorded result there is nothing to judge');
  assert.equal(count(model.seen, 'honors'), 2, 'each intent is checked against the merged text');
  assert.deepEqual(wiki.drainEvents().map(event => event.operation), ['wiki.merge', 'wiki.maintain']);
  assert.equal(report.structure.revision, report.page.revision);
  assert.deepEqual(report.structure.outline.sections.map(row => row.id), ['install']);
  const cell = await wiki.runCell('demo', 'hello');
  assert.equal(cell.value_text, '"HELLO"');
});

test('contradictory edits stay a conflict with every alternative and the base text; compatible edits to other blocks merge', async () => {
  const model = wikiModel();
  const wiki = new WikiWorkspace(page, { profile, runtime: wikiRuntime(model) });
  const base = wiki.snapshot();
  const report = await wiki.merge(base, [
    update(base, 'a', 'intro', 'Indent with tabs.'), update(base, 'b', 'intro', 'Indent with spaces.'),
    update(base, 'c', 'title', '# Install\nHow to install the engine. Needs Node.')]);
  assert.equal(report.status, 'unresolved');
  assert.equal(report.page.blocks[1].text, 'This guide explains the engine.', 'the conflicted block keeps the base text');
  assert.match(report.page.blocks[0].text, /Needs Node/);
  assert.deepEqual(report.page.unresolved.map(row => [row.update_id, row.block_id, row.alternatives]),
    [['a', 'intro', ['Indent with tabs.', 'Indent with spaces.']], ['b', 'intro', ['Indent with tabs.', 'Indent with spaces.']]]);
});

test('a merged text that loses an intent is written again with the problem; an intent still lost becomes a conflict', async () => {
  let round = 0;
  const model = wikiModel({
    verdict: (id, merged) => (round === 0 && id === 'b' && !merged.includes('Mention the CLI')) ? 'lost' : 'honored',
    overrides: { prose: `
      const hasProblem = typeof problem === 'string' && problem.length > 0;
      return { text: [block.text, ...changes.map(change => change.adds.join(' '))].join(' ') + (hasProblem ? ' Mention the CLI.' : ''), clear: true, reason: '' };` } });
  const wiki = new WikiWorkspace(page, { profile, runtime: wikiRuntime(model) });
  const base = wiki.snapshot();
  const updates = [update(base, 'a', 'intro', 'This guide explains the engine. First.'), update(base, 'b', 'intro', 'This guide explains the engine. Second.')];
  const report = await wiki.merge(base, updates);
  assert.equal(report.status, 'merged', report.detail);
  assert.equal(count(model.seen, 'prose'), 2, 'written, found lost, written again');
  assert.match(report.page.blocks[1].text, /Mention the CLI/);

  // An intent that stays lost: the whole group is left for a person and the base text is kept.
  const stubborn = wikiModel({ verdict: id => id === 'b' ? 'lost' : 'honored' });
  const other = new WikiWorkspace(page, { profile, runtime: wikiRuntime(stubborn) });
  const second = await other.merge(other.snapshot(), updates.map(row => ({ ...row, base_revision: other.snapshot().revision })));
  assert.equal(second.status, 'unresolved');
  assert.equal(second.page.blocks[1].text, 'This guide explains the engine.');
  assert.deepEqual(second.page.unresolved.map(row => row.update_id), ['a', 'b']);
});

test('settings pick the crisp implementation of a hot-path policy without calling the model for it', async () => {
  const model = wikiModel();
  const wiki = new WikiWorkspace(page, { profile, runtime: wikiRuntime(model), settings: { changes: 'crisp', staleness: 'crisp' } });
  await wiki.runCell('demo', 'x');
  const base = wiki.snapshot();
  const report = await wiki.merge(base, [update(base, 'a', 'intro', 'This guide explains the engine. More.'), update(base, 'b', 'intro', 'This guide explains the engine.  ')]);
  assert.equal(report.status, 'merged', report.detail);
  assert.equal(count(model.seen, 'summarize'), 0, 'exact change summaries need no model');
  assert.equal(count(model.seen, 'judge'), 0, 'the crisp staleness rule needs no model');
  assert.equal(wiki.output('demo'), null, 'the crisp rule drops every result after an edit');
  assert.deepEqual(report.structure.cells, [{ block_id: 'demo', status: 'stale', reason: 'the page changed' }]);
});

test('a cell result survives an edit to blocks it does not read and is dropped when one it reads changes', async () => {
  const withReads = { ...page, blocks: [...page.blocks, { id: 'count', kind: 'cell', language: 'javascript', returns: 'number', text: '// counts the words of [[intro]]\nreturn input.split(" ").length;' }] };
  const model = wikiModel();
  const wiki = new WikiWorkspace(withReads, { profile, runtime: wikiRuntime(model) });
  await wiki.runCell('demo', 'x');
  await wiki.runCell('count', 'a b c');
  let base = wiki.snapshot();
  const first = await wiki.merge(base, [update(base, 'a', 'title', '# Install\nHow to install the engine quickly.')]);
  assert.equal(first.status, 'merged', first.detail);
  assert.deepEqual(first.structure.cells.map(row => [row.block_id, row.status]), [['demo', 'fresh'], ['count', 'fresh']]);
  assert.equal(wiki.output('demo').page_revision, first.page.revision, 'the surviving result is re-keyed to the new revision');
  assert.equal(wiki.output('count').value_text, '3');
  base = wiki.snapshot();
  const second = await wiki.merge(base, [update(base, 'b', 'intro', 'This guide explains the whole engine.')]);
  assert.deepEqual(second.structure.cells.map(row => [row.block_id, row.status, row.reason]),
    [['demo', 'fresh', 'nothing it reads changed'], ['count', 'stale', 'intro changed']]);
  assert.equal(wiki.output('count'), null);
  assert.ok(wiki.output('demo'));
});

test('renaming a section leaves a link repair as an ordinary update; the structure lists sections and links', async () => {
  const linked = { id: 'links', blocks: [
    { id: 'head', kind: 'prose', text: '# Setup' },
    { id: 'body', kind: 'prose', text: 'Run it. See [[setup]] and [[missing]] and [[head]].' },
  ] };
  const model = wikiModel();
  const wiki = new WikiWorkspace(linked, { profile, runtime: wikiRuntime(model) });
  const base = wiki.snapshot();
  const report = await wiki.merge(base, [update(base, 'r', 'head', '# Installation')]);
  assert.equal(report.status, 'merged', report.detail);
  assert.deepEqual(report.structure.outline.sections.map(row => row.id), ['installation']);
  assert.deepEqual(report.structure.links.map(row => [row.target, row.status, row.resolves_to]),
    [['setup', 'renamed', 'installation'], ['missing', 'broken', null], ['head', 'ok', 'head']]);
  assert.equal(report.repairs.length, 1);
  assert.equal(report.repairs[0].base_revision, report.page.revision);
  assert.equal(report.repairs[0].text, 'Run it. See [[installation]] and [[missing]] and [[head]].');
  const next = await wiki.merge(wiki.snapshot(), report.repairs);
  assert.equal(next.status, 'merged', 'the repair merges like any edit');
  assert.match(next.page.blocks[1].text, /\[\[installation\]\]/);
});

test('the cell policy refuses, orders by dependency and runs the plan batch by batch', async () => {
  const cells = { id: 'cells', blocks: [
    { id: 'total', kind: 'cell', language: 'javascript', returns: 'string', text: '// sums [[part]]\nreturn "total:" + input;' },
    { id: 'part', kind: 'cell', language: 'javascript', returns: 'string', text: 'return "part:" + input;' },
    { id: 'ask', kind: 'cell', language: 'natlang', returns: 'string', text: 'Answer the question in input.' },
    { id: 'note', kind: 'prose', text: 'Not a cell.' },
  ] };
  const model = wikiModel({ overrides: { ask: 'return "answer";' } });
  const wiki = new WikiWorkspace(cells, { profile, runtime: wikiRuntime(model) });
  cells.blocks[0].text = cells.blocks[0].text.replace('[[part]]', '[[part]]');
  const { plan, results } = await wiki.schedule([
    { block_id: 'total', input: 'T', origin: 'user' }, { block_id: 'part', input: 'P', origin: 'auto' },
    { block_id: 'ask', input: 'Q', origin: 'auto' }, { block_id: 'note', input: 'N', origin: 'user' }, { block_id: 'ghost', input: 'G', origin: 'user' }]);
  assert.deepEqual(plan.batches.map(batch => batch.map(run => run.block_id)), [['part'], ['total']]);
  assert.deepEqual(plan.refused.map(row => [row.block_id, row.reason]).sort(),
    [['ask', 'automatic runs use javascript cells only'], ['ghost', 'no such cell'], ['note', 'no such cell']]);
  assert.deepEqual(Object.fromEntries(Object.entries(results).map(([id, row]) => [id, row.value_text])), { part: '"part:P"', total: '"total:T"' });
  assert.equal(count(model.seen, 'ask'), 0, 'the refused natlang cell never ran');
  const again = await wiki.schedule([{ block_id: 'part', input: 'P', origin: 'auto' }]);
  assert.deepEqual(again.plan.refused, [{ block_id: 'part', reason: 'its result is fresh' }]);
});

test('natlang cells get the project folder, read fresh for each run', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-wiki-files-'));
  writeFileSync(join(folder, 'note.txt'), 'first note');
  const model = scriptedModel(() => 'return (await files.file("note.txt").readText())');
  const wiki = new WikiWorkspace({ id: 'files', blocks: [
    { id: 'quick', kind: 'cell', language: 'javascript', returns: 'string', text: 'return input;' },
    { id: 'natural', kind: 'cell', language: 'natlang', returns: 'string', text: 'Read the project note and return its text.' },
  ] }, { profile, runtime: wikiRuntime(model), files: () => openFolder(folder).root() });
  try {
    assert.equal((await wiki.runCell('quick', 'quick value')).value_text, '"quick value"');
    assert.equal(model.openings.length, 0, 'JavaScript cells do not call the model');
    const first = await wiki.runCell('natural', 'question');
    assert.equal(first.value_text, '"first note"');
    assert.ok(first.trace_events > 0 && wiki.trace('natural').length === 1);
    writeFileSync(join(folder, 'note.txt'), 'second note');
    assert.equal((await wiki.runCell('natural', 'question')).value_text, '"second note"');
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('mismatched profiles, missing updates, uncompilable cells and foreign conflicts are rejected by the commit', () => {
  const wiki = new WikiWorkspace(page, { profile, runtime: createNatlangRuntime() });
  const base = wiki.snapshot();
  const edit = { id: 'x', block_id: 'intro', base_revision: base.revision, author: 'editor', text: 'Clarify source.' };
  assert.equal(wiki.prepare(base, [edit], { ...profile, seed: 18 }).valid, false);
  const prepared = wiki.prepare(base, [edit], profile);
  assert.equal(wiki.publish(base, prepared, { blocks: base.blocks, accounted: [], unresolved: [] }, profile).status, 'rejected');
  assert.equal(wiki.publish(base, prepared, { blocks: [base.blocks[0], base.blocks[1], { ...base.blocks[2], returns: 'Any' }],
    accounted: ['x'], unresolved: [] }, profile).status, 'rejected');
  assert.equal(wiki.publish(base, prepared, { blocks: [base.blocks[0], base.blocks[1], { ...base.blocks[2], text: 'return (' }],
    accounted: ['x'], unresolved: [] }, profile).status, 'rejected', 'a JavaScript cell must compile');
  assert.equal(wiki.publish(base, prepared, { blocks: [base.blocks[1], base.blocks[0], base.blocks[2]], accounted: ['x'], unresolved: [] }, profile).status,
    'rejected', 'block order is the base order');
  assert.equal(wiki.publish(base, prepared, { blocks: base.blocks, accounted: ['x'],
    unresolved: [{ update_id: 'x', block_id: 'title', alternatives: [] }] }, profile).status, 'rejected', 'a conflict names its update\'s block');
  const good = wiki.publish(base, prepared, { blocks: [base.blocks[0], { ...base.blocks[1], text: 'Clarified source.' }, base.blocks[2]],
    accounted: ['x'], unresolved: [] }, profile);
  assert.equal(good.status, 'merged');
  assert.equal(wiki.prepare(base, [edit], profile).valid, false, 'the old base is stale after a merge');
});

test('a natlang cell that finishes after the page changed is stale and not shown', async () => {
  let release, entered;
  const started = new Promise(resolve => { entered = resolve; });
  const gate = new Promise(resolve => { release = resolve; });
  const model = scriptedModel(async () => { entered(); await gate; return 'return "Old answer"'; });
  const wiki = new WikiWorkspace({ id: 'live', blocks: [
    { id: 'note', kind: 'prose', text: 'Old page' },
    { id: 'answer', kind: 'cell', language: 'natlang', returns: 'string', text: 'Return a short answer to input.' },
  ] }, { profile, runtime: wikiRuntime(model) });
  const running = wiki.runCell('answer', 'question');
  await started;
  const base = wiki.snapshot();
  const prepared = wiki.prepare(base, [{ id: 'u1', block_id: 'note', base_revision: base.revision, author: 'editor', text: 'New page' }], profile);
  assert.equal(wiki.publish(base, prepared, { blocks: [{ ...base.blocks[0], text: 'New page' }, base.blocks[1]],
    accounted: ['u1'], unresolved: [] }, profile).status, 'merged');
  release();
  assert.equal((await running).status, 'stale');
  assert.equal(wiki.output('answer'), null);
});

test('wiki settings are crisp, nl or shadow; the older spelling natlang means nl', () => {
  const page = { id: 'page', blocks: [{ id: 'intro', kind: 'prose', text: 'Hello.' }] };
  const profile = { model: 'm', source: 's', seed: 1 };
  const wiki = settings => new WikiWorkspace(page, { profile, runtime: createNatlangRuntime({ model: scriptedModel(() => null).driver }), ...settings ? { settings } : {} }).settings;
  assert.deepEqual(wiki(), { changes: 'nl', staleness: 'nl' });
  assert.deepEqual(wiki({ changes: 'natlang', staleness: 'shadow' }), { changes: 'nl', staleness: 'shadow' });
  assert.deepEqual(wiki({ changes: 'crisp' }), { changes: 'crisp', staleness: 'nl' });
});
