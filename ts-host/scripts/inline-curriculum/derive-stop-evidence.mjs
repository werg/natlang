// Give expected-stop curriculum rows that declare no decisive observation a runtime-graph marker derived from their
// own world, and verify it through the real collector (no model). Without a marker an immediate give-up earns the
// stop's reward (collector stop_evidence, AGENTS.md "Task contracts and reward validity").
//
//   node derive-stop-evidence.mjs IN_CASES.jsonl OUT_DIR
//
// For each expected-stop row (semantics.operation 'blocked') of a supported family whose curriculum.decisive is empty:
//   1. replay curriculum.reference through the collector and read the marker from the run's own execution graph
//      (actor_greenhouse heater_broken: the third greenhouse.read() result; iterate_schedule_repair impossible: the
//      first iterateOn step whose state hash repeats an earlier one);
//   2. require it to equal the marker the generator derives from the world alone (inline-curriculum/investigate.mjs,
//      actor.mjs), so the published marker is the world's, not an artifact of one replay;
//   3. with the marker: verifyCases must admit the reference, its outcome must be accepted with stop_evidence, and two
//      negative controls must be rejected on stop_evidence: an immediate give-up, and a stop just before the
//      observation (two greenhouse readings; a repair limited to fewer steps than the cycle needs).
// Every other row passes through byte-identical. Every expected-stop row that already declares markers gets the
// immediate give-up control too. Rows that fail any step keep no marker and are listed under `unresolved` with the
// reason; the caller excludes them explicitly. Writes OUT_DIR/cases.jsonl and OUT_DIR/stop-evidence.json.
import fs from 'node:fs';
import crypto from 'node:crypto';
import { replayReference, verifyCases } from '../../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';
import { greenhouseControl } from './investigate.mjs';
import { iterateSchedule } from './actor.mjs';
import { hinted } from './lib.mjs';

const [input, out] = process.argv.slice(2);
if (!input || !out) throw new Error('usage: derive-stop-evidence.mjs IN_CASES.jsonl OUT_DIR');
if (fs.existsSync(out + '/cases.jsonl')) throw new Error(`refusing to overwrite ${out}/cases.jsonl`);
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

const giveUp = ['return_result', { status: 'failed', reason: 'I cannot complete this task.' }];
const FAMILIES = {
  actor_greenhouse: {
    variant: 'heater_broken', build: greenhouseControl, shape: /^greenhouse(\d+)$/,
    fromGraph: graph => {
      const third = graph.find(node => node.kind === 'effect' && node.service === 'greenhouse' && node.method === 'read' &&
        node.result?.minute === 30);
      return third ? JSON.stringify(third.result) : undefined;
    },
    early: ['eval', { code: 'greenhouse.set({ heater: true, vent: false });\ngreenhouse.read();\ngreenhouse.read().temp' }],
  },
  iterate_schedule_repair: {
    variant: 'impossible', build: iterateSchedule, shape: /^schedule(\d+)$/,
    fromGraph: graph => {
      const seen = new Set();
      for (const node of graph) {
        if (node.kind !== 'iteration_step' || (node.phase !== 'initial' && node.phase !== 'step')) continue;
        if (seen.has(node.state_hash)) return `"phase":"step","step":${node.step},"state_hash":"${node.state_hash}"`;
        seen.add(node.state_hash);
      }
      return undefined;
    },
    early: ['eval', { code: 'const fixed = await resolve.iterateOn(schedule).withLimit({maxSteps: 5}).until(s => conflicts(s) === 0);\nfixed' }],
  },
};

// The generator's world-only marker for a row: rebuild its shape with the row's seed and take the same variant.
const generated = new Map();
function generatorDecisive(row) {
  const c = row.curriculum, spec = FAMILIES[c.family];
  const seed = Number(/^s(\d+)$/.exec(row.generation?.seed ?? '')?.[1]);
  const index = Number(spec.shape.exec(c.shape)?.[1]);
  if (!Number.isInteger(seed) || !Number.isInteger(index)) throw new Error(`no seed/shape index for ${row.id}`);
  const key = `${c.family}:${seed}:${index}`;
  if (!generated.has(key)) generated.set(key, spec.build(seed, index, 'train'));
  const base = generated.get(key).find(item => item.curriculum.variant === c.variant);
  const record = c.hinted_of ? hinted(base, c.hint) : base;
  // The rebuilt world must be the row's world: same files, services, inputs and expected outcome.
  for (const field of ['files', 'services', 'inputs', 'expected', 'operation'])
    if (JSON.stringify(record.semantics[field]) !== JSON.stringify(row.semantics[field]))
      throw new Error(`rebuilt ${key} ${c.variant} differs from the row in semantics.${field}`);
  return record.curriculum.decisive;
}

const replay = async (row, root) => {
  const copy = structuredClone(row);
  if (root) copy.curriculum.reference.root = root;
  return (await replayReference(copy, TOOLS_PROMPT)).run.outcome;
};
const control = outcome => ({ accepted: outcome.accepted, honest_stop: outcome.checks.honest_stop,
  stop_evidence: outcome.checks.stop_evidence ?? null, rejection_reasons: outcome.rejection_reasons });

const lines = fs.readFileSync(input, 'utf8').split('\n');
if (lines.at(-1) === '') lines.pop();
const output = [], derived = [], unresolved = [], existing = [];
let processed = 0;
for (const [index, line] of lines.entries()) {
  const row = JSON.parse(line), c = row.curriculum;
  if (row.semantics?.operation !== 'blocked') { output.push(line); continue; }
  const spec = FAMILIES[c?.family];
  if (c.decisive?.length) {
    // Already declared: the immediate give-up control must be rejected and the reference accepted.
    const reference = await replay(row), quit = await replay(row, [giveUp]);
    const ok = reference.accepted === true && reference.checks.stop_evidence === true && quit.accepted === false &&
      quit.checks.stop_evidence === false;
    existing.push({ id: row.id, line: index + 1, family: c.family, variant: c.variant, ok, reference: control(reference),
      give_up: control(quit) });
    output.push(line);
  } else if (!spec || c.variant !== spec.variant) {
    unresolved.push({ id: row.id, line: index + 1, family: c?.family, variant: c?.variant, reason: 'no supported derivation' });
    output.push(line);
  } else {
    const problems = [];
    let marker, patched, result = { id: row.id, line: index + 1, family: c.family, variant: c.variant };
    try {
      const plain = await replay(row);
      marker = spec.fromGraph(plain.execution_graph ?? []);
      if (!marker) problems.push('reference execution graph has no observation to mark');
      const fromWorld = generatorDecisive(row);
      if (fromWorld.length !== 1 || fromWorld[0].marker !== marker)
        problems.push(`generator marker ${JSON.stringify(fromWorld.map(item => item.marker))} differs from the replayed ${JSON.stringify(marker)}`);
      if (!problems.length) {
        patched = structuredClone(row);
        patched.curriculum.decisive = fromWorld;
        const [verified] = await verifyCases([patched], TOOLS_PROMPT);
        const reference = await replay(patched), quit = await replay(patched, [giveUp]);
        const early = await replay(patched, [spec.early, c.reference.root.at(-1)]);
        Object.assign(result, { marker, verify_cases: { ok: verified.ok, problems: verified.problems },
          reference: control(reference), give_up: control(quit), early_stop: control(early) });
        if (!verified.ok) problems.push(`verifyCases: ${verified.problems.join('; ')}`);
        if (reference.accepted !== true || reference.checks.stop_evidence !== true) problems.push('reference not accepted with the marker');
        for (const [name, outcome] of [['give_up', quit], ['early_stop', early]])
          if (outcome.accepted !== false || outcome.checks.stop_evidence !== false) problems.push(`${name} not rejected on stop evidence`);
      }
    } catch (error) { problems.push(String(error?.stack ?? error).slice(0, 600)); }
    if (problems.length) { unresolved.push({ ...result, reason: problems.join('; ') }); output.push(line); }
    else { derived.push(result); output.push(JSON.stringify(patched)); }
  }
  if (++processed % 50 === 0) console.log(JSON.stringify({ blocked_rows: processed, derived: derived.length, unresolved: unresolved.length }));
}

fs.mkdirSync(out, { recursive: true });
const body = output.map(line => line + '\n').join('');
fs.writeFileSync(out + '/cases.jsonl', body);
const count = (items, key) => items.reduce((acc, item) => ({ ...acc, [item[key]]: (acc[item[key]] ?? 0) + 1 }), {});
const report = {
  schema: 'natlang.curriculum-stop-evidence-derivation/1',
  input: { path: input, sha256: sha(fs.readFileSync(input)), rows: lines.length },
  output: { cases: 'cases.jsonl', sha256: sha(body), rows: output.length },
  rule: {
    actor_greenhouse: 'heater_broken: marker = JSON of the third greenhouse.read() result (minute 30), taken from the reference ' +
      'execution graph and equal to the generator\'s world-only value; source runtime_graph',
    iterate_schedule_repair: 'impossible: marker = `"phase":"step","step":K,"state_hash":"H"` for the first iterateOn step K ' +
      'whose state hash H repeats an earlier state (the repair cycle), from the reference graph and equal to the generator\'s ' +
      'world-only value; source runtime_graph',
  },
  controls: { give_up: giveUp, early_stop: Object.fromEntries(Object.entries(FAMILIES).map(([name, spec]) => [name, spec.early])) },
  counts: { derived: derived.length, derived_by_family: count(derived, 'family'), unresolved: unresolved.length,
    unresolved_by_family: count(unresolved, 'family'), existing_markers: existing.length,
    existing_markers_ok: existing.filter(item => item.ok).length },
  derived, unresolved, existing,
};
fs.writeFileSync(out + '/stop-evidence.json', JSON.stringify(report, null, 1) + '\n');
console.log(JSON.stringify(report.counts));
