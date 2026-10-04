#!/usr/bin/env node
/** Build deterministic, uniquely solvable finite-CSP skill episodes. No model calls or external task data. */
import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { validateEpisode } from '../../dist/skills/episode.js';
import { cspProgressBound, solveFiniteCsp } from '../../dist/skills/csp-objective.js';

const sha = value => createHash('sha256').update(typeof value === 'string' ? value : JSON.stringify(value)).digest('hex');
const TEMPLATE_VERSION = 'csp-templates-v1';
const VARIANTS = ['generic', 'description-tuned', 'body-tuned'];
const BASIC_BODY = `Treat the listed domains and every written rule as binding. Keep a table of remaining values for each variable. Apply the clues repeatedly; reject any tentative choice that makes a rule impossible. If needed, branch on a variable with few remaining values and backtrack when a contradiction appears. Return a complete assignment and check every rule before answering.`;
const EXPERT_BODIES = {
  'order-schedule': `Model each task's slot as a variable. Propagate precedence bounds and the all-different rule before branching. Choose a task with the fewest remaining slots; try slots in order, propagate again, and backtrack on a contradiction. At the end, verify every precedence clue and that each slot is used once.`,
  'logic-grid': `Use one variable per named person and propagate the one-to-one assignment rule. For each pair clue, remove any value pair not in its compatibility table. Revisit affected variables after each removal; branch on the smallest remaining domain and backtrack if any domain empties. Check the one-to-one rule and every pair table in the final mapping.`,
  'resource-placement': `Treat each station number as a resource assignment. Propagate all-different and subset-sum clues together: after a choice, compute the minimum and maximum attainable sum from remaining domains and reject impossible totals. Branch on the smallest remaining domain, backtrack on contradiction, then recompute every displayed total.`,
  'latin-grid': `Represent each cell as a variable. Remove symbols already fixed in its row or column, and apply each given clue. Choose an empty cell with the fewest candidates, try a candidate, propagate to its row and column, and backtrack if a domain empties. Verify every row and column contains each symbol exactly once.`,
  'nonogram-grid': `For each row and column, enumerate binary line patterns that match its run clues. Intersect those patterns with current cell domains, propagate forced cells across crossing lines, then branch on the least flexible cell if needed. Reject contradictions and verify all row and column runs from the completed grid.`,
};
const DESCRIPTIONS = {
  'order-schedule': 'Use for assigning distinct time slots to tasks when precedence clues constrain their order.',
  'logic-grid': 'Use for one-to-one matching puzzles with named entities, finite choices, and pairwise compatibility clues.',
  'resource-placement': 'Use for assigning distinct resource stations when group totals constrain the placement.',
  'latin-grid': 'Use for completing a finite symbol grid with row and column uniqueness plus visible givens.',
  'nonogram-grid': 'Use for binary picture-grid puzzles whose row and column run lengths constrain each cell.',
};

function rng(seed) {
  let state = seed >>> 0;
  return () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 2 ** 32; };
}
function shuffled(items, random) {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [out[i], out[j]] = [out[j], out[i]]; }
  return out;
}
function permutations(values) {
  const out = [];
  const visit = (prefix, rest) => { if (!rest.length) { out.push(prefix); return; }
    for (let i = 0; i < rest.length; i++) visit([...prefix, rest[i]], [...rest.slice(0, i), ...rest.slice(i + 1)]); };
  visit([], values); return out;
}
function runs(values) {
  const output = []; let n = 0;
  for (const v of values) { if (v) n++; else if (n) { output.push(n); n = 0; } }
  if (n) output.push(n); return output;
}
function varMap(ids, domain) { return ids.map(id => ({ id, domain: [...domain] })); }
function solveUnique(instance, maxNodes = 250_000) {
  try {
    const result = solveFiniteCsp(instance, { maxNodes, maxSolutions: 2 });
    return result.status === 'unique' ? { solution: result.solutions[0], nodes: result.nodes } : { status: result.status, nodes: result.nodes };
  } catch (error) { return { status: 'invalid_or_capped', error: error.message }; }
}
function sameAssignment(a, b) {
  return a && b && Object.keys(a).length === Object.keys(b).length && Object.keys(a).every(key => a[key] === b[key]);
}

function orderSchedule(seed) {
  const random = rng(seed), names = shuffled(['Aster', 'Birch', 'Cedar', 'Dune', 'Elm'], random);
  const slots = shuffled([1, 2, 3, 4, 5], random), solution = Object.fromEntries(names.map((name, i) => [name, slots[i]]));
  const cluePairs = shuffled(names.flatMap((a, i) => names.slice(i + 1).flatMap(b =>
    solution[a] < solution[b] ? [[a, b]] : [[b, a]])), random);
  const constraints = [{ kind: 'allDifferent', vars: names }], accepted = [];
  let current;
  for (const [before, after] of cluePairs) {
    constraints.push({ kind: 'order', before, after, op: 'lt' }); accepted.push([before, after]);
    current = { schema: 'natlang.finite-csp/1', template: 'order-schedule-v1',
      narrative: `Schedule five jobs in distinct slots 1 through 5. ${accepted.map(([a, b]) => `${a} must be scheduled before ${b}.`).join(' ')}`,
      variables: varMap(names, [1, 2, 3, 4, 5]), constraints: structuredClone(constraints) };
    const check = solveUnique(current);
    if (check.solution) return { instance: current, solution: { assignment: check.solution }, nodes: check.nodes };
    if (check.status === 'limit' || check.status === 'invalid_or_capped') throw Error(`order template cap ${check.status}`);
  }
  throw Error('failed to construct a unique schedule puzzle');
}

function logicGrid(seed) {
  const random = rng(seed), people = shuffled(['Ivo', 'Kira', 'Lena', 'Miro', 'Nia'], random);
  const colors = shuffled(['amber', 'blue', 'coral', 'jade', 'violet'], random);
  const solution = Object.fromEntries(people.map((person, i) => [person, colors[i]]));
  const pairClues = shuffled(people.flatMap((a, i) => people.slice(i + 1).map(b => [a, b])), random);
  const constraints = [{ kind: 'allDifferent', vars: people }], shown = [];
  for (const [a, b] of pairClues) {
    const tuples = [];
    for (const av of colors) for (const bv of colors) {
      if (av === bv) continue;
      const isTarget = av === solution[a] && bv === solution[b];
      if (isTarget || random() < 0.32) tuples.push([av, bv]);
    }
    // Tuple order is aligned to the declared variables, which are the displayed person order.
    constraints.push({ kind: 'allowedTuples', vars: [a, b], tuples });
    shown.push({ a, b, tuples });
    const instance = { schema: 'natlang.finite-csp/1', template: 'logic-grid-pair-compat-v1',
      narrative: `Match each person to one distinct color: ${people.join(', ')}; colors are ${colors.join(', ')}. For each pair below, only the listed color pairs are compatible (first color belongs to the first named person): ${shown.map(c => `${c.a}/${c.b}: ${c.tuples.map(t => `${t[0]}+${t[1]}`).join(', ')}.`).join(' ')}`,
      variables: varMap(people, colors), constraints: structuredClone(constraints) };
    const check = solveUnique(instance);
    if (check.solution) return { instance, solution: { assignment: check.solution }, nodes: check.nodes };
    if (check.status === 'limit' || check.status === 'invalid_or_capped') throw Error(`logic-grid template cap ${check.status}`);
  }
  throw Error('failed to construct a unique logic-grid puzzle');
}

function resourcePlacement(seed) {
  const random = rng(seed), crews = shuffled(['Atlas', 'Beacon', 'Comet', 'Delta', 'Echo'], random);
  const stations = [1, 2, 3, 4, 5], values = shuffled(stations, random), solution = Object.fromEntries(crews.map((id, i) => [id, values[i]]));
  const subsets = shuffled(crews.flatMap((a, i) => crews.slice(i + 1).map(b => [a, b])), random), constraints = [
    { kind: 'allDifferent', vars: crews },
  ], clues = [];
  for (const [a, b] of subsets) {
    const value = solution[a] + solution[b];
    constraints.push({ kind: 'sum', vars: [a, b], op: 'eq', value }); clues.push([a, b, value]);
    const instance = { schema: 'natlang.finite-csp/1', template: 'resource-sums-v1',
      narrative: `Assign each crew one distinct station number from 1 through 5. These totals are known: ${clues.map(([x, y, total]) => `${x} and ${y} together have station numbers summing to ${total}.`).join(' ')}`,
      variables: varMap(crews, stations), constraints: structuredClone(constraints) };
    const check = solveUnique(instance);
    if (check.solution) return { instance, solution: { assignment: check.solution }, nodes: check.nodes };
    if (check.status === 'limit' || check.status === 'invalid_or_capped') throw Error(`resource template cap ${check.status}`);
  }
  throw Error('failed to construct a unique resource puzzle');
}

function latinGrid(seed) {
  const random = rng(seed), symbols = shuffled([1, 2, 3, 4], random), rows = shuffled([0, 1, 2, 3], random), cols = shuffled([0, 1, 2, 3], random);
  const solution = {};
  const id = (r, c) => `r${r + 1}c${c + 1}`;
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) solution[id(r, c)] = symbols[(rows[r] + cols[c]) % 4];
  const variables = varMap(Object.keys(solution), [1, 2, 3, 4]), constraints = [];
  for (let r = 0; r < 4; r++) constraints.push({ kind: 'allDifferent', vars: Array.from({ length: 4 }, (_, c) => id(r, c)) });
  for (let c = 0; c < 4; c++) constraints.push({ kind: 'allDifferent', vars: Array.from({ length: 4 }, (_, r) => id(r, c)) });
  const cells = shuffled(Object.keys(solution), random), givens = [];
  for (const cell of cells) {
    constraints.push({ kind: 'equal', variable: cell, value: solution[cell] }); givens.push([cell, solution[cell]]);
    const instance = { schema: 'natlang.finite-csp/1', template: 'latin-4x4-row-column-v1',
      narrative: `Fill a 4 by 4 grid using symbols 1, 2, 3, and 4. Each symbol appears exactly once in every row and every column. Givens: ${givens.map(([cellName, value]) => `${cellName} is ${value}.`).join(' ')}`,
      variables: variables.map(v => ({ ...v })), constraints: structuredClone(constraints) };
    const check = solveUnique(instance);
    if (check.solution) return { instance, solution: { assignment: check.solution }, nodes: check.nodes };
    if (check.status === 'limit' || check.status === 'invalid_or_capped') throw Error(`Latin template cap ${check.status}`);
  }
  throw Error('failed to construct a unique Latin puzzle');
}

function nonogramGrid(seed) {
  const random = rng(seed), n = 4;
  for (let attempt = 0; attempt < 512; attempt++) {
    const bits = Array.from({ length: n }, () => Array.from({ length: n }, () => random() < 0.48 ? 1 : 0));
    if (bits.every(row => row.every(v => !v)) || bits.every(row => row.every(v => v))) continue;
    const vars = Array.from({ length: n * n }, (_, i) => `c${Math.floor(i / n) + 1}_${i % n + 1}`);
    const at = (r, c) => `c${r + 1}_${c + 1}`;
    const constraints = [];
    for (let r = 0; r < n; r++) constraints.push({ kind: 'lineRuns', vars: Array.from({ length: n }, (_, c) => at(r, c)), runs: runs(bits[r]) });
    for (let c = 0; c < n; c++) constraints.push({ kind: 'lineRuns', vars: Array.from({ length: n }, (_, r) => at(r, c)), runs: runs(bits.map(row => row[c])) });
    const rowClues = bits.map(row => runs(row).join(',') || 'empty'), colClues = Array.from({ length: n }, (_, c) => runs(bits.map(row => row[c])).join(',') || 'empty');
    const instance = { schema: 'natlang.finite-csp/1', template: 'nonogram-4x4-runs-v1',
      narrative: `Fill each cell with 0 (blank) or 1 (filled). In each row and column, the listed numbers are the lengths of consecutive filled runs in order; separate runs have at least one blank cell. Row clues from top to bottom: ${rowClues.join(' | ')}. Column clues from left to right: ${colClues.join(' | ')}.`,
      variables: varMap(vars, [0, 1]), constraints };
    const check = solveUnique(instance, 100_000);
    if (check.solution) return { instance, solution: { assignment: check.solution }, nodes: check.nodes };
    if (check.status === 'limit' || check.status === 'invalid_or_capped') continue;
  }
  throw Error('could not construct unique 4x4 nonogram within deterministic search cap');
}

const TEMPLATES = {
  'order-schedule': orderSchedule,
  'logic-grid': logicGrid,
  'resource-placement': resourcePlacement,
  'latin-grid': latinGrid,
  'nonogram-grid': nonogramGrid,
};
const TRAIN_FAMILIES = ['order-schedule', 'logic-grid', 'resource-placement', 'latin-grid'];
function skillFile(family, variant) {
  const description = variant === 'generic' ? 'Use this general method for a finite assignment task.' : DESCRIPTIONS[family];
  const body = variant === 'body-tuned' ? EXPERT_BODIES[family] : BASIC_BODY;
  return { 'SKILL.md': `---\nname: finite-constraint-method\ndescription: ${JSON.stringify(description)}\n---\n\n${body}\n` };
}
function transferFamily(family) {
  const related = { 'order-schedule': 'logic-grid', 'logic-grid': 'resource-placement',
    'resource-placement': 'latin-grid', 'latin-grid': 'order-schedule' };
  return related[family];
}
function makeEpisode(family, split, cases, variant) {
  const sourceGroups = cases.flatMap(([role, rows]) => rows.map(x => x.group));
  const target = { kind: 'improvement-case', entry: 'solve.nl', exportName: 'default',
    source: { schema: 'natlang.skill-csp-target/1', id: `finite-csp-${family}-${TEMPLATE_VERSION}` },
    files: { 'solve.nl': `---\nargs: { instance: string }\nreturns: string\n---\nThe input is a finite constraint puzzle. Its narrative, variables, domains and every rule are visible in the instance. Return JSON with an assignment object mapping each variable ID to one value. You may return a consistent partial assignment while reasoning, but the final answer must assign every variable. Do not add fields that claim a score or feasibility; the host checks all rules independently.\n` } };
  const episode = { version: 'natlang.skill-episode/1', id: `skill-csp-${family}-${variant}-v1`, family: `csp-${family}`, split,
    source_groups: [...new Set(sourceGroups)], license: 'project-generated', target,
    library: { kind: 'existing', skills: { 'finite-constraint-method': skillFile(family, variant) } },
    support: { cases: cases.find(([role]) => role === 'support')[1] },
    query: { cases: cases.find(([role]) => role === 'query')[1] },
    ...(cases.some(([role]) => role === 'transfer') ? { transfer: { family: `csp-${transferFamily(family)}`,
      target: { ...target, source: { ...target.source, id: `finite-csp-${transferFamily(family)}-${TEMPLATE_VERSION}` } },
      cases: cases.find(([role]) => role === 'transfer')[1] } } : {}),
    operations: ['revise', 'select', 'test'], limits: { maxSteps: 6 },
    provenance: { generator: 'natlang.finite-csp-episodes/1', template_version: TEMPLATE_VERSION, skill_condition: variant,
      metric: { schema: 'natlang.skill-csp/1', kind: 'csp-progress' },
      ...(cases.some(([role]) => role === 'transfer') ? { transfer_metric: { schema: 'natlang.skill-csp/1', kind: 'csp-progress' } } : {}) } };
  return episode;
}

function buildBase(family, split, baseSeed, supportN = 4, queryN = 4, transferN = 4) {
  const maker = TEMPLATES[family], support = [], query = [], transfer = [], held = [];
  for (const [role, count, output, roleSeed] of [['support', supportN, support, 101], ['query', queryN, query, 211],
    ...(split === 'train' ? [['transfer', transferN, transfer, 307]] : [])]) {
    const kind = role === 'transfer' ? transferFamily(family) : family, make = TEMPLATES[kind];
    for (let i = 0; i < count; i++) {
      const seed = baseSeed * 1009 + roleSeed + i * 97;
      try {
        const built = make(seed), bound = cspProgressBound(built.instance);
        const report = solveFiniteCsp(built.instance, { maxSolutions: 2 });
        if (report.status !== 'unique' || !sameAssignment(report.solutions[0], built.solution.assignment)) {
          held.push({ family: kind, seed, role, reason: report.status === 'unique' ? 'generator_solution_mismatch' : report.status }); continue;
        }
        output.push({ id: `csp-${kind}-${seed}`, group: `generated:${TEMPLATE_VERSION}:${kind}:${seed}`,
          args: [JSON.stringify(built.instance)], expected: built.solution, expectedFiles: undefined,
          // Host-only construction audit; stripped before JSON serialization below.
          __bound: bound, __nodes: report.nodes });
      } catch (error) { held.push({ family: kind, seed, role, reason: 'construction_failure', detail: error.message }); }
    }
  }
  const clean = arr => arr.map(({ __bound, __nodes, ...row }) => row);
  const episodes = VARIANTS.map(variant => makeEpisode(family, split, [
    ['support', clean(support)], ['query', clean(query)], ...(split === 'train' ? [['transfer', clean(transfer)]] : []),
  ], variant));
  const cases = [...support, ...query, ...transfer].map(({ id, group, args, expected, __bound, __nodes }) => ({ id, group, args, expected, bound: __bound, solver_nodes: __nodes }));
  return { episodes, cases, held };
}

function audit(episodes) {
  const errors = [], assigned = new Map();
  for (const episode of episodes) {
    errors.push(...validateEpisode(episode).map(item => ({ episode: episode.id, ...item })));
    for (const role of ['support', 'query', 'transfer']) for (const item of episode[role]?.cases ?? []) {
      const key = `${episode.split}/${role}`;
      if (assigned.has(item.group) && assigned.get(item.group) !== key) errors.push({ code: 'group-role-collision', group: item.group });
      assigned.set(item.group, key);
    }
  }
  if (errors.length) throw new Error(`CSP episodes rejected: ${JSON.stringify(errors.slice(0, 12))}`);
  return assigned.size;
}

export function buildCspPacket(counts = { support: 4, query: 4, transfer: 4 }) {
  if (Object.values(counts).some(n => !Number.isSafeInteger(n) || n < 2 || n > 12)) throw Error('case counts must be integers from 2 through 12');
  const all = [], held = [], caseAudit = [];
  for (let i = 0; i < TRAIN_FAMILIES.length; i++) {
    const result = buildBase(TRAIN_FAMILIES[i], 'train', i + 11, counts.support, counts.query, counts.transfer);
    all.push(...result.episodes); held.push(...result.held); caseAudit.push(...result.cases);
  }
  const validation = buildBase('nonogram-grid', 'validation', 401, counts.support, counts.query, 0);
  all.push(...validation.episodes); held.push(...validation.held); caseAudit.push(...validation.cases);
  const groups = audit(all);
  const body = all.map(row => JSON.stringify(row)).join('\n') + '\n';
  const heldBody = held.map(row => JSON.stringify(row)).join('\n') + (held.length ? '\n' : '');
  const auditBody = caseAudit.map(row => JSON.stringify(row)).join('\n') + '\n';
  const manifest = { schema: 'natlang.finite-csp-episodes/1', template_version: TEMPLATE_VERSION,
    episodes: all.length, families: [...new Set(all.map(e => e.family))].sort(), variants: VARIANTS,
    by_split: Object.fromEntries(['train', 'validation', 'test'].map(s => [s, all.filter(e => e.split === s).length])),
    unique_case_groups: groups, unique_generated_cases: caseAudit.length, held_construction_candidates: held.length,
    exact_unique_solutions: caseAudit.length, max_nodes_observed: Math.max(0, ...caseAudit.map(c => c.solver_nodes)),
    heldout_structure: { train_templates: TRAIN_FAMILIES, validation_template: 'nonogram-grid/nonogram-4x4-runs-v1',
      validation_is_new_constraint_shape: true },
    objective: { schema: 'natlang.skill-csp/1', kind: 'csp-progress', partial_quality: 'extendable assigned-variable fraction; contradictions and dead ends score 0; complete valid assignment scores 1' },
    limits: { ...Object.fromEntries(Object.entries(counts).map(([k, v]) => [`${k}_per_base_family`, v])), solver_nodes_per_uniqueness_check: 250_000 },
    license: 'project-generated', model_calls: 0, provider_calls: 0,
    sha256: { episodes: sha(body), held: sha(heldBody), construction_audit: sha(auditBody) } };
  return { all, held, caseAudit, manifest, body, heldBody, auditBody };
}

async function main(argv = process.argv) {
  const arg = name => { const i = argv.indexOf(name); return i < 0 ? undefined : argv[i + 1]; };
  if (!arg('--out')) throw Error('Usage: node scripts/skills/build-csp-episodes.mjs --out DIR [--support N --query N --transfer N]');
  const out = resolve(arg('--out'));
  const counts = { support: Number(arg('--support') ?? 4), query: Number(arg('--query') ?? 4), transfer: Number(arg('--transfer') ?? 4) };
  const packet = buildCspPacket(counts), { manifest } = packet;
  await mkdir(out, { recursive: true });
  await writeFile(join(out, 'csp-episodes.jsonl'), packet.body, { flag: 'wx' });
  await writeFile(join(out, 'held-constructions.jsonl'), packet.heldBody, { flag: 'wx' });
  await writeFile(join(out, 'construction-audit.jsonl'), packet.auditBody, { flag: 'wx' });
  await writeFile(join(out, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify(manifest, null, 2));
}
if (process.argv[1] && import.meta.url === new URL(`file://${resolve(process.argv[1])}`).href) await main();
