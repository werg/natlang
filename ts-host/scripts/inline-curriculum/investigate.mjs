// Investigation families: a frontier search whose length is not known in advance (iterateOn, with a progress
// review on a long but useful run), and an exact aggregate whose winner is decided by the last page.
import { createHash } from 'node:crypto';
import { blockedCall, capitalize, curriculumCase, evalCall, literal, nonceWords, Random, returnCall } from './lib.mjs';

/**
 * Breadth-first search over a graph that can only be explored link by link. The target is near, far (past the
 * progress review, which should let the shrinking search continue), or unreachable (the frontier empties).
 */
export function frontierSearch(seed, index) {
  const rng = new Random(seed, `frontier:${index}`);
  const used = new Set();
  const id = () => `n-${nonceWords(rng, 1, used)[0]}`;
  const from = id(), to = id();
  const build = (depth, reachable) => {
    const links = { [from]: [], [to]: [id()] };
    let previous = from;
    const length = reachable ? depth : 4;
    for (let i = 1; i <= length; i++) {
      const node = i === depth && reachable ? to : id();
      links[previous].push(node);
      links[node] ??= [];
      // A dead-end decoy next to every node on the path.
      const decoy = id();
      links[previous].push(decoy);
      links[decoy] = [id()];
      previous = node;
    }
    for (const node of Object.values(links).flat()) links[node] ??= [];
    for (const key of Object.keys(links)) links[key] = rng.shuffle(links[key]);
    return links;
  };
  const shape = `graph${index}`;
  const variants = { near: [rng.int(2, 3), true], far: [rng.int(12, 14), true], unreachable: [0, false] };
  return Object.entries(variants).map(([variant, [depth, reachable]]) => {
    const links = build(depth, reachable);
    const expected = reachable ? depth : null;
    return curriculumCase({ family: 'iterate_frontier', shape, variant, pairGroup: `frontier:${shape}`,
      slice: 'iterate', domain: 'relational', mode: 'single_call', inline: 'avoid', iterate: 'required', worldSemantics: 'closed_world',
      evidence: { world: [`distance ${expected}`], retrieved: [], background: [] },
      minimumSequence: ['expand the frontier one hop per step with iterateOn', 'stop when the target is found or nothing is left'],
      reference: { root: [evalCall(`type Search = { frontier: string[], seen: string[], depth: number, found: boolean };
const expand = (state: Search): Search => {
  const next: string[] = [];
  for (const node of state.frontier) for (const link of graph.links(node)) if (!state.seen.includes(link) && !next.includes(link)) next.push(link);
  return { frontier: next, seen: [...state.seen, ...next], depth: state.depth + 1, found: next.includes(to) };
};
const final = await iterateOn(expand, { frontier: [from], seen: [from], depth: 0, found: from === to }).until(state => state.found || state.frontier.length === 0);
return final.found ? final.depth : null;`), returnCall(expected)],
        children: [{ match: 'An iterative process', value: { verdict: 'continue', reason: 'Every step reaches new nodes and the search has not repeated a state.' } }] },
      root: { name: 'link_distance', args: { from: 'string', to: 'string' }, returns: 'number | null',
        instructions: 'Find the smallest number of links to follow from node from to reach node to, where graph.links(node) lists the nodes a node links to. Return null when to cannot be reached.' },
      files: { 'link_distance/graph.ts': `const LINKS: Record<string, string[]> = ${literal(links)};\n/** The nodes that node links to (outgoing links only). */\nexport function links(node: string): string[] { return LINKS[node] ?? []; }\n` },
      inputs: { from, to }, expected });
  });
}

/** The supplier with the most late third-quarter shipments; the last page decides between two suppliers. */
export function lateArgmax(seed, index) {
  const rng = new Random(seed, `argmax:${index}`);
  const [leader, rival, third] = nonceWords(rng, 3).map(capitalize);
  const row = (supplier, quarter, status) => ({ supplier, quarter, status });
  const base = [
    ...Array.from({ length: 4 }, () => row(leader, 'Q3', 'late')),
    ...Array.from({ length: 3 }, () => row(rival, 'Q3', 'late')),
    ...Array.from({ length: 2 }, () => row(third, 'Q3', 'late')),
    ...Array.from({ length: 6 }, () => row(rng.pick([leader, rival, third]), 'Q3', 'on_time')),
    ...Array.from({ length: 5 }, () => row(rng.pick([leader, rival, third]), 'Q2', rng.pick(['late', 'on_time']))),
    ...Array.from({ length: 3 }, () => row(third, 'Q2', 'late')),
  ];
  const early = rng.shuffle(base);
  const filler = [row(third, 'Q3', 'on_time'), row(leader, 'Q2', 'on_time'), row(third, 'Q2', 'on_time'), row(leader, 'Q3', 'on_time'), row(third, 'Q3', 'on_time')];
  const shape = `shipments${index}`;
  const variants = {
    late_q3: { tail: [row(rival, 'Q3', 'late'), row(rival, 'Q3', 'late')], winner: rival },
    other_quarter: { tail: [row(rival, 'Q2', 'late'), row(rival, 'Q2', 'late')], winner: leader },
    cancelled: { tail: [row(rival, 'Q3', 'cancelled'), row(rival, 'Q3', 'cancelled')], winner: leader },
  };
  return Object.entries(variants).map(([variant, spec]) => {
    const rows = [...early, ...rng.shuffle([...spec.tail, ...filler.slice(0, 5)])].map((item, i) => ({ id: `S${i + 1}`, ...item }));
    const counts = {};
    for (const item of rows) if (item.quarter === 'Q3' && item.status === 'late') counts[item.supplier] = (counts[item.supplier] ?? 0) + 1;
    const best = Object.entries(counts).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0][0];
    if (best !== spec.winner) throw new Error(`late_argmax ${variant}: expected ${spec.winner}, computed ${best}`);
    return curriculumCase({ family: 'relational_late_argmax', shape, variant, pairGroup: `argmax:${shape}`,
      slice: 'observation_followup', domain: 'relational', mode: 'single_call', inline: 'avoid', worldSemantics: 'closed_world',
      evidence: { world: spec.tail.map(t => `${t.supplier} ${t.quarter} ${t.status}`), retrieved: ['all pages'], background: [] },
      minimumSequence: ['read every page', 'count late Q3 shipments per supplier', 'take the maximum, ties alphabetical'],
      reference: { root: [evalCall(`const counts: Record<string, number> = {};
const pages = shipments.page_count();
for (let n = 1; n <= pages; n++) for (const item of shipments.page(n)) if (item.quarter === 'Q3' && item.status === 'late') counts[item.supplier] = (counts[item.supplier] ?? 0) + 1;
const ranked = Object.entries(counts).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
return ranked[0][0];`), returnCall(best)] },
      root: { name: 'worst_supplier', args: {}, returns: 'string',
        instructions: 'Which supplier had the most late shipments in Q3? A cancelled shipment is not late. Break a tie by the alphabetically first supplier name. The shipments are in shipments.page(n).' },
      files: { 'worst_supplier/shipments.ts': `const ROWS = ${literal(rows)};
/** Page n of the shipment log (1-based, eight rows per page). */
export function page(n: number): { id: string, supplier: string, quarter: string, status: "late" | "on_time" | "cancelled" }[] { return ROWS.slice((n - 1) * 8, n * 8); }
/** The number of pages in the shipment log. */
export function page_count(): number { return ${Math.ceil(rows.length / 8)}; }
` },
      inputs: {}, expected: best });
  });
}

/**
 * A reactive controller run with iterateOn: each step reads the greenhouse and sets heater and vent until the
 * temperature holds between 20 and 22 °C for three readings. Sunlight changes the dynamics partway through, a
 * cold start runs past the progress review (which should let it continue), and a broken heater makes the goal
 * unreachable (the review stops it, and the honest outcome is blocked).
 */
export function greenhouseControl(seed, index) {
  const rng = new Random(seed, `greenhouse:${index}`);
  const certificate = `stable-${createHash('sha256').update(`${seed}:${index}:greenhouse`).digest('hex').slice(0, 8)}`;
  const shape = `greenhouse${index}`;
  const variants = {
    normal: { start: rng.pick([15, 16, 17]), sunAt: 0, broken: false },
    sun_arrives: { start: rng.pick([15, 16]), sunAt: rng.int(3, 5), broken: false },
    long: { start: rng.pick([6, 7]), sunAt: 0, broken: false },
    heater_broken: { start: rng.pick([15, 16]), sunAt: 0, broken: true },
  };
  // The reference controller, simulated here to know how the run ends and whether the review is reached.
  const simulate = spec => {
    let temp = spec.start, tick = 0, heater = false, vent = false;
    const read = () => {
      tick++;
      const sunny = spec.sunAt > 0 && tick >= spec.sunAt;
      temp = Math.round((temp + (sunny ? 0.8 : -0.4) + (heater && !spec.broken ? 1.2 : 0) - (vent ? 1.5 : 0)) * 10) / 10;
      return temp;
    };
    const readings = [read()];
    for (let step = 1; step <= 40; step++) {
      heater = temp < 20.5; vent = temp > 21.5;
      readings.push(read());
      if (readings.slice(-3).length === 3 && readings.slice(-3).every(t => t >= 20 && t <= 22)) return { steps: step, reached: true };
    }
    return { steps: 40, reached: false };
  };
  return Object.entries(variants).map(([variant, spec]) => {
    const run = simulate(spec);
    if (!spec.broken && !run.reached) throw new Error(`greenhouse ${variant}: the reference controller does not stabilise`);
    const module = `let temp = ${spec.start};
let tick = 0;
let heater = false;
let vent = false;
const readings: number[] = [];
/** Set the heater and the vent; they act from the next reading on. */
export function set(controls: { heater: boolean, vent: boolean }): void { heater = controls.heater; vent = controls.vent; }
/** Let ten minutes pass and read the temperature in °C (one decimal), and whether the sun is on the glass. */
export function read(): { minute: number, temp: number, sunny: boolean } {
  tick += 1;
  const sunny = ${spec.sunAt} > 0 && tick >= ${spec.sunAt};
  temp = Math.round((temp + (sunny ? 0.8 : -0.4) + (heater && !${spec.broken} ? 1.2 : 0) - (vent ? 1.5 : 0)) * 10) / 10;
  readings.push(temp);
  return { minute: tick * 10, temp, sunny };
}
/** The stability certificate once the last three readings were all between 20 and 22 °C; otherwise null. */
export function certificate(): string | null {
  const last = readings.slice(-3);
  return last.length === 3 && last.every(t => t >= 20 && t <= 22) ? ${JSON.stringify(certificate)} : null;
}
`;
    const loop = `type Run = { temp: number, certified: string | null };
const control = (run: Run): Run => {
  greenhouse.set({ heater: run.temp < 20.5, vent: run.temp > 21.5 });
  const reading = greenhouse.read();
  return { temp: reading.temp, certified: greenhouse.certificate() };
};
const first = greenhouse.read();`;
    const reference = spec.broken ? [
      evalCall(`${loop}
let outcome: string;
try { outcome = (await iterateOn(control, { temp: first.temp, certified: null }).until(run => run.certified !== null)).certified!; }
catch (error) { outcome = String(error) + ' | last reading ' + greenhouse.read().temp; }
outcome`),
      blockedCall('The greenhouse keeps cooling with the heater on: the heater does not respond, so the temperature cannot be brought into the band.'),
    ] : [evalCall(`${loop}
const final = await iterateOn(control, { temp: first.temp, certified: null }).until(run => run.certified !== null);
return final.certified;`), returnCall(certificate)];
    const judge = spec.broken ?
      { verdict: 'divergent', reason: 'The temperature falls every step although the heater is on; the run is moving away from the band.' } :
      { verdict: 'continue', reason: 'The temperature rises steadily toward the band.' };
    return curriculumCase({ family: 'actor_greenhouse', shape, variant, pairGroup: `greenhouse:${shape}`,
      slice: 'iterate', domain: 'actor', mode: 'single_call', inline: 'avoid', iterate: 'required',
      sketch: 'the state is { temp: number, certified: string | null }; the step sets heater and vent from state.temp, ' +
        'calls greenhouse.read(), and returns the new temperature with greenhouse.certificate(); stop when certified is not null. ' +
        'A simple rule is enough: heat below the band and vent above it; there is no need to measure the rates first.',
      evidence: { world: [`start ${spec.start} °C`, spec.sunAt ? `sun from reading ${spec.sunAt}` : 'no sun', spec.broken ? 'heater broken' : 'heater works'],
        retrieved: [], background: [`reference controller: ${run.reached ? `${run.steps} steps` : 'never stabilises'}`] },
      minimumSequence: ['read the greenhouse', 'step a controller with iterateOn until the certificate appears', spec.broken ? 'report the unreachable goal' : 'return the certificate'],
      reference: { root: reference, children: [{ match: 'An iterative process', value: judge }] },
      root: { name: 'stabilise_greenhouse', args: {}, returns: 'string',
        instructions: 'Bring the greenhouse to between 20 and 22 °C and keep it there until greenhouse.certificate() issues a certificate (three readings in a row in that band), by setting its heater and vent between readings. Return the certificate.' },
      files: { 'stabilise_greenhouse/greenhouse.ts': module },
      inputs: {}, expected: spec.broken ? null : certificate, ...(spec.broken ? { operation: 'blocked' } : {}) });
  });
}
