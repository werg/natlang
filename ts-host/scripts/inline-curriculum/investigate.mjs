// Investigation families: a frontier search whose length is not known in advance (iterateOn, with a progress
// review on a long but useful run), and an exact aggregate whose winner is decided by the last page.
import { capitalize, curriculumCase, evalCall, literal, nonceWords, Random, returnCall } from './lib.mjs';

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
