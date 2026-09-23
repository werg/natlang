// Relational families: paged multi-hop graph queries, semantic policy over exact candidates, and snapshot updates.
import { Random, capitalize, curriculumCase, evalCall, literal, nonceWords, returnCall } from './lib.mjs';

const EDGE_PAGE = 4;

/** A paged graph adapter over typed edges. */
function graphModule(edges, notes = {}) {
  return `/** A relation read as a sentence: from relation to (for example "c-a owned_by c-p": c-p owns c-a), true from since until until (null: still true). */
type Edge = { from: string, relation: string, to: string, since: number, until: number | null };
const EDGES: Edge[] = ${literal(edges)};
const NOTES: Record<string, string[]> = ${literal(notes)};
/** Edges into entity with the given relation, ${EDGE_PAGE} per page, pages numbered from 1; more is true when another page follows. */
export function incoming(entity: string, relation: string, page: number): { edges: Edge[], more: boolean } {
  if (!Number.isInteger(page) || page < 1) throw new RangeError('pages are numbered from 1');
  const all = EDGES.filter(e => e.to === entity && e.relation === relation);
  return { edges: all.slice((page - 1) * ${EDGE_PAGE}, page * ${EDGE_PAGE}), more: all.length > page * ${EDGE_PAGE} };
}
/** Edges out of entity with the given relation, ${EDGE_PAGE} per page, pages numbered from 1; more is true when another page follows. */
export function outgoing(entity: string, relation: string, page: number): { edges: Edge[], more: boolean } {
  if (!Number.isInteger(page) || page < 1) throw new RangeError('pages are numbered from 1');
  const all = EDGES.filter(e => e.from === entity && e.relation === relation);
  return { edges: all.slice((page - 1) * ${EDGE_PAGE}, page * ${EDGE_PAGE}), more: all.length > page * ${EDGE_PAGE} };
}
/** Audit notes recorded for an entity, oldest first. */
export function notes(entity: string): string[] { return NOTES[entity] ?? []; }
`;
}
/** Reference code: every edge into `entity` with `relation`, across pages. */
const allIncoming = (fn, entity, relation) => `(() => { const out = []; for (let page = 1; page <= 20; page++) {
  const r = graph.incoming(${entity}, ${JSON.stringify(relation)}, page); out.push(...r.edges); if (!r.more) break; } return out; })()`;

/**
 * Two hops with temporal qualifiers: people who currently work for a company the parent currently owns.
 * Variants differ in hidden edges: a divestment, an extra employee on a later page, or a departure.
 */
export function multihopQualifier(seed, index) {
  const rng = new Random(seed, `multihop:${index}`);
  const used = new Set();
  const [parentName, rivalName, ...companyNames] = nonceWords(rng, 6, used).map(capitalize);
  const people = nonceWords(rng, 16, used).map(name => `p-${name}`);
  const parent = `c-${parentName.toLowerCase()}`, rival = `c-${rivalName.toLowerCase()}`;
  const companies = companyNames.map(name => `c-${name.toLowerCase()}`);
  const [a, b, c, d] = companies;
  const base = [
    { from: a, relation: 'owned_by', to: parent, since: 2019, until: null },
    { from: b, relation: 'owned_by', to: parent, since: 2021, until: null },
    { from: c, relation: 'owned_by', to: parent, since: 2015, until: 2020 },
    { from: c, relation: 'owned_by', to: rival, since: 2020, until: null },
    { from: d, relation: 'owned_by', to: rival, since: 2018, until: null },
  ];
  const staff = (company, list, pastIndex) => list.map((person, i) => ({ from: person, relation: 'works_for', to: company,
    since: 2016 + i, until: i === pastIndex ? 2023 : null }));
  const baseStaff = [...staff(a, people.slice(0, 6), 2), ...staff(b, people.slice(6, 9), -1),
    ...staff(c, people.slice(9, 12), -1), ...staff(d, people.slice(12, 14), -1)];
  const variants = {
    base: { edges: [...base, ...baseStaff] },
    divested: { edges: [...base.map(e => e.from === b ? { ...e, until: 2025 } : e), ...baseStaff] },
    late_hire: { edges: [...base, ...baseStaff, { from: people[14], relation: 'works_for', to: a, since: 2026, until: null }] },
    departed: { edges: [...base, ...baseStaff.map(e => e.from === people[5] ? { ...e, until: 2026 } : e)] },
  };
  const shape = `owners${index}`;
  return Object.entries(variants).map(([variant, v]) => {
    const current = e => e.until === null;
    const owned = v.edges.filter(e => e.relation === 'owned_by' && e.to === parent && current(e)).map(e => e.from);
    const expected = v.edges.filter(e => e.relation === 'works_for' && owned.includes(e.to) && current(e)).map(e => e.from).sort();
    const code = `const owned = ${allIncoming('graph', JSON.stringify(parent), 'owned_by')}.filter(e => e.until === null).map(e => e.from);
const people: string[] = [];
for (const company of owned) people.push(...${allIncoming('graph', 'company', 'works_for')}.filter(e => e.until === null).map(e => e.from));
return people.sort();`;
    return curriculumCase({ family: 'relational_multihop_qualifier', shape, variant, pairGroup: `multihop:${shape}`,
      slice: 'nested_scoped', domain: 'relational', mode: 'single_call', inline: 'avoid', worldSemantics: 'closed_world',
      evidence: { world: v.edges.map(e => `${e.from} ${e.relation} ${e.to} ${e.since}-${e.until ?? 'now'}`), retrieved: expected, background: [] },
      assumptions: ['A relation is current when it has no end year.'],
      plausibleActions: [], minimumSequence: ['page through owned_by edges into the parent', 'page through works_for edges into each current company'],
      reference: { root: [evalCall(code), ['return_result', { value: expected }]] },
      root: { name: 'current_staff', args: { parent: 'string' }, returns: 'string[]',
        instructions: `List the people who currently work for a company that parent currently owns, using graph.
Each edge reads as a sentence, from relation to: "c-a owned_by c-p" means c-p owns c-a, and "p-x works_for c-a" means p-x works for c-a. An edge is current when its until is null.
Return the person ids sorted alphabetically.` },
      files: { 'current_staff/graph.ts': graphModule(v.edges) }, inputs: { parent }, expected });
  });
}

const AUDIT_NOTES = {
  disqualifying: ['Audit found two fire exits blocked by pallets; the fix is still pending.',
    'Inspectors found leaking solvent drums in storage; no follow-up has been done.',
    'A guard on the cutting press was missing and the press was still in use at the end of the audit.',
    'Workers reported unlabelled acid containers; the supplier has not responded.'],
  benign: ['A delivery form was missing a signature; it was resubmitted the same week.',
    'A forklift brake fault was found, repaired, and passed re-inspection.',
    'The supplier filed its annual report two days late.',
    'An emergency light was dim; it was replaced during the audit and retested.'],
};

/**
 * Exact candidate retrieval, then a semantic policy judgment per candidate through a typed callback helper.
 * Variants share the opening; which suppliers' notes describe unresolved safety problems differs.
 */
export function policyCandidates(seed, index) {
  const rng = new Random(seed, `policy:${index}`);
  const used = new Set();
  const [productName, ...names] = nonceWords(rng, 8, used);
  const product = `prod-${productName}`;
  const suppliers = names.slice(0, 5).map(name => `s-${name}`);
  const edges = [...suppliers.slice(0, 4).map((s, i) => ({ from: s, relation: 'supplies', to: product, since: 2020 + i, until: null })),
    { from: suppliers[4], relation: 'supplies', to: product, since: 2017, until: 2022 }];
  const current = suppliers.slice(0, 4);
  const patterns = { one_bad: [true, false, false, false], two_bad: [false, true, false, true], none_bad: [false, false, false, false] };
  const shape = `suppliers${index}`;
  return Object.entries(patterns).map(([variant, bad]) => {
    const notes = Object.fromEntries(current.map((s, i) => [s, [rng.pick(AUDIT_NOTES.benign),
      ...(bad[i] ? [rng.pick(AUDIT_NOTES.disqualifying)] : [])]]));
    notes[suppliers[4]] = [rng.pick(AUDIT_NOTES.disqualifying)];
    const expected = current.filter((_, i) => !bad[i]).sort();
    const code = `const current = ${allIncoming('graph', 'product', 'supplies')}.filter(e => e.until === null).map(e => e.from);
const kept = await review_each(current, nl\`Does supplier qualify under policy, given graph.notes(supplier)?\`);
return kept.sort();`;
    return curriculumCase({ family: 'relational_policy_inline', shape, variant, pairGroup: `policy:${shape}`,
      slice: 'inline_placement', domain: 'relational', mode: 'single_call', inline: 'required', worldSemantics: 'closed_world',
      evidence: { world: Object.entries(notes).map(([s, list]) => `${s}: ${list.join(' ')}`), retrieved: expected,
        background: ['A repaired or resubmitted finding is resolved; a pending or ignored hazard is not.'] },
      plausibleActions: [], minimumSequence: ['collect current suppliers exactly', 'judge each supplier against the policy in a child call'],
      reference: { root: [evalCall(code), ['return_result', { value: expected }]],
        children: current.map((s, i) => ({ match: JSON.stringify(s), value: !bad[i] })) },
      root: { name: 'qualified_suppliers', args: { product: 'string', policy: 'string' }, returns: 'string[]',
        instructions: `Find the suppliers that currently supply product (supplies edges in graph whose until is null), then keep those that qualify under policy.
Each supplier's audit notes are in graph.notes. Use review_each to judge the suppliers one at a time.
Return the qualifying supplier ids sorted alphabetically.` },
      files: { 'qualified_suppliers/graph.ts': graphModule(edges, notes),
        'qualified_suppliers/review_each.ts': `/** Judge each id separately and keep the ones judged true, in input order. */
export default async function review_each(ids: string[], judge: (supplier: string) => Promise<boolean>): Promise<string[]> {
  const kept: string[] = [];
  for (const id of ids) if (await judge(id)) kept.push(id);
  return kept;
}
` },
      inputs: { product, policy: 'A supplier qualifies unless its audit notes describe a safety hazard that has not been resolved. Paperwork problems and hazards that were fixed do not disqualify.' },
      expected });
  });
}

/**
 * Snapshot updates: a cached team roster predates pending events written in prose. Variants differ in
 * whether the events change the answer.
 */
export function dynamicSnapshot(seed, index) {
  const rng = new Random(seed, `snapshot:${index}`);
  const used = new Set();
  const [teamA, teamB] = nonceWords(rng, 2, used).map(capitalize);
  const names = nonceWords(rng, 8, used).map(capitalize);
  const roster = { [teamA]: names.slice(0, 4).sort(), [teamB]: names.slice(4, 7).sort() };
  const newcomer = names[7];
  const eventSets = {
    transfer: [`${names[5]} moved from ${teamB} to ${teamA}.`, `${names[1]} took two weeks of leave and stays on ${teamA}.`],
    departure: [`${names[2]} left the company.`, `${teamB} hired ${newcomer}.`],
    unrelated: [`${names[4]} was promoted within ${teamB}.`, `${teamB} hired ${newcomer}.`],
    rejoin: [`${names[0]} left the company.`, `${names[0]} rejoined the company on ${teamA} a week later.`, `${newcomer} joined ${teamA}.`],
  };
  const apply = events => {
    const teams = structuredClone(roster);
    for (const event of events) {
      let m;
      if ((m = /^(\w+) moved from (\w+) to (\w+)\.$/.exec(event))) { teams[m[2]] = teams[m[2]].filter(x => x !== m[1]); teams[m[3]].push(m[1]); }
      else if ((m = /^(\w+) left the company\.$/.exec(event))) for (const t of Object.keys(teams)) teams[t] = teams[t].filter(x => x !== m[1]);
      else if ((m = /^(\w+) hired (\w+)\.$/.exec(event))) teams[m[1]].push(m[2]);
      else if ((m = /^(\w+) joined (\w+)\.$/.exec(event))) teams[m[2]].push(m[1]);
      else if ((m = /^(\w+) rejoined the company on (\w+) a week later\.$/.exec(event))) teams[m[2]].push(m[1]);
    }
    return teams[teamA].sort();
  };
  const shape = `roster${index}`;
  return Object.entries(eventSets).map(([variant, events]) => {
    const expected = apply(events);
    const module = `const EVENTS: string[] = ${literal(events)};
/** Events recorded since the cached roster was computed, oldest first. */
export function pending(): string[] { return EVENTS; }
`;
    return curriculumCase({ family: 'relational_dynamic_snapshot', shape, variant, pairGroup: `snapshot:${shape}`,
      slice: 'observation_followup', domain: 'relational', mode: 'followup',
      evidence: { world: [`cached ${teamA}: ${roster[teamA].join(', ')}`, ...events], retrieved: events, background: [] },
      assumptions: ['Events apply in order; a later event can undo an earlier one.'],
      decisive: [{ marker: events[0], source: 'eval', note: 'the first pending event' }],
      plausibleActions: ['return the cached roster', 'apply the events and recompute'],
      minimumSequence: ['read the pending events', 'apply them to the cached roster', 'return the current roster'],
      reference: { root: [evalCall('changes.pending()'), returnCall(expected)] },
      root: { name: 'current_roster', args: { team: 'string', cached: 'Record<string, string[]>' }, returns: 'string[]',
        instructions: `Return the current members of team, sorted alphabetically.
cached holds every team's roster as of the last snapshot; changes.pending() lists what has happened since, in order.` },
      files: { 'current_roster/changes.ts': module }, inputs: { team: teamA, cached: roster }, expected });
  });
}
