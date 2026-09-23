// Failure-and-repair families: a rejected loop rewritten within the eval rules, a child that honestly cannot
// answer, and an application event that must not be applied twice when only its view failed.
import { blockedCall, curriculumCase, evalCall, failedCall, literal, nlFile, nonceWords, Random, returnCall } from './lib.mjs';

/**
 * A seeded eval pages through a ledger with a `while` loop, which eval rejects. With a page count the repair is a
 * counted loop; without one it is `iterateOn` until an empty page.
 */
export function loopRewrite(seed, index) {
  const rng = new Random(seed, `loop:${index}`);
  const categories = ['travel', 'meals', 'software', 'office'];
  const pages = rng.int(4, 6);
  const entries = Array.from({ length: pages * 5 }, (_, i) => ({ id: `E${i + 1}`, category: rng.pick(categories), amount: rng.int(5, 400) }));
  const pageOf = n => entries.slice((n - 1) * 5, n * 5);
  const target = rng.pick(categories);
  const expected = entries.filter(e => e.category === target).reduce((sum, e) => sum + e.amount, 0);
  const seedCode = `let total = 0;
let n = 1;
while (true) {
  const rows = ledger.page(n);
  if (rows.length === 0) break;
  for (const row of rows) if (row.category === ${JSON.stringify(target)}) total += row.amount;
  n++;
}
total`;
  const module = counted => `const ENTRIES = ${literal(entries)};
/** Page n of the ledger (1-based, five entries per page); an empty list after the last page. */
export function page(n: number): { id: string, category: string, amount: number }[] { return ENTRIES.slice((n - 1) * 5, n * 5); }
${counted ? `/** The number of ledger pages. */\nexport function page_count(): number { return ${pages}; }\n` : ''}`;
  const shape = `ledger${index}`;
  const common = { slice: 'folder_failure', domain: 'other', mode: 'followup', inline: 'avoid',
    evidence: { world: [`${pages} pages`, `${target} total ${expected}`], retrieved: ['the loop rejection'], background: [] },
    decisive: [{ marker: 'loops are not allowed here', source: 'error', note: 'the eval rejects the while loop' }],
    plausibleActions: ['retry the same loop', 'rewrite it as a counted loop', 'rewrite it with iterateOn'],
    root: { name: 'category_total', args: {}, returns: 'number',
      instructions: `Add up the amounts of every ledger entry in category "${target}", across all pages of ledger.page(n).` },
    inputs: {}, expected, failureSeed: { kind: 'compile', code: seedCode } };
  return [
    curriculumCase({ ...common, family: 'loop_rewrite', shape, variant: 'counted',
      minimumSequence: ['read the rejection', 'loop over the known page count'],
      reference: { root: [evalCall(`let total = 0;
const pages = ledger.page_count();
for (let n = 1; n <= pages; n++) for (const row of ledger.page(n)) if (row.category === ${JSON.stringify(target)}) total += row.amount;
return total;`), returnCall(expected)] },
      files: { 'category_total/ledger.ts': module(true) } }),
    curriculumCase({ ...common, family: 'loop_rewrite', shape, variant: 'open_ended', iterate: 'required',
      minimumSequence: ['read the rejection', 'page with iterateOn until an empty page'],
      reference: { root: [evalCall(`type Scan = { n: number, total: number, done: boolean };
const scan = (state: Scan): Scan => {
  const rows = ledger.page(state.n);
  const add = rows.filter(row => row.category === ${JSON.stringify(target)}).reduce((sum, row) => sum + row.amount, 0);
  return { n: state.n + 1, total: state.total + add, done: rows.length === 0 };
};
const final = await iterateOn(scan, { n: 1, total: 0, done: false }).until(state => state.done);
return final.total;`), returnCall(expected)] },
      files: { 'category_total/ledger.ts': module(false) } }),
  ];
  void pageOf;
}

const INVOICES = [
  { text: 'Invoice 2211 from Fernwood Supplies. 40 reams of paper, 12 toner cartridges. Total due: 1,284.50 EUR by April 30.', total: 1284.5 },
  { text: 'Invoice 87 - Kestrel Cleaning. March office cleaning, 8 visits. Amount due 640.00 EUR.', total: 640 },
  { text: 'Invoice A-19 from Brightline IT. Laptop repair and two replacement batteries. Please pay 312.75 EUR within 14 days.', total: 312.75 },
  { text: 'Invoice 5530, Oakridge Catering. Lunch for the board meeting on March 12, 14 guests. Total 455.00 EUR.', total: 455 },
];
const UNPRICED = [
  'Invoice 9002 from Larkspur Print. 500 brochures, 200 posters. The amount will follow in a separate statement once the paper surcharge is known.',
  'Invoice 31 - Tidewater Movers. Office move on March 3, two vans, four movers. Pricing: see the attached quote (not included here).',
];

/**
 * An honest opt-out: one invoice states no total, so its child cannot produce a number and says so; the
 * parent records null for it instead of computing a figure from line items.
 */
export function childOptOut(seed, index) {
  const rng = new Random(seed, `optout:${index}`);
  const priced = rng.sample(INVOICES, 3);
  const shape = `invoices${index}`;
  const variants = {
    one_unpriced: [...priced.slice(0, 2), { text: rng.pick(UNPRICED), total: null }],
    all_priced: priced,
  };
  return Object.entries(variants).map(([variant, list]) => {
    const invoices = rng.shuffle(list).map((invoice, i) => ({ id: `I${i + 1}`, ...invoice }));
    const expected = Object.fromEntries(invoices.map(invoice => [invoice.id, invoice.total]));
    const plain = invoices.map(({ id, text }) => ({ id, text }));
    return curriculumCase({ family: 'child_opt_out', shape, variant, pairGroup: `optout:${shape}`,
      slice: 'nested_scoped', domain: 'other', mode: 'single_call', inline: 'required',
      evidence: { world: invoices.map(i => `${i.id}: ${i.total}`), retrieved: [], background: [] },
      minimumSequence: ['read each invoice in its own child', 'record null where the child reports no stated total'],
      reference: { root: [evalCall(`const totals: Record<string, number | null> = {};
for (const invoice of payables.invoices()) {
  try {
    totals[invoice.id] = await nl<number>\`Read the total amount due that invoice states, in euros.\`(invoice);
  } catch (error) {
    totals[invoice.id] = null;
  }
}
return totals;`), returnCall(expected)],
        children: invoices.map(i => i.total === null ?
          { match: JSON.stringify(i.id), call: blockedCall('The invoice does not state its total amount.') } :
          { match: JSON.stringify(i.id), value: i.total }) },
      root: { name: 'invoice_totals', args: {}, returns: 'Record<string, number | null>',
        instructions: 'Record the total amount due that each invoice from payables.invoices() states, in euros. Read each invoice in its own judgment. When an invoice does not state its total, record null for it; never compute a figure it does not state. Return a record from invoice id to amount.' },
      files: { 'invoice_totals/payables.ts': `const INVOICES = ${literal(plain)};\n/** This month's invoices, as text. */\nexport function invoices(): { id: string, text: string }[] { return INVOICES; }\n` },
      inputs: {}, expected });
  });
}

/**
 * Application event sequencing: an event is committed, then its view is rendered. A transient view failure is
 * retried without applying the event again (its revision proves it); a rejected event is reported as failed.
 */
export function eventRetry(seed, index) {
  const rng = new Random(seed, `event:${index}`);
  const [card] = nonceWords(rng, 1);
  const start = rng.int(10, 40);
  const shape = `board${index}`;
  const column = rng.pick(['review', 'done', 'blocked']);
  const variants = {
    view_timeout: { reject: false, expected: { revision: start + 1, column }, operation: undefined },
    rejected: { reject: true, expected: null, operation: 'blocked' },
  };
  return Object.entries(variants).map(([variant, spec]) => {
    const module = `type Card = { id: string, column: string };
const cards: Card[] = [{ id: ${JSON.stringify(card)}, column: "doing" }];
let revision = ${start};
let renders = 0;
const REJECT = ${spec.reject};
/** Apply a move event: move card to column. Commits it and returns the new board revision. */
export function commit_move(event: { card: string, to: string }): { revision: number } {
  if (REJECT) throw new Error("event rejected: card " + event.card + " is locked by an open audit");
  const found = cards.find(item => item.id === event.card);
  if (!found) throw new Error("no such card");
  found.column = event.to;
  revision += 1;
  return { revision };
}
/** Render the board view: its revision and each card's column. */
export function render(): { revision: number, cards: Card[] } {
  renders += 1;
  if (renders === 1) throw new Error("view renderer timed out; the board state is unaffected");
  return { revision, cards: cards.map(item => ({ ...item })) };
}
`;
    const applyCode = `const applied = board.commit_move({ card: ${JSON.stringify(card)}, to: ${JSON.stringify(column)} });
let view;
try { view = board.render(); } catch (error) { view = String(error); }
({ applied, view })`;
    const reference = spec.reject ?
      [evalCall(applyCode), failedCall(`The move was rejected: card ${card} is locked by an open audit.`)] :
      [evalCall(applyCode), evalCall(`const view2 = board.render();
return { revision: view2.revision, column: view2.cards.find(item => item.id === ${JSON.stringify(card)})!.column };`), returnCall(spec.expected)];
    return curriculumCase({ family: 'event_retry', shape, variant, pairGroup: `event:${shape}`,
      slice: 'folder_failure', domain: 'actor', mode: 'followup', inline: 'avoid',
      evidence: { world: [spec.reject ? 'the event is rejected' : 'the first render times out'], retrieved: [], background: [] },
      decisive: [{ marker: spec.reject ? 'locked by an open audit' : 'view renderer timed out', source: 'error', note: spec.reject ? 'the event is rejected' : 'only the view failed' }],
      plausibleActions: ['apply the event again', 'render again', 'report failure'],
      minimumSequence: spec.reject ? ['apply the event', 'see the rejection', 'report it'] : ['apply the event once', 'see the view time out', 'render again without re-applying'],
      reference: { root: reference },
      root: { name: 'move_card', args: {}, returns: '{ revision: number, column: string }',
        instructions: `Move card ${card} to column "${column}" with board.commit_move, then render the board with board.render() and return the rendered revision and the card's column. The move must be applied exactly once.` },
      files: { 'move_card/board.ts': module },
      inputs: {}, expected: spec.expected, ...(spec.operation ? { operation: spec.operation } : {}) });
  });
}

/**
 * A seeded eval measures an org chart's depth with a recursive helper, which natlang code rejects; the repair
 * walks the chart with an explicit work list (a counted loop over the known size, or iterateOn).
 */
export function recursionRewrite(seed, index) {
  const rng = new Random(seed, `recursion:${index}`);
  const shape = `org${index}`;
  const build = depth => {
    const people = nonceWords(rng, 14).map(word => word[0].toUpperCase() + word.slice(1));
    const nodes = [{ name: people[0], manager: null }];
    // A spine of the given depth, then the rest attached at random above it.
    for (let i = 1; i < depth; i++) nodes.push({ name: people[i], manager: people[i - 1] });
    for (let i = depth; i < people.length; i++) nodes.push({ name: people[i], manager: rng.pick(nodes.slice(0, Math.max(1, depth - 2))).name });
    return rng.shuffle(nodes);
  };
  const depthOf = nodes => {
    const manager = Object.fromEntries(nodes.map(n => [n.name, n.manager]));
    return Math.max(...nodes.map(n => { let d = 1, m = manager[n.name]; while (m) { d++; m = manager[m]; } return d; }));
  };
  const seedCode = `const chart = org.people();
const levels = (name: string): number => {
  const reports = chart.filter(p => p.manager === name);
  return reports.length ? 1 + Math.max(...reports.map(r => levels(r.name))) : 1;
};
levels(chart.find(p => p.manager === null)!.name)`;
  return [['shallow', build(rng.int(3, 4))], ['deep', build(rng.int(6, 8))]].map(([variant, nodes]) => {
    const expected = depthOf(nodes);
    return curriculumCase({ family: 'recursion_rewrite', shape, variant, pairGroup: `recursion:${shape}`,
      slice: 'folder_failure', domain: 'other', mode: 'followup', inline: 'avoid',
      evidence: { world: [`depth ${expected}`], retrieved: ['the recursion rejection'], background: [] },
      decisive: [{ marker: 'recursion is not allowed', source: 'error', note: 'natlang code rejects the recursive helper' }],
      plausibleActions: ['retry the recursive helper', 'rewrite it with an explicit work list', 'guess the depth'],
      minimumSequence: ['read the rejection', 'compute each person\'s chain length with a bounded loop'],
      reference: { root: [evalCall(`const chart = org.people();
const managerOf: Record<string, string | null> = {};
for (const p of chart) managerOf[p.name] = p.manager;
let deepest = 0;
for (const p of chart) {
  let length = 1;
  let manager = managerOf[p.name];
  for (let step = 0; step < chart.length && manager !== null; step++) { length++; manager = managerOf[manager]; }
  deepest = Math.max(deepest, length);
}
return deepest;`), returnCall(expected)] },
      root: { name: 'chart_depth', args: {}, returns: 'number',
        instructions: 'How many levels does the organisation chart in org.people() have? The top person has no manager and is level 1; each report is one level below their manager.' },
      files: { 'chart_depth/org.ts': `const PEOPLE = ${literal(nodes)};\n/** Everyone in the organisation and their manager (null for the top). */\nexport function people(): { name: string, manager: string | null }[] { return PEOPLE; }\n` },
      inputs: {}, expected, failureSeed: { kind: 'compile', code: seedCode } });
  });
}

/**
 * A directory reducer applied to the one subfolder a semantic reading selects: the release notes say which
 * package changed, and only that package's version is bumped (by the bump_version reducer, run on its folder).
 */
export function reducerApply(seed, index) {
  const rng = new Random(seed, `reducer:${index}`);
  const packages = rng.sample(['api', 'web', 'worker', 'cli', 'sdk'], 3);
  const versions = Object.fromEntries(packages.map(name => [name, `1.${rng.int(0, 9)}.${rng.int(0, 9)}`]));
  const bump = version => version.replace(/\d+$/, n => String(Number(n) + 1));
  const manifest = name => `{\n  "name": "@acme/${name}",\n  "version": "${versions[name]}"\n}\n`;
  const shape = `repo${index}`;
  const changes = {
    [packages[0]]: `Fixed a crash in @acme/${packages[0]} when the request body is empty. Documentation for @acme/${packages[1]} was proofread (no code change).`,
    [packages[1]]: `@acme/${packages[1]} now retries failed uploads. The @acme/${packages[0]} README gained a diagram (no code change).`,
  };
  // Both variants' notes have the same size: a directory reducer's opening lists file sizes.
  const width = Math.max(...Object.values(changes).map(text => text.length));
  return Object.entries(changes).map(([changed, notes]) => {
    const folderFiles = { 'RELEASE_NOTES.md': `# Unreleased\n\n${notes}\n${' '.repeat(width - notes.length)}`, ...Object.fromEntries(packages.map(name => [`packages/${name}/package.json`, manifest(name)])) };
    const expectedFiles = { ...folderFiles, [`packages/${changed}/package.json`]: manifest(changed).replace(versions[changed], bump(versions[changed])) };
    return curriculumCase({ family: 'reducer_apply', shape, variant: changed, pairGroup: `reducer:${shape}`,
      slice: 'folder_failure', domain: 'other', mode: 'followup', inline: 'avoid', named: 'required',
      evidence: { world: [notes], retrieved: ['the release notes'], background: [] },
      decisive: [{ marker: notes.slice(0, 40), source: 'file', note: 'the notes say which package changed code' }],
      plausibleActions: ['bump every package', 'bump the package named first', 'bump only the package with a code change'],
      minimumSequence: ['read the release notes', 'decide which package changed code', 'apply bump_version to that package\'s folder'],
      reference: { root: [['read_file', { path: 'RELEASE_NOTES.md' }], evalCall(`await folder.dir("packages/${changed}").apply(bump_version)`),
        returnCall(`packages/${changed}`)],
        children: [{ match: ['You are inside this call: bump_version'], calls: [
          ['edit_file', { path: 'package.json', find: `"version": "${versions[changed]}"`, replace_with: `"version": "${bump(versions[changed])}"` }]],
          value: bump(versions[changed]) }] },
      root: { name: 'release', kind: 'directory-reducer', args: {}, returns: 'string',
        instructions: 'Prepare the release: RELEASE_NOTES.md describes the unreleased changes. Bump the version of each package whose code changed, and only those, by applying bump_version to that package\'s folder (packages/<name>). Return the folder of each bumped package (for one package, just its path).' },
      files: { 'release/bump_version.nl': nlFile({ kind: 'directory-reducer', args: {}, returns: 'string',
        description: 'Bump the patch version in this package\'s package.json.',
        instructions: 'Increase the last number of "version" in package.json by one, and return the new version.' }) },
      folderFiles, expectedFiles, inputs: {}, expected: `packages/${changed}` });
  });
}
