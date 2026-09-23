// TypeScript authoring track: a directory reducer edits a small project so that a module implements a function
// with nl, a named callable function, or iterateOn. Acceptance runs what was written (semantics.authoring):
// the export is called on test inputs and its natural-language children are answered by an oracle keyed by
// argument values. Rows carry curriculum.track "authoring" and stay out of interpreter shards by default.
import { capitalize, curriculumCase, literal, nlFile, nonceWords, Random, returnCall } from './lib.mjs';

const writeFile = (path, content) => ['write_file', { path, content }];

function authoringCase({ family, shape, variant, instructions, files, module, spec, solution, slice = 'folder_failure', domain = 'other' }) {
  // The authored code's nl calls happen in the acceptance run, not the trajectory: spec.requires enforces them.
  const inline = 'optional';
  const record = curriculumCase({ family, shape, variant, slice, domain, mode: 'single_call', inline,
    evidence: { world: [], retrieved: [], background: [`authoring: ${module}`] },
    minimumSequence: ['read the project', `implement the function in ${module}`, 'finish'],
    reference: { root: [['read_file', { path: module }], writeFile(module, solution), returnCall(`Implemented ${spec.export} in ${module}.`)] },
    root: { name: 'implement', kind: 'directory-reducer', args: {}, returns: 'string', instructions },
    folderFiles: files, inputs: {}, expected: null });
  record.curriculum.track = 'authoring';
  record.semantics.authoring = spec;
  return record;
}

const TICKETS = [
  { text: 'Checkout has failed for every customer since 9:00 this morning.', live: true },
  { text: 'Could you add a dark mode to the dashboard?', live: false },
  { text: 'The export button crashed yesterday but works again since your fix.', live: false },
  { text: 'Password reset emails are not arriving at all right now.', live: true },
  { text: 'It would be great to have keyboard shortcuts for search.', live: false },
  { text: 'Our invoices page shows a blank screen for all users today.', live: true },
  { text: 'Thanks, the login loop from last week is solved.', live: false },
  { text: 'Uploads above 2 MB are failing with a timeout this afternoon.', live: true },
];

/** Implement a per-item semantic filter with an inline nl function in a module. */
export function authoringInlineReview(seed, index) {
  const rng = new Random(seed, `author-review:${index}`);
  const draw = set => rng.sample(TICKETS, 5).map((ticket, i) => ({ id: `T${index}-${set}${i + 1}`, ...ticket }));
  const sets = [draw('a'), draw('b')];
  const plain = set => set.map(({ id, text }) => ({ id, text }));
  const all = sets.flat();
  const stub = `import type { Ticket } from "./types";

/**
 * The ids of the tickets that report a problem customers are experiencing right now (not feature requests,
 * not problems already fixed), in input order.
 */
export async function live_incidents(tickets: Ticket[]): Promise<string[]> {
  // TODO: implement.
  return [];
}
`;
  const solution = `import type { Ticket } from "./types";

/**
 * The ids of the tickets that report a problem customers are experiencing right now (not feature requests,
 * not problems already fixed), in input order.
 */
export async function live_incidents(tickets: Ticket[]): Promise<string[]> {
  const live = await Promise.all(tickets.map(ticket =>
    nl\`Does ticket report a problem customers are experiencing right now, rather than a feature request or a problem already fixed?\`(ticket)));
  return tickets.filter((ticket, i) => live[i]).map(ticket => ticket.id);
}
`;
  return [authoringCase({ family: 'authoring_inline_review', shape: `review${index}`, variant: 'a', module: 'review.ts', inline: 'required',
    instructions: 'Implement live_incidents in review.ts as its doc comment describes. Whether a ticket reports a live problem is a judgment about its meaning: make it inside the function with an inline nl function, one ticket at a time, and keep the exact computation in TypeScript. Keep the signature.',
    files: { 'review.ts': stub, 'types.ts': 'export type Ticket = { id: string, text: string };\n' },
    spec: { module: 'review.ts', export: 'live_incidents', requires: { nl: true },
      runs: sets.map(set => ({ args: [plain(set)], expected: set.filter(t => t.live).map(t => t.id) })),
      oracle: all.map(t => ({ match: [JSON.stringify(t.id)], value: t.live })) },
    solution })];
}

/** Implement an open-ended search with iterateOn (modules may not use while or recursion). */
export function authoringIterate(seed, index) {
  const rng = new Random(seed, `author-iterate:${index}`);
  const used = new Set();
  const id = () => `n-${nonceWords(rng, 1, used)[0]}`;
  const links = {};
  const chain = (length, start) => {
    let previous = start;
    for (let i = 0; i < length; i++) { const node = id(); (links[previous] ??= []).push(node); links[node] ??= []; previous = node; }
    return previous;
  };
  const origin = id();
  const near = chain(3, origin);
  const far = chain(12, origin);
  const island = id(); links[island] = [id()];
  for (const key of Object.keys(links)) links[key] = rng.shuffle(links[key]);
  const graph = `const LINKS: Record<string, string[]> = ${literal(links)};\n/** The nodes that node links to (outgoing links only). */\nexport function links(node: string): string[] { return LINKS[node] ?? []; }\n`;
  const stub = `/**
 * The smallest number of links to follow from node from to reach node to (graph.links lists a node's outgoing
 * links), or null when to cannot be reached.
 */
export async function distance(from: string, to: string): Promise<number | null> {
  // TODO: implement. Callable modules may not use while loops or recursion.
  return null;
}
`;
  const solution = `import { links } from "./graph.js";

/**
 * The smallest number of links to follow from node from to reach node to (graph.links lists a node's outgoing
 * links), or null when to cannot be reached.
 */
export async function distance(from: string, to: string): Promise<number | null> {
  type Search = { frontier: string[], seen: string[], depth: number, found: boolean };
  const expand = (state: Search): Search => {
    const next: string[] = [];
    for (const node of state.frontier) for (const link of links(node)) if (!state.seen.includes(link) && !next.includes(link)) next.push(link);
    return { frontier: next, seen: [...state.seen, ...next], depth: state.depth + 1, found: next.includes(to) };
  };
  const final = await iterateOn(expand, { frontier: [from], seen: [from], depth: 0, found: from === to })
    .until(state => state.found || state.frontier.length === 0);
  return final.found ? final.depth : null;
}
`;
  return [authoringCase({ family: 'authoring_iterate', shape: `graph${index}`, variant: 'a', module: 'search.ts', slice: 'iterate', domain: 'relational', inline: 'avoid',
    instructions: 'Implement distance in search.ts as its doc comment describes, using graph.links from graph.ts. The search has no bound known in advance, and callable modules may not use while loops or recursion: step it with iterateOn. Keep the signature.',
    files: { 'search.ts': stub, 'graph.ts': graph },
    spec: { module: 'search.ts', export: 'distance', requires: { iterateOn: true },
      runs: [{ args: [origin, near], expected: 3 }, { args: [origin, far], expected: 12 }, { args: [origin, island], expected: null }],
      oracle: [{ match: ['An iterative process'], value: { verdict: 'continue', reason: 'Each step reaches new nodes.' } }] },
    solution })];
}

const REVIEWS = [
  { text: 'Arrived broken and support never answered.', label: 'negative' },
  { text: 'Does exactly what it says, very happy.', label: 'positive' },
  { text: 'It is fine, nothing special either way.', label: 'neutral' },
  { text: 'Stopped working after two days; waste of money.', label: 'negative' },
  { text: 'Great value and it arrived early.', label: 'positive' },
  { text: 'The colour is slightly different from the photo, otherwise okay.', label: 'neutral' },
  { text: 'Terrible smell, returned it immediately.', label: 'negative' },
  { text: 'My kids love it and use it every day.', label: 'positive' },
];

/** Implement an aggregate that must use the folder's existing sentiment function rather than a new judgment. */
export function authoringNamedHelper(seed, index) {
  const rng = new Random(seed, `author-named:${index}`);
  const [product] = nonceWords(rng, 1).map(capitalize);
  const draw = set => rng.sample(REVIEWS, 5).map((review, i) => ({ id: `R${index}-${set}${i + 1}`, ...review }));
  const sets = [draw('a'), draw('b')];
  const share = set => Math.round(set.filter(r => r.label === 'negative').length / set.length * 100) / 100;
  const stub = `import type { Review } from "./types";

/** The share of reviews whose sentiment is negative, rounded to two decimals (0 for no reviews). */
export async function negative_share(reviews: Review[]): Promise<number> {
  // TODO: implement.
  return 0;
}
`;
  const solution = `import type { Review } from "./types";
import sentiment from "./sentiment.nl";

/** The share of reviews whose sentiment is negative, rounded to two decimals (0 for no reviews). */
export async function negative_share(reviews: Review[]): Promise<number> {
  if (reviews.length === 0) return 0;
  const labels = await Promise.all(reviews.map(review => sentiment(review)));
  const negative = labels.filter(label => label === "negative").length;
  return Math.round(negative / reviews.length * 100) / 100;
}
`;
  return [authoringCase({ family: 'authoring_named_helper', shape: `reviews${index}`, variant: 'a', module: 'report.ts', slice: 'nested_scoped', inline: 'avoid',
    instructions: `Implement negative_share in report.ts as its doc comment describes, for the ${product} review report. The folder already has a sentiment function for reviews; use it rather than writing a new judgment. Keep the signature.`,
    files: { 'report.ts': stub, 'types.ts': 'export type Review = { id: string, text: string };\nexport type Sentiment = "positive" | "neutral" | "negative";\n',
      'sentiment.nl': nlFile({ args: { review: 'Review' }, returns: 'Sentiment', description: 'The sentiment of one product review.',
        instructions: 'Classify the sentiment of review as positive, neutral, or negative.' }) },
    spec: { module: 'report.ts', export: 'negative_share', requires: { calls: ['sentiment'], noNl: true },
      runs: sets.map(set => ({ args: [set.map(({ id, text }) => ({ id, text }))], expected: share(set) })),
      oracle: sets.flat().map(r => ({ match: [JSON.stringify(r.id)], value: r.label })) },
    solution })];
}
