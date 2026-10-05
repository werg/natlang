// Semantic workbench: a library over a collection (a crisp service with the data and exact helpers, plus named nl
// helpers, some of them near misses) and a request composed at runtime from an operation grammar. The request's
// semantic criteria are drawn per case, so no helper answers them: the natural program maps ad hoc nl lambdas over
// the items and combines the results with crisp code. Every item carries hidden labels; the expected answer is the
// composed operation applied to them. Paired worlds share the library, the crisp fields and the request, and differ
// in a few items' hidden labels, so the answer differs. (Owner 2026-10-05: far more inline lambdas, with the work
// formulated at runtime over libraries of crisp and natlang functions.)
import { Random, curriculumCase, evalCall, literal, nlFile, returnCall } from './lib.mjs';

const pick = (rng, list) => list[rng.int(0, list.length - 1)];

// Domains --------------------------------------------------------------------------------------------------------
// attrs: hidden labels. Boolean attrs have `yes`/`no` sentence banks (`no` holds near misses; an empty string means the
// item says nothing about it); categorical attrs have a bank per value. `ask` is the lambda's question, `clause` the
// request's wording, `key` a distinctive fragment of the question (it identifies the child call in the reference).

const TICKETS = {
  name: 'tickets', root: 'support_desk', service: 'helpdesk', noun: 'ticket', plural: 'tickets', type: 'Ticket',
  crisp: { plan: ['free', 'pro', 'enterprise'], seats: [3, 250], opened: ['2026-03-02', '2026-03-27'] },
  group: 'plan', numeric: 'seats', date: 'opened',
  customers: ['Arbor Labs', 'Brightline', 'Cobalt Foods', 'Delta Print', 'Evergreen Clinic', 'Fjord Travel', 'Granite Legal',
    'Harbor Media', 'Ivy Schools', 'Juniper Bank', 'Kestrel Air', 'Lumen Studio', 'Maple Freight', 'Nimbus Retail'],
  attrs: {
    kind: { values: ['bug', 'billing', 'how_to', 'feature_request'], ask: 'Classify ticket as "bug", "billing", "how_to" or "feature_request".',
      key: 'Classify ticket as', helper: true,
      bank: {
        bug: ['The {feature} page throws an error whenever we {action}.', 'Since yesterday\'s update, {feature} shows the wrong numbers.',
          '{Feature} crashes as soon as we {action}.', 'When we {action}, {feature} just spins forever and never loads.'],
        billing: ['We were charged twice for {month}.', 'Our invoice lists {n} seats but we only have {m}.',
          'The renewal price on our invoice does not match the quote we signed.', 'We downgraded in {month} but are still billed at the old rate.'],
        how_to: ['How do I {task}? I could not find it in the docs.', 'Is there a way to {task} that I am missing?',
          'Could someone walk me through how to {task}?', 'I am new here: what is the right way to {task}?'],
        feature_request: ['It would be great if {feature} could {wish}.', 'Please consider adding a way to {wish}.',
          'We would love an option to {wish}; today there is none.', 'Any plans to let {feature} {wish}? It would save us hours.'],
      } },
    churn: { ask: 'Does ticket say or imply that the customer may cancel or move to another provider?', key: 'cancel or move to another provider',
      clause: 'signal that the customer may cancel or leave', yes: ['If this is not fixed this week we will move to another provider.',
        'Honestly, we are evaluating alternatives at this point.', 'Our contract is up next month and this makes renewal hard to justify.',
        'We are considering cancelling our subscription over this.', 'Two of our managers already asked me to look at other tools.'],
      no: ['', 'Thanks for the help, otherwise we are very happy with the product.', 'We cancelled our team meeting to look into this, so a quick reply would help.',
        'We moved our data over to the new workspace last week and that went smoothly.', 'We are not going anywhere, we just want this fixed.',
        'Our previous provider never had support this responsive, so thanks in advance.'] },
    blocking: { ask: 'Is the customer\'s work blocked right now by what ticket describes (no workaround)?', key: 'work blocked right now',
      clause: 'describe work that is blocked right now', yes: ['Our whole team is stuck until this is sorted.', 'We cannot ship anything today because of this.',
        'Nobody on our side can work right now.', 'This stops our month-end close completely.'],
      no: ['', 'It is not urgent; we have a workaround for now.', 'No rush, whenever you get to it.', 'It is not blocking us, but it is annoying.',
        'We can live with it for a few weeks.'] },
  },
  fill: { feature: ['the reports', 'the dashboard', 'the export', 'the calendar sync', 'the search', 'the invoice view'],
    action: ['filter by date', 'open an old project', 'add a second admin', 'switch languages', 'upload a CSV'],
    task: ['share a dashboard with a client', 'reset a user\'s two-factor login', 'change our notification email', 'archive old projects'],
    wish: ['export to PDF', 'schedule reports weekly', 'group items by tag', 'show totals per team', 'remember the last filter'],
    month: ['January', 'February', 'March'] },
  helpers: { kind: { name: 'ticket_kind', description: 'Classify a support ticket.', returns: 'Kind',
    instructions: 'Classify ticket: "bug" when something is broken, "billing" for charges and invoices, "how_to" for questions about using the product, "feature_request" for something the product does not do yet.' } },
  nearMiss: { name: 'ticket_sentiment', description: 'How the customer feels in a support ticket.', returns: 'Sentiment',
    args: { ticket: 'Ticket' }, instructions: 'Rate the customer\'s tone in ticket: "angry", "neutral" or "positive".' },
  types: 'export type Ticket = { id: string, customer: string, plan: "free" | "pro" | "enterprise", seats: number, opened: string, text: string };\n' +
    'export type Kind = "bug" | "billing" | "how_to" | "feature_request";\nexport type Sentiment = "angry" | "neutral" | "positive";\n',
};

const REVIEWS = {
  name: 'reviews', root: 'review_desk', service: 'shop', noun: 'review', plural: 'reviews', type: 'Review',
  crisp: { product: ['Trail Jacket', 'Desk Lamp', 'Coffee Grinder', 'Yoga Mat'], price: [18, 140], posted: ['2026-02-01', '2026-02-28'] },
  group: 'product', numeric: 'price', date: 'posted',
  customers: ['ana_k', 'bjorn77', 'cleo.m', 'dev_r', 'emi', 'farid', 'gwen_t', 'hugo', 'isa_b', 'jonas', 'kemi', 'lior', 'mara', 'nils'],
  attrs: {
    complaint: { values: ['shipping', 'quality', 'sizing', 'none'], ask: 'What does review mainly complain about: "shipping", "quality", "sizing", or "none"?',
      key: 'mainly complain about', helper: false,
      bank: {
        shipping: ['It took three weeks to arrive.', 'The box showed up crushed and the item was scratched.', 'Delivery was a week later than promised.',
          'The courier left it in the rain and the packaging was soaked.'],
        quality: ['The stitching came apart after a few uses.', 'It stopped working after two weeks.', 'A part snapped the first time I used it.',
          'The finish started peeling within a month.'],
        sizing: ['It runs at least a size small.', 'Way too big even though I ordered my usual size.', 'The fit is off; I had to exchange it for a larger one.',
          'The dimensions on the listing are wrong, it is much smaller in person.'],
        none: ['Arrived on time and works exactly as described.', 'Shipping was quick and it feels solid.', 'I worried the size would be off, but it is perfect.',
          'Exactly what I expected, no surprises.'],
      } },
    recommends: { ask: 'Would the reviewer recommend the product, judging by review?', key: 'reviewer recommend the product',
      clause: 'recommend the product', yes: ['I would recommend it to anyone.', 'Would buy again without hesitation.', 'Already told my sister to get one.',
        'Despite that, I would still buy it again.'],
      no: ['I would not buy it again.', 'Cannot recommend it.', 'Save your money.', 'My friend recommended it, but I would not.'] },
    gift: { ask: 'Was the product bought as a gift for someone else, according to review?', key: 'bought as a gift',
      clause: 'were bought as a gift', yes: ['Bought this as a birthday present for my dad.', 'Got it for my partner for our anniversary.',
        'It was a gift for a colleague who is retiring.'],
      no: ['', 'Bought it for myself after my old one broke.', 'I almost gave it away as a gift, but kept it for myself.', 'Treated myself to this one.'] },
  },
  fill: {},
  helpers: {},
  nearMiss: { name: 'star_summary', description: 'Summarize a review in one short sentence.', returns: 'string',
    args: { review: 'Review' }, instructions: 'Summarize review in one short sentence.' },
  types: 'export type Review = { id: string, author: string, product: string, price: number, posted: string, text: string };\n' +
    'export type Complaint = "shipping" | "quality" | "sizing" | "none";\n',
};

const APPLICANTS = {
  name: 'applicants', root: 'hiring_desk', service: 'ats', noun: 'applicant', plural: 'applicants', type: 'Applicant',
  crisp: { city: ['Berlin', 'Lisbon', 'Warsaw', 'Dublin'], salary: [48, 120], applied: ['2026-01-05', '2026-01-30'] },
  group: 'city', numeric: 'salary', date: 'applied',
  customers: ['Aiko Tan', 'Ben Osei', 'Carla Ruiz', 'Dima Petrov', 'Elif Kaya', 'Femi Ade', 'Greta Holm', 'Hamid Rahimi', 'Ines Silva',
    'Jae Park', 'Kofi Mensah', 'Lena Vogel', 'Marco Bianchi', 'Nora Lind'],
  attrs: {
    focus: { values: ['data', 'frontend', 'infrastructure'], ask: 'What is applicant\'s main area: "data", "frontend" or "infrastructure"?',
      key: 'main area', helper: true,
      bank: {
        data: ['Most of my work is building data pipelines and SQL models for analytics.', 'I spend my days on dashboards, ETL jobs and statistical analysis.',
          'I own our warehouse models and the metrics layer the business reads.'],
        frontend: ['I build React interfaces and care a lot about accessibility.', 'My focus is web UIs: component libraries, CSS and browser performance.',
          'I turn designs into fast, accessible pages and maintain our design system.'],
        infrastructure: ['I run our Kubernetes clusters and the CI/CD pipelines.', 'I am responsible for cloud infrastructure, Terraform and on-call.',
          'I keep our deploys, networking and monitoring running.'],
      } },
    led: { ask: 'Has applicant formally led or managed other people, according to their profile?', key: 'formally led or managed',
      clause: 'have led or managed people', yes: ['For the last two years I managed a team of five engineers.', 'As team lead I ran hiring and weekly one-on-ones.',
        'I led the platform group of four after our lead left.'],
      no: ['', 'I have always preferred hands-on work over managing people.', 'I reported to the team lead and focused on my own projects.',
        'I worked closely with our manager on planning, but nobody reported to me.'] },
    remote: { ask: 'Does applicant want a remote role?', key: 'want a remote role', clause: 'want a remote role',
      yes: ['I am looking for a fully remote role.', 'I would only consider remote positions.', 'Remote work is a must for me because of family.'],
      no: ['I am happy to be in the office most days.', 'My current job is remote, but I would like to be back in an office.',
        'I am relocating and would like to work on-site.'] },
  },
  fill: {},
  helpers: { focus: { name: 'main_area', description: 'The main technical area of an applicant.', returns: 'Area',
    instructions: 'Return applicant\'s main area: "data" for analytics and pipelines, "frontend" for web interfaces, "infrastructure" for cloud, deploys and operations.' } },
  nearMiss: { name: 'seniority', description: 'Estimate an applicant\'s seniority.', returns: 'Level',
    args: { applicant: 'Applicant' }, instructions: 'Estimate applicant\'s seniority from their profile: "junior", "mid" or "senior".' },
  types: 'export type Applicant = { id: string, name: string, city: string, salary: number, applied: string, text: string };\n' +
    'export type Area = "data" | "frontend" | "infrastructure";\nexport type Level = "junior" | "mid" | "senior";\n',
};

export const DOMAINS = { tickets: TICKETS, reviews: REVIEWS, applicants: APPLICANTS };

// Worlds ---------------------------------------------------------------------------------------------------------
const crispNames = d => Object.keys(d.crisp);
const boolAttrs = d => Object.entries(d.attrs).filter(([, a]) => !a.values).map(([k]) => k);
const catAttr = d => Object.entries(d.attrs).find(([, a]) => a.values)[0];
const capital = s => s[0].toUpperCase() + s.slice(1);

function fillSlots(rng, d, text, item) {
  return text.replace(/\{(\w+)\}/g, (_, slot) => {
    const lower = slot.toLowerCase();
    if (lower === 'n') return String(item.seats ?? 5);
    if (lower === 'm') return String(Math.max(1, (item.seats ?? 5) - rng.int(1, 3)));
    const value = pick(rng, d.fill[lower] ?? ['it']);
    return slot === lower ? value : capital(value.replace(/^the /, ''));
  });
}

function render(rng, d, item, labels) {
  const parts = [];
  for (const [attr, spec] of Object.entries(d.attrs)) {
    const sentence = spec.values ? pick(rng, spec.bank[labels[attr]]) : pick(rng, labels[attr] ? spec.yes : spec.no);
    if (sentence) parts.push(fillSlots(rng, d, sentence, item));
  }
  return rng.shuffle(parts).join(' ');
}

function crispItem(rng, d, index, people) {
  const [a, b, c] = crispNames(d);
  const [lo, hi] = d.crisp[b];
  const [start, end] = d.crisp[c].map(day => Date.parse(day + 'T00:00:00Z'));
  const day = new Date(start + rng.int(0, Math.round((end - start) / 86400000)) * 86400000).toISOString().slice(0, 10);
  const who = d.name === 'tickets' ? 'customer' : d.name === 'reviews' ? 'author' : 'name';
  return { id: `${d.noun[0].toUpperCase()}${index + 1}`, [who]: people[index], [a]: pick(rng, d.crisp[a]), [b]: rng.int(lo, hi), [c]: day };
}

function labelsFor(rng, d) {
  const labels = {};
  for (const [attr, spec] of Object.entries(d.attrs)) labels[attr] = spec.values ? pick(rng, spec.values) : rng.next() < 0.4;
  return labels;
}

// Operations -----------------------------------------------------------------------------------------------------
// A request: an optional crisp scope, one or two semantic predicates, and an output. `plan` builds the instruction,
// the reference program and the expected value from hidden labels.

function scopeFor(rng, d) {
  const roll = rng.next();
  const [group, , date] = [d.group, d.numeric, d.date];
  if (roll < 0.35) return null;
  if (roll < 0.7) {
    const value = pick(rng, d.crisp[group]);
    return { text: `with ${group} "${value}"`, code: `x.${group} === ${JSON.stringify(value)}`, test: x => x[group] === value };
  }
  const [start, end] = d.crisp[date];
  const mid = new Date((Date.parse(start) + Date.parse(end)) / 2).toISOString().slice(0, 10);
  return { text: `${date} on or after ${mid}`, code: `x.${date} >= ${JSON.stringify(mid)}`, test: x => x[date] >= mid };
}

const OUTPUTS = ['count', 'ids', 'group_count', 'sum', 'classify_count', 'extract', 'top_group'];

function request(rng, d) {
  const output = pick(rng, OUTPUTS);
  const scope = scopeFor(rng, d);
  const bools = rng.shuffle(boolAttrs(d));
  const preds = (output === 'classify_count' ? [] : bools.slice(0, rng.next() < 0.3 ? 2 : 1))
    .map(attr => ({ attr, want: rng.next() < 0.8 }));
  return { output, scope, preds };
}

function clauseOf(d, pred) {
  const spec = d.attrs[pred.attr];
  return pred.want ? spec.clause : `do not ${spec.clause}`;
}

function describe(d, req) {
  const cat = catAttr(d), spec = d.attrs[cat];
  const scoped = req.scope ? `the ${d.plural} ${req.scope.text}` : `all ${d.plural}`;
  const which = req.preds.length ? ` that ${req.preds.map(p => clauseOf(d, p)).join(' and ')}` : '';
  const among = `Among ${scoped}, consider those${which}.`;
  switch (req.output) {
    case 'count': return { text: `${among} How many are there?`, returns: 'number' };
    case 'ids': return { text: `${among} Return their ids sorted by ${d.date} (oldest first), ties by id.`, returns: 'string[]' };
    case 'group_count': return { text: `${among} Count them per ${d.group}; leave out ${d.group} values with none.`, returns: 'Record<string, number>' };
    case 'sum': return { text: `${among} Return the total ${d.numeric} across them.`, returns: 'number' };
    case 'classify_count': return { text: `Classify each of ${scoped} by ${cat} (${spec.values.map(v => `"${v}"`).join(', ')}) and count them per ${cat}, including every ${cat} with 0 when there are none.`, returns: 'Record<string, number>' };
    case 'extract': return { text: `${among} For each, return { id, ${cat} } where ${cat} is one of ${spec.values.map(v => `"${v}"`).join(', ')}, sorted by id.`, returns: `{ id: string, ${cat}: string }[]` };
    case 'top_group': return { text: `${among} Which ${d.group} has the most of them? Break ties alphabetically; return null if there are none.`, returns: 'string | null' };
  }
}

const byId = (a, b) => Number(a.id.slice(1)) - Number(b.id.slice(1));

function evaluate(d, req, items) {
  const cat = catAttr(d);
  const scoped = items.filter(x => !req.scope || req.scope.test(x));
  const kept = scoped.filter(x => req.preds.every(p => x.labels[p.attr] === p.want));
  switch (req.output) {
    case 'count': return kept.length;
    case 'ids': return [...kept].sort((a, b) => a[d.date].localeCompare(b[d.date]) || byId(a, b)).map(x => x.id);
    case 'group_count': { const out = {}; for (const x of kept) out[x[d.group]] = (out[x[d.group]] ?? 0) + 1; return out; }
    case 'sum': return kept.reduce((s, x) => s + x[d.numeric], 0);
    case 'classify_count': { const out = Object.fromEntries(d.attrs[cat].values.map(v => [v, 0])); for (const x of scoped) out[x.labels[cat]]++; return out; }
    case 'extract': return [...kept].sort(byId).map(x => ({ id: x.id, [cat]: x.labels[cat] }));
    case 'top_group': {
      const counts = {}; for (const x of kept) counts[x[d.group]] = (counts[x[d.group]] ?? 0) + 1;
      const best = Object.entries(counts).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0];
      return best ? best[0] : null;
    }
  }
}

/** The reference program: crisp scope, one lambda per item per predicate (concurrently), typed lambdas or the named
 * helper for the categorical attribute, then crisp aggregation. */
function program(d, req) {
  const cat = catAttr(d), spec = d.attrs[cat];
  const helper = d.helpers[cat];
  const lines = [`const all = ${d.service}.${d.plural}();`, `const scoped = all.filter(x => ${req.scope ? req.scope.code : 'true'});`];
  let kept = 'scoped';
  req.preds.forEach((p, i) => {
    const ask = d.attrs[p.attr].ask.replace(/`/g, '\\`');
    lines.push(`const p${i} = await Promise.all(${kept}.map(${d.noun} => nl\`${ask}\`(${d.noun})));`);
    lines.push(`const k${i} = ${kept}.filter((_, j) => ${p.want ? '' : '!'}p${i}[j]);`);
    kept = `k${i}`;
  });
  const classify = list => helper ? `await Promise.all(${list}.map(${d.noun} => ${helper.name}(${d.noun})))`
    : `await Promise.all(${list}.map(${d.noun} => nl<${capital(cat)}>\`${spec.ask.replace(/`/g, '\\`')}\`(${d.noun})))`;
  switch (req.output) {
    case 'count': lines.push(`return ${kept}.length;`); break;
    case 'ids': lines.push(`return [...${kept}].sort((a, b) => a.${d.date}.localeCompare(b.${d.date}) || Number(a.id.slice(1)) - Number(b.id.slice(1))).map(x => x.id);`); break;
    case 'group_count': lines.push(`const out: Record<string, number> = {};`, `for (const x of ${kept}) out[x.${d.group}] = (out[x.${d.group}] ?? 0) + 1;`, 'return out;'); break;
    case 'sum': lines.push(`return ${kept}.reduce((s, x) => s + x.${d.numeric}, 0);`); break;
    case 'classify_count': lines.push(`const labels = ${classify('scoped')};`,
      `const out: Record<string, number> = ${literal(Object.fromEntries(spec.values.map(v => [v, 0])))};`, 'for (const label of labels) out[label]++;', 'return out;'); break;
    case 'extract': lines.push(`const sorted = [...${kept}].sort((a, b) => Number(a.id.slice(1)) - Number(b.id.slice(1)));`,
      `const labels = ${classify('sorted')};`, `return sorted.map((x, j) => ({ id: x.id, ${cat}: labels[j] }));`); break;
    case 'top_group': lines.push(`const counts: Record<string, number> = {};`, `for (const x of ${kept}) counts[x.${d.group}] = (counts[x.${d.group}] ?? 0) + 1;`,
      `const best = Object.entries(counts).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0];`, 'return best ? best[0] : null;'); break;
  }
  return lines.join('\n');
}

/** Child answers for the reference: every item, for every question it may be asked (fragments: question key + id). */
function childAnswers(d, items) {
  const out = [];
  for (const x of items) for (const [attr, spec] of Object.entries(d.attrs)) {
    const keys = [spec.key, ...(d.helpers[attr] ? [d.helpers[attr].instructions.slice(0, 24)] : [])];
    for (const key of keys) out.push({ match: [key, JSON.stringify(x.id)], value: x.labels[attr] });
  }
  return out;
}

function serviceSource(d, items) {
  const plain = items.map(({ labels, ...rest }) => rest);
  const [a, , c] = crispNames(d);
  return `const ITEMS = ${literal(plain)};
/** Every ${d.noun} in the system, with its text. */
export function ${d.plural}(): ${d.type}[] { return ITEMS; }
/** The ${d.plural} with ${a} equal to value. */
export function by_${a}(value: string): ${d.type}[] { return ITEMS.filter(x => x.${a} === value); }
/** The ${d.plural} whose ${c} (YYYY-MM-DD) falls in [start, end]. */
export function ${c}_between(start: string, end: string): ${d.type}[] { return ITEMS.filter(x => x.${c} >= start && x.${c} <= end); }
`;
}

function libraryFiles(d, items) {
  const files = { [`${d.root}/${d.service}.ts`]: serviceSource(d, items), 'types.ts': d.types };
  for (const [attr, helper] of Object.entries(d.helpers))
    files[`${d.root}/${helper.name}.nl`] = nlFile({ args: { [d.noun]: d.type }, returns: helper.returns, description: helper.description,
      instructions: helper.instructions });
  const near = d.nearMiss;
  files[`${d.root}/${near.name}.nl`] = nlFile({ args: near.args, returns: near.returns, description: near.description, instructions: near.instructions });
  return files;
}

/** Collection processing over one domain: paired worlds (a, b) with one runtime-composed request. */
export function semanticCollection(seed, index, domainName) {
  const rng = new Random(seed, `workbench:${domainName ?? ''}:${index}`);
  const d = domainName ? DOMAINS[domainName] : DOMAINS[pick(rng, Object.keys(DOMAINS))];
  for (let attempt = 0; attempt < 40; attempt++) {
    const n = rng.int(7, 12);
    const people = rng.sample(d.customers, n);
    const base = Array.from({ length: n }, (_, i) => crispItem(rng, d, i, people));
    const req = request(rng, d);
    const worldA = base.map(x => ({ ...x, labels: labelsFor(rng, d) }));
    // World b: the same items with two to three items' hidden labels redrawn.
    const changed = new Set(rng.sample(base.map(x => x.id), rng.int(2, 3)));
    const worldB = worldA.map(x => changed.has(x.id) ? { ...x, labels: labelsFor(rng, d) } : x);
    const ea = evaluate(d, req, worldA), eb = evaluate(d, req, worldB);
    if (JSON.stringify(ea) === JSON.stringify(eb)) continue;
    // Trivial requests (nothing kept in either world, or every scoped item kept) teach little.
    if (req.output !== 'classify_count' && [worldA, worldB].every(w => evaluate(d, { ...req, output: 'count' }, w) === 0)) continue;
    const { text, returns } = describe(d, req);
    const shape = `${d.name}${index}`;
    const inline = req.preds.length || !d.helpers[catAttr(d)] ? 'required' : 'optional';
    return [['a', worldA], ['b', worldB]].map(([variant, world]) => {
      const items = world.map(x => ({ ...x, text: render(new Random(seed, `text:${shape}:${x.id}:${JSON.stringify(x.labels)}`), d, x, x.labels) }));
      const expected = evaluate(d, req, items);
      return curriculumCase({ family: `workbench_${d.name}`, shape, variant, pairGroup: `workbench:${shape}`,
        slice: 'inline_placement', domain: 'other', mode: 'single_call', inline,
        evidence: { world: items.map(x => `${x.id}: ${JSON.stringify(x.labels)}`), retrieved: [], background: [] },
        minimumSequence: ['narrow the collection with the crisp helpers or fields', 'judge each remaining item in its own nl call for every semantic criterion',
          'aggregate the judgments exactly in code'],
        reference: { root: [evalCall(program(d, req)), returnCall(expected)], children: childAnswers(d, items) },
        root: { name: d.root, args: {}, returns,
          instructions: `${text} Use ${d.service}.${d.plural}() for the data. Judge what each ${d.noun} says from its text, one ${d.noun} at a time; compute counts, filters and totals exactly.` },
        files: libraryFiles(d, items), inputs: {}, expected });
    });
  }
  throw new Error(`no discriminating workbench request for ${seed}:${index}`);
}

export const workbenchTickets = (seed, index) => semanticCollection(seed, index, 'tickets');
export const workbenchReviews = (seed, index) => semanticCollection(seed, index, 'reviews');
export const workbenchApplicants = (seed, index) => semanticCollection(seed, index, 'applicants');
