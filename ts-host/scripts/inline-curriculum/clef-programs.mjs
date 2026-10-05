// Clef-distilled programs: root programs in the shapes decision models are used for (Jev / Clef use cases: routing with
// policy and customer context, retrieve-then-judge relevance, composite scoring, moderation against rules), where each
// inline nl lambda takes several values from scope. Children are answered by Cloudflare's Clef-flash decision model:
// a lambda's arguments are its state, its typed result is the question (noul -> boolean, choice -> union member,
// score -> level). Low-confidence answers are not used (confidence gating); the case's expected result is computed by
// code from the kept answers. Three stages:
//   node clef-programs.mjs skeletons SEED COUNT OUT.skeletons.jsonl     cases + Clef requests (no model calls)
//   node clef-label.mjs OUT.skeletons.jsonl OUT.answers.jsonl            Clef-flash answers (daily token budget)
//   node clef-programs.mjs assemble OUT.skeletons.jsonl OUT.answers.jsonl OUT.ir.jsonl
// The assembled cases verify like any curriculum shard and replay into demonstrations (replay-demonstrations.mjs).
import { readFileSync, writeFileSync } from 'node:fs';
import { rowsOf } from './labeled.mjs';
import { Random, curriculumCase, evalCall, literal, returnCall } from './lib.mjs';

const SCIFACT = new URL('../../../data/neuralese/corpora/source-episode-pools-20261005/self-improvement-expansion-20261004/scifact-source-v6/candidates.jsonl', import.meta.url);
const COMPANIES = ['Arbor Labs', 'Brightline', 'Cobalt Foods', 'Delta Print', 'Evergreen Clinic', 'Fjord Travel', 'Granite Legal', 'Harbor Media',
  'Ivy Schools', 'Juniper Bank', 'Kestrel Air', 'Lumen Studio', 'Maple Freight', 'Nimbus Retail', 'Orchid Health', 'Pioneer Steel'];

// Templates: build(rng) -> { name, instructions, returns, types, files, items, lambda: { key, text, signature, args(item) },
// questions (Clef questions for one item), decode(answers) -> child value | null, aggregate(values, items) -> expected,
// code (reference program) }.
const TEMPLATES = {
  // Routing with customer and policy context (support triage + escalation).
  support_routing(rng) {
    const policy = rng.pick([
      'Enterprise customers with an outage or data problem are escalated to the on-call engineer. Billing disputes go to finance. Everything else goes to the support queue.',
      'Anything that blocks a paying customer from working is escalated. Questions about invoices go to finance. Feature ideas go to product. The rest goes to support.',
      'Security or data-loss reports are always escalated, whatever the plan. Refund requests go to finance. Free-plan questions go to the community forum. The rest goes to support.']);
    const teams = { escalate: 'escalate to the on-call engineer', finance: 'the finance team', product: 'the product team', forum: 'the community forum', support: 'the support queue' };
    const items = rng.sample(rowsOf('banking77'), rng.int(5, 7)).map((row, i) => ({ id: `T${i + 1}`, text: row.text,
      customer: { company: rng.pick(COMPANIES), plan: rng.pick(['free', 'pro', 'enterprise']), seats: rng.int(1, 400), open_tickets: rng.int(0, 6) } }));
    return { name: 'route_tickets', returns: 'Record<string, Destination>', policy,
      instructions: 'Route every open ticket from desk.tickets() under the routing policy (desk.policy()), taking the customer record of each ticket into account. Return a record from ticket id to its destination.',
      types: `export type Ticket = { id: string, text: string, customer: { company: string, plan: string, seats: number, open_tickets: number } };\nexport type Destination = ${Object.keys(teams).map(t => JSON.stringify(t)).join(' | ')};\n`,
      service: items => `const TICKETS = ${literal(items)};\nconst POLICY = ${JSON.stringify(policy)};\n/** The open tickets, each with its customer record. */\nexport function tickets(): Ticket[] { return TICKETS; }\n/** The routing policy in force. */\nexport function policy(): string { return POLICY; }\n`,
      items, lambda: { key: 'Route ticket under policy', text: 'Route ticket under policy, considering its customer record. Destinations: escalate, finance, product, forum or support.',
        signature: '(ticket: Ticket, policy: string) => Promise<Destination>', call: 'route(ticket, policy)' },
      state: item => ({ ticket: item.text, customer: item.customer, policy }),
      questions: { destination: { type: 'choice', instructions: 'Under the policy, where should this ticket go, considering the customer record?', criteria: teams } },
      decode: a => a.destination.confidence >= 0.6 ? a.destination.choice : null,
      aggregate: (values, items) => Object.fromEntries(items.map((item, i) => [item.id, values[i]])),
      code: `const policy = desk.policy();\nconst all = desk.tickets();\nconst destinations = await Promise.all(all.map(ticket => route(ticket, policy)));\nreturn Object.fromEntries(all.map((ticket, i) => [ticket.id, destinations[i]]));` };
  },
  // Retrieve, then judge: narrow passages in code (by the cited document), judge relevance per sentence.
  evidence_sentences(rng) {
    const claims = readFileSync(SCIFACT, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l))
      .filter(r => r.role === 'train' && r.task.documents[0].abstract_sentences.length >= 5 && r.task.documents[0].abstract_sentences.length <= 10);
    const c = rng.pick(claims), doc = c.task.documents[0];
    const items = doc.abstract_sentences.map(s => ({ id: `S${s.sentence_id}`, text: s.text }));
    return { name: 'evidence_sentences', returns: 'string[]', claim: c.task.claim,
      instructions: 'Find the sentences of the abstract (papers.sentences()) that bear on the claim (papers.claim()): they support or contradict it directly. Return their ids in abstract order.',
      types: 'export type Sentence = { id: string, text: string };\n',
      service: items => `const CLAIM = ${JSON.stringify(c.task.claim)};\nconst TITLE = ${JSON.stringify(doc.title)};\nconst SENTENCES = ${literal(items)};\n/** The claim under review. */\nexport function claim(): string { return CLAIM; }\n/** The title of the cited paper. */\nexport function title(): string { return TITLE; }\n/** The cited abstract, sentence by sentence. */\nexport function sentences(): Sentence[] { return SENTENCES; }\n`,
      items, lambda: { key: 'Does sentence bear on claim', text: 'Does sentence bear on claim, supporting or contradicting it directly?',
        signature: '(sentence: Sentence, claim: string) => Promise<boolean>', call: 'bears(sentence, claim)' },
      state: item => ({ claim: c.task.claim, paper: doc.title, sentence: item.text }),
      questions: { relevant: { type: 'noul', instructions: 'Does this sentence directly support or contradict the claim?' } },
      decode: a => a.relevant.noul >= 0.85 ? true : a.relevant.noul <= 0.15 ? false : null,
      aggregate: (values, items) => items.filter((_, i) => values[i]).map(item => item.id),
      code: `const claim = papers.claim();\nconst all = papers.sentences();\nconst hits = await Promise.all(all.map(sentence => bears(sentence, claim)));\nreturn all.filter((_, i) => hits[i]).map(sentence => sentence.id);` };
  },
  // Composite scoring: independent dimensions per lead, weights in code.
  lead_scoring(rng) {
    const product = rng.pick(['a payroll and HR platform for growing teams', 'cloud security monitoring', 'logistics route-planning software',
      'a data warehouse for analytics teams', 'customer-support automation']);
    const items = rng.sample(rowsOf('ag_news').filter(r => r.label === 'Business' || r.label === 'Sci/Tech'), rng.int(4, 6))
      .map((row, i) => ({ id: `L${i + 1}`, company: COMPANIES[(i * 5 + rng.int(0, 15)) % COMPANIES.length], news: row.text }));
    const weights = { need: rng.int(2, 3), timing: rng.int(1, 2) };
    return { name: 'rank_leads', returns: 'string[]', product,
      instructions: `We sell ${product}. For each lead in crm.leads(), score from the company news how strongly it signals a need for our product (0-3) and how urgent the timing looks (0-3). Rank leads by ${weights.need} x need + ${weights.timing} x timing, ties by id, and return the ids of the top two.`,
      types: 'export type Lead = { id: string, company: string, news: string };\nexport type Level = 0 | 1 | 2 | 3;\n',
      service: items => `const LEADS = ${literal(items)};\n/** The leads, each with a recent news item about the company. */\nexport function leads(): Lead[] { return LEADS; }\n`,
      items, lambda: { key: 'Score lead for product', text: 'Score lead for product: need (0-3) is how strongly its news signals a need for product; timing (0-3) is how urgent that need looks.',
        signature: '(lead: Lead, product: string) => Promise<{ need: Level, timing: Level }>', call: `score(lead, ${JSON.stringify(product)})` },
      state: item => ({ product, company: item.company, news: item.news }),
      questions: { need: { type: 'score', instructions: 'How strongly does this news signal that the company needs the product?', criteria: ['No signal', 'Weak', 'Clear', 'Strong'] },
        timing: { type: 'score', instructions: 'How urgent does that need look right now?', criteria: ['Not urgent', 'Someday', 'Soon', 'Now'] } },
      decode: a => a.need.confidence >= 0.3 && a.timing.confidence >= 0.3 ? { need: Math.round(a.need.score), timing: Math.round(a.timing.score) } : null,
      aggregate: (values, items) => items.map((item, i) => ({ id: item.id, s: weights.need * values[i].need + weights.timing * values[i].timing }))
        .sort((a, b) => b.s - a.s || Number(a.id.slice(1)) - Number(b.id.slice(1))).slice(0, 2).map(x => x.id),
      code: `const all = crm.leads();\nconst scores = await Promise.all(all.map(lead => score(lead, ${JSON.stringify(product)})));\nreturn all.map((lead, i) => ({ id: lead.id, s: ${weights.need} * scores[i].need + ${weights.timing} * scores[i].timing }))\n  .sort((a, b) => b.s - a.s || Number(a.id.slice(1)) - Number(b.id.slice(1))).slice(0, 2).map(x => x.id);` };
  },
  // Moderation against community rules and the thread's topic.
  moderation(rng) {
    const thread = rng.pick(['a film club discussing this month\'s screenings', 'a neighbourhood group chat about local events', 'a fan forum for a football club']);
    const rules = rng.pick([
      'No advertising or links to sales. Personal attacks are removed; harsh opinions about films are fine.',
      'Stay on topic. Promotions and prize messages are removed. Be civil: insults get a warning the first time.',
      'Spam is removed. Off-topic posts get a gentle warning. Criticism is welcome when it is about the subject, not the person.']);
    const items = rng.sample([...rowsOf('sms_spam'), ...rowsOf('sst2')], rng.int(5, 7)).map((row, i) => ({ id: `P${i + 1}`, text: row.text }));
    const actions = { allow: 'leave the post up', warn: 'leave it up and warn the author', remove: 'remove the post' };
    return { name: 'moderate_thread', returns: 'Record<string, Action>', rules, thread,
      instructions: 'Moderate every new post in forum.posts() under the community rules (forum.rules()), keeping the thread\'s topic (forum.topic()) in mind. Return a record from post id to the action.',
      types: 'export type Post = { id: string, text: string };\nexport type Action = "allow" | "warn" | "remove";\n',
      service: items => `const POSTS = ${literal(items)};\nconst RULES = ${JSON.stringify(rules)};\nconst TOPIC = ${JSON.stringify(thread)};\n/** New posts awaiting moderation. */\nexport function posts(): Post[] { return POSTS; }\n/** The community rules. */\nexport function rules(): string { return RULES; }\n/** What the thread is about. */\nexport function topic(): string { return TOPIC; }\n`,
      items, lambda: { key: 'Moderate post under rules', text: 'Moderate post under rules, given that the thread is about topic: allow, warn or remove.',
        signature: '(post: Post, rules: string, topic: string) => Promise<Action>', call: 'moderate(post, rules, topic)' },
      state: item => ({ thread, rules, post: item.text }),
      questions: { action: { type: 'choice', instructions: 'Under the rules and the thread topic, what should a moderator do with this post?', criteria: actions } },
      decode: a => a.action.confidence >= 0.6 ? a.action.choice : null,
      aggregate: (values, items) => Object.fromEntries(items.map((item, i) => [item.id, values[i]])),
      code: `const rules = forum.rules(), topic = forum.topic();\nconst all = forum.posts();\nconst actions = await Promise.all(all.map(post => moderate(post, rules, topic)));\nreturn Object.fromEntries(all.map((post, i) => [post.id, actions[i]]));` };
  },
};
const SERVICES = { route_tickets: 'desk', evidence_sentences: 'papers', rank_leads: 'crm', moderate_thread: 'forum' };

function skeletons(seed, count, out) {
  const lines = [];
  for (let index = 0; index < count; index++) {
    const rng = new Random(Number(seed), `clef:${index}`);
    const name = rng.pick(Object.keys(TEMPLATES));
    const t = TEMPLATES[name](rng);
    lines.push(JSON.stringify({ seed: Number(seed), index, template: name,
      requests: t.items.map(item => ({ item: item.id, state: t.state(item), questions: t.questions })) }));
  }
  writeFileSync(out, lines.join('\n') + '\n');
  console.log(`${count} skeletons -> ${out}`);
}

function assemble(skeletonPath, answersPath, out) {
  const answers = new Map(readFileSync(answersPath, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l)).map(a => [`${a.seed}:${a.index}:${a.item}`, a.answers]));
  const records = [], dropped = {};
  for (const line of readFileSync(skeletonPath, 'utf8').split('\n').filter(Boolean)) {
    const sk = JSON.parse(line);
    const rng = new Random(sk.seed, `clef:${sk.index}`);
    rng.pick(Object.keys(TEMPLATES));  // replay the draws of skeletons()
    const t = TEMPLATES[sk.template](rng);
    // Confidence gating per item: an item Clef is unsure about leaves the case (its data too); a case needs three.
    const decoded = t.items.map(item => { const a = answers.get(`${sk.seed}:${sk.index}:${item.id}`); return a ? t.decode(a) : null; });
    const keep = t.items.map((_, i) => decoded[i] !== null);
    t.items = t.items.filter((_, i) => keep[i]);
    const values = decoded.filter(v => v !== null);
    dropped.items = (dropped.items ?? 0) + keep.filter(k => !k).length;
    if (t.items.length < 3) { dropped[sk.template] = (dropped[sk.template] ?? 0) + 1; continue; }
    const expected = t.aggregate(values, t.items);
    const service = SERVICES[t.name];
    const [fn] = t.lambda.call.split('(');
    const record = curriculumCase({ family: `clef_${sk.template}`, shape: `clef${sk.index}`, variant: 'v0', splitGroup: `clef:${sk.seed}:${sk.index}`,
      slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'required',
      evidence: { world: t.items.map((item, i) => `${item.id}: ${JSON.stringify(values[i])}`), retrieved: [], background: [] },
      minimumSequence: ['read the items and the context they are judged in', 'judge each item in its own nl call that takes the item and its context',
        'combine the judgments in code'],
      reference: { root: [evalCall(`const ${fn} = nl<${t.lambda.signature}>\`${t.lambda.text}\`;\n${t.code}`), returnCall(expected)],
        children: t.items.map((item, i) => ({ match: [t.lambda.key, JSON.stringify(item.id)], value: values[i] })) },
      root: { name: t.name, args: {}, returns: t.returns, instructions: t.instructions },
      files: { [`${t.name}/${service}.ts`]: t.service(t.items), 'types.ts': t.types }, inputs: {}, expected });
    record.gold_sources = ['clef-flash-decisions'];
    record.curriculum.child_labels = { model: '@cf/cloudflare/clef-flash', gating: 'confidence' };
    records.push(record);
  }
  writeFileSync(out, records.map(r => JSON.stringify(r)).join('\n') + '\n');
  console.log(JSON.stringify({ assembled: records.length, dropped }));
}

const [mode, ...args] = process.argv.slice(2);
if (mode === 'skeletons') skeletons(...args);
else if (mode === 'assemble') assemble(...args);
else if (mode) throw new Error('usage: clef-programs.mjs skeletons SEED COUNT OUT | assemble SKELETONS ANSWERS OUT');
