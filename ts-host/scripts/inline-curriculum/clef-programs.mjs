// Clef-distilled programs: root programs in the shapes decision models are used for (Jev / Clef use cases: routing with
// policy and customer context, retrieve-then-judge relevance, composite scoring, moderation against rules), where each
// inline nl lambda takes several values from scope. Children are answered by Cloudflare's Clef-flash decision model:
// a lambda's arguments are its state, its typed result is the question (noul -> boolean, choice -> union member,
// score -> level). Low-confidence answers are not used (confidence gating); the case's expected result is computed by
// code from the kept answers. Three stages:
//   node clef-programs.mjs skeletons SEED COUNT OUT.skeletons.jsonl [POOL]   cases + Clef requests (no model calls);
//        POOL is a comma list of templates (default: the original four; `new` = NEW_POOLS)
//   node clef-label.mjs OUT.skeletons.jsonl OUT.answers.jsonl            Clef-flash answers (daily token budget)
//   node clef-programs.mjs assemble OUT.skeletons.jsonl OUT.answers.jsonl OUT.ir.jsonl
// The assembled cases verify like any curriculum shard and replay into demonstrations (replay-demonstrations.mjs).
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { rowsOf } from './labeled.mjs';
import { coeditRows } from './folder-data.mjs';
import { SOURCES, cachePath } from './acquire.mjs';
import { sourceRecordId, sourceRecordSplit } from './source-split.mjs';
import { Random, curriculumCase, evalCall, literal, returnCall } from './lib.mjs';

const CUAD_CACHE = process.env.NATLANG_DATASETS ?? fileURLToPath(new URL('../../../vendor/datasets', import.meta.url));

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
  // Skill routing for an assistant (Jev skill suggestion): each request goes to one of the installed skills or none.
  skill_routing(rng) {
    const intents = [...new Set(rowsOf('clinc_oos').map(row => row.label))].sort();
    const skills = rng.sample(intents, rng.int(4, 6));
    const describe = label => label.replace(/_/g, ' ');
    const pool = rowsOf('clinc_oos');
    const inSkills = rng.sample(pool.filter(row => skills.includes(row.label)), rng.int(3, 5));
    const outside = rng.sample(pool.filter(row => !skills.includes(row.label)), rng.int(1, 2));
    const items = rng.shuffle([...inSkills, ...outside]).map((row, i) => ({ id: `R${i + 1}`, text: row.text }));
    const criteria = { ...Object.fromEntries(skills.map(skill => [skill, `the ${describe(skill)} skill`])), none: 'none of the installed skills fits' };
    const catalog = Object.fromEntries(skills.map(skill => [skill, `Handles requests about ${describe(skill)}.`]));
    return { name: 'route_requests', returns: 'Record<string, Skill>',
      instructions: 'Route every pending user request in assistant.requests() to the installed skill that should handle it (assistant.skills() describes each), or to "none" when no installed skill fits. Return a record from request id to skill.',
      types: `export type Request = { id: string, text: string };\nexport type Skill = ${[...skills, 'none'].map(s => JSON.stringify(s)).join(' | ')};\n`,
      service: items => `const REQUESTS = ${literal(items)};\nconst SKILLS = ${literal(catalog)};\n/** Pending user requests. */\nexport function requests(): Request[] { return REQUESTS; }\n/** The installed skills, each with what it handles. */\nexport function skills(): Record<string, string> { return SKILLS; }\n`,
      items, lambda: { key: 'Pick the installed skill for request', text: 'Pick the installed skill for request from skills, or "none" when none of them fits.',
        signature: '(request: Request, skills: Record<string, string>) => Promise<Skill>', call: 'pick(request, skills)' },
      state: item => ({ request: item.text, skills: catalog }),
      questions: { skill: { type: 'choice', instructions: 'Which installed skill should handle this request? Choose none when no skill fits.', criteria } },
      decode: a => a.skill.confidence >= 0.6 ? a.skill.choice : null,
      aggregate: (values, items) => Object.fromEntries(items.map((item, i) => [item.id, values[i]])),
      code: `const skills = assistant.skills();\nconst all = assistant.requests();\nconst chosen = await Promise.all(all.map(request => pick(request, skills)));\nreturn Object.fromEntries(all.map((request, i) => [request.id, chosen[i]]));` };
  },
  // Wellbeing triage in a peer-support community: follow-up by policy, from the post and the member's history.
  wellbeing_triage(rng) {
    const community = rng.pick(['a peer-support forum for students during exam season', 'an online community for new parents',
      'a workplace wellbeing channel', 'a running club\'s group chat']);
    const policy = rng.pick([
      'Moderators reach out privately to anyone who sounds distressed or alone. Strong anger at another member gets a check-in. Everything else needs no action.',
      'Reach out when a post shows fear or sadness that has lasted, judging by the member\'s recent history. Monitor a single low moment. Happy or neutral posts need no action.',
      'Members who sound overwhelmed get a private message from a moderator. Members who sound upset for the first time are monitored. The rest need no action.']);
    const items = rng.sample(rowsOf('emotion'), rng.int(5, 7)).map((row, i) => ({ id: `P${i + 1}`, text: row.text,
      member: { posts_this_month: rng.int(1, 30), previous_check_ins: rng.int(0, 3) } }));
    const actions = { reach_out: 'a moderator reaches out privately', monitor: 'keep an eye on the member', none: 'no action needed' };
    return { name: 'triage_posts', returns: 'Record<string, FollowUp>',
      instructions: 'For every new post in community.posts(), decide the moderators\' follow-up under the community\'s wellbeing policy (community.policy()), taking the member\'s record into account and keeping in mind what the community is (community.about()). Return a record from post id to follow-up.',
      types: 'export type Post = { id: string, text: string, member: { posts_this_month: number, previous_check_ins: number } };\nexport type FollowUp = "reach_out" | "monitor" | "none";\n',
      service: items => `const POSTS = ${literal(items)};\nconst POLICY = ${JSON.stringify(policy)};\nconst ABOUT = ${JSON.stringify(community)};\n/** New posts, each with the author's member record. */\nexport function posts(): Post[] { return POSTS; }\n/** The wellbeing policy. */\nexport function policy(): string { return POLICY; }\n/** What the community is. */\nexport function about(): string { return ABOUT; }\n`,
      items, lambda: { key: 'Decide follow-up for post', text: 'Decide follow-up for post under policy, in a community that is about: reach_out, monitor or none.',
        signature: '(post: Post, policy: string, about: string) => Promise<FollowUp>', call: 'followUp(post, policy, about)' },
      state: item => ({ community, policy, post: item.text, member: item.member }),
      questions: { action: { type: 'choice', instructions: 'Under the wellbeing policy, what follow-up does this post need from the moderators?', criteria: actions } },
      decode: a => a.action.confidence >= 0.6 ? a.action.choice : null,
      aggregate: (values, items) => Object.fromEntries(items.map((item, i) => [item.id, values[i]])),
      code: `const policy = community.policy(), about = community.about();\nconst all = community.posts();\nconst actions = await Promise.all(all.map(post => followUp(post, policy, about)));\nreturn Object.fromEntries(all.map((post, i) => [post.id, actions[i]]));` };
  },
  // Edit review: an editor's proposed revisions, each checked against the instruction it was given.
  edit_review(rng) {
    const tasks = { gec: 'Fix the grammar and spelling.', simplification: 'Make the text simpler to read.', paraphrase: 'Paraphrase the text.',
      coherence: 'Make the text more coherent.', neutralize: 'Remove biased or non-neutral wording.' };
    const pairs = Object.keys(tasks).flatMap(task => rng.sample(coeditRows(task), 2).map(row => ({ task, row })));
    // A proposal is either the reference revision or another draft's revision (a mismatched proposal), so both answers occur.
    const items = rng.sample(pairs, rng.int(5, 6)).map(({ task, row }, i) => {
      const mismatched = rng.next() < 0.3;
      const proposal = mismatched ? rng.pick(pairs.filter(p => p.row.id !== row.id)).row.target : row.target;
      return { id: `E${i + 1}`, instruction: tasks[task], original: row.text, proposal };
    });
    return { name: 'review_edits', returns: 'string[]',
      instructions: 'Review the proposed revisions in editor.proposals(): a proposal is accepted when it carries out its instruction on the original text and keeps its meaning. Return the ids of the accepted proposals in order.',
      types: 'export type Proposal = { id: string, instruction: string, original: string, proposal: string };\n',
      service: items => `const PROPOSALS = ${literal(items)};\n/** Proposed revisions, each with its instruction and the original text. */\nexport function proposals(): Proposal[] { return PROPOSALS; }\n`,
      items, lambda: { key: 'Does proposal carry out its instruction', text: 'Does proposal carry out its instruction on the original text and keep its meaning?',
        signature: '(proposal: Proposal) => Promise<boolean>', call: 'accept(proposal)' },
      state: item => ({ instruction: item.instruction, original: item.original, proposal: item.proposal }),
      questions: { accept: { type: 'noul', instructions: 'Does the proposal carry out the instruction on the original text while keeping its meaning?' } },
      decode: a => a.accept.noul >= 0.85 ? true : a.accept.noul <= 0.15 ? false : null,
      aggregate: (values, items) => items.filter((_, i) => values[i]).map(item => item.id),
      code: `const all = editor.proposals();\nconst verdicts = await Promise.all(all.map(proposal => accept(proposal)));\nreturn all.filter((_, i) => verdicts[i]).map(proposal => proposal.id);` };
  },
  // Contract clause review from one side's position: score each clause's risk, the code flags the high ones.
  clause_risk(rng) {
    const contract = rng.pick(cuadClauseContracts().filter(c => c.clauses.length >= 4));
    const side = rng.pick(['the licensee', 'the licensor', 'the customer', 'the supplier']);
    const position = rng.pick([
      `We are ${side}. We want to keep the freedom to end the contract, to work with anyone, and to limit what we owe if something goes wrong.`,
      `We are ${side}. Our priorities are protecting our intellectual property and avoiding open-ended liability or exclusivity.`,
      `We are ${side}. We care most about flexibility: short commitments, no restrictions on competitors or hiring, and easy assignment.`]);
    const items = rng.sample(contract.clauses, Math.min(contract.clauses.length, rng.int(4, 6))).map((clause, i) => ({ id: `C${i + 1}`, heading: clause.category, text: clause.text }));
    const threshold = 2;
    return { name: 'flag_clauses', returns: 'string[]',
      instructions: `Review the clauses of the contract in legal.clauses() against our position (legal.position()). Score each clause's risk to us from 0 (none) to 3 (serious) and return the ids of the clauses scoring ${threshold} or more, in order.`,
      types: 'export type Clause = { id: string, heading: string, text: string };\nexport type Risk = 0 | 1 | 2 | 3;\n',
      service: items => `const CLAUSES = ${literal(items)};\nconst POSITION = ${JSON.stringify(position)};\n/** Clauses of the contract under review. */\nexport function clauses(): Clause[] { return CLAUSES; }\n/** Our negotiating position. */\nexport function position(): string { return POSITION; }\n`,
      items, lambda: { key: 'Score clause risk to us', text: 'Score clause risk to us given position, from 0 (none) to 3 (serious).',
        signature: '(clause: Clause, position: string) => Promise<Risk>', call: 'risk(clause, position)' },
      state: item => ({ position, heading: item.heading, clause: item.text }),
      questions: { risk: { type: 'score', instructions: 'How much risk does this clause carry for us, given our position?', criteria: ['None', 'Minor', 'Significant', 'Serious'] } },
      // The most probable level when it holds at least half the mass (a rounded expectation hides a split answer).
      decode: a => { const [level, p] = Object.entries(a.risk.probabilities ?? {}).sort((x, y) => y[1] - x[1])[0] ?? [];
        return p >= 0.5 ? Number(level) : null; },
      aggregate: (values, items) => items.filter((_, i) => values[i] >= threshold).map(item => item.id),
      code: `const position = legal.position();\nconst all = legal.clauses();\nconst risks = await Promise.all(all.map(clause => risk(clause, position)));\nreturn all.filter((_, i) => risks[i] >= ${threshold}).map(clause => clause.id);` };
  },
};
const SERVICES = { route_tickets: 'desk', evidence_sentences: 'papers', rank_leads: 'crm', moderate_thread: 'forum',
  route_requests: 'assistant', triage_posts: 'community', review_edits: 'editor', flag_clauses: 'legal' };
// Skeletons written before the pool field drew from these four templates; their draws replay with this list.
const ORIGINAL_POOL = ['support_routing', 'evidence_sentences', 'lead_scoring', 'moderation'];
// The new source pools (2026-10-11): CLINC150 skills, emotion posts, CoEdIT revisions, CUAD clauses.
export const NEW_POOLS = ['skill_routing', 'wellbeing_triage', 'edit_review', 'clause_risk'];

// CUAD contracts (train split by the shared content partition, same ids as folder_extract) with their annotated
// clause spans under risk-bearing headings.
const CLAUSE_HEADINGS = ['Anti-Assignment', 'Cap On Liability', 'Uncapped Liability', 'Termination For Convenience', 'Exclusivity',
  'Non-Compete', 'Ip Ownership Assignment', 'Change Of Control', 'Liquidated Damages', 'Audit Rights', 'Minimum Commitment',
  'Most Favored Nation', 'No-Solicit Of Employees', 'Covenant Not To Sue', 'Renewal Term', 'Rofr/Rofo/Rofn', 'Volume Restriction',
  'Joint Ip Ownership', 'Non-Disparagement', 'Insurance', 'Warranty Duration', 'Revenue/Profit Sharing', 'Price Restrictions'];
let clauseCache;
function cuadClauseContracts() {
  if (clauseCache) return clauseCache;
  const source = SOURCES.cuad;
  const parsed = JSON.parse(readFileSync(cachePath(CUAD_CACHE, 'cuad', source.revision, source.files[0].path), 'utf8'));
  clauseCache = [];
  for (const item of parsed.data ?? []) {
    const title = String(item.title ?? ''), paragraphs = item.paragraphs ?? [];
    const content = [...new Set(paragraphs.map(part => String(part.context ?? '')).filter(Boolean))].join('\n\n');
    if (!title || !content || sourceRecordSplit(sourceRecordId('cuad', title, content)) !== 'train') continue;
    const clauses = [];
    for (const qa of paragraphs.flatMap(part => part.qas ?? [])) {
      const category = String(qa.id ?? '').split('__').pop();
      if (!CLAUSE_HEADINGS.includes(category)) continue;
      const text = (qa.answers ?? []).map(a => String(a.text ?? '').replace(/\s+/g, ' ').trim()).find(t => t.length >= 40 && t.length <= 700);
      if (text) clauses.push({ category, text });
    }
    clauseCache.push({ title, clauses });
  }
  return clauseCache;
}

function skeletons(seed, count, out, pool) {
  const lines = [];
  const names = pool ? pool.split(',') : ORIGINAL_POOL;
  for (const name of names) if (!TEMPLATES[name]) throw new Error(`unknown template ${name}`);
  for (let index = 0; index < count; index++) {
    const rng = new Random(Number(seed), `clef:${index}`);
    const name = rng.pick(names);
    const t = TEMPLATES[name](rng);
    lines.push(JSON.stringify({ seed: Number(seed), index, template: name, ...(pool ? { pool: names } : {}),
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
    rng.pick(sk.pool ?? ORIGINAL_POOL);  // replay the draws of skeletons()
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
    // Pool-era skeletons put the seed in the shape: case ids of different seeds must not collide.
    const record = curriculumCase({ family: `clef_${sk.template}`, shape: sk.pool ? `clef-s${sk.seed}-${sk.index}` : `clef${sk.index}`, variant: 'v0', splitGroup: `clef:${sk.seed}:${sk.index}`,
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
if (mode === 'skeletons') skeletons(args[0], args[1], args[2], args[3] === 'new' ? NEW_POOLS.join(',') : args[3]);
else if (mode === 'assemble') assemble(...args);
else if (mode) throw new Error('usage: clef-programs.mjs skeletons SEED COUNT OUT [POOL] | assemble SKELETONS ANSWERS OUT');
