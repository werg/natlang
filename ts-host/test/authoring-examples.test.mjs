// The programs the authoring skill and README teach from (skills/natlang-authoring/references/composition.md,
// README.md): they build, they run with a scripted model, and every code block the documents quote is the code.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { CallStore, buildProject, createNatlangRuntime, Folder, untrusted } from '../dist/index.js';
import { scriptedModel } from './support/natlang.mjs';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const runtimeModule = { url: new URL('../dist/index.js', import.meta.url).href, path: fileURLToPath(new URL('../dist/index.js', import.meta.url)),
  types: fileURLToPath(new URL('../dist/index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };

async function build(example) {
  const outDir = mkdtempSync(join(tmpdir(), `natlang-${example}-`));
  const built = buildProject({ project: join(repo, 'examples', example), outDir, runtimeModule, writeDeclarations: false });
  assert.equal(built.ok, true, JSON.stringify(built.diagnostics));
  return async module => import(pathToFileURL(join(outDir, module)).href);
}

/** Everything a request shows the model: message text and the code of pre-filled tool calls (the scope eval). */
const text = request => request.messages.flatMap(message => [typeof message.content === 'string' ? message.content : JSON.stringify(message.content),
  ...(message.tool_calls ?? []).map(call => call.function?.arguments ?? '')]).join('\n');
const scores = probabilities => ({ log_probs: probabilities.map(p => Math.log(p)) });

/** A scripted local model: eval stages answer from `respond`, decisions are scored by `score(options, prompt)`. */
function localModel(respond, score) {
  const model = scriptedModel(respond);
  const decisions = [];
  const driver = async request => request.tools.length ? model.driver(request) : { text: 'true', calls: [] };
  driver.decide = async request => {
    const prompt = text(request);
    decisions.push({ options: request.options, prompt });
    // The refinement judge asks true/false about a value; the scripted values all hold.
    if (request.options.length === 2 && request.options[0] === 'true' && /Property: the value is/.test(prompt)) return scores([0.97, 0.03]);
    return scores(score(request.options, prompt));
  };
  return { driver, decisions, openings: model.openings };
}

const policies = {
  outdoor: readFileSync(join(repo, 'examples/refunds/policies/outdoor.md'), 'utf8'),
  books: readFileSync(join(repo, 'examples/refunds/policies/books.md'), 'utf8'),
};
const requests = JSON.parse(readFileSync(join(repo, 'examples/refunds/requests.json'), 'utf8'));

function refundsModel() {
  return localModel(opening => {
    if (opening.includes('Write the reply to request')) return 'return "We have settled your request. Thank you for writing to us."';
    if (opening.includes('Revise reply')) return 'return reply';
    if (opening.includes('Summarize outcome')) return 'return "The request was settled under the shop policy."';
    if (opening.includes('answers every question')) return 'return true';
    return null;
  }, (options, prompt) => {
    // eligible.nl (boolean), the stopping check (boolean) and verdict.nl (three verdicts).
    if (options.length === 2) return [0.9, 0.1];
    const p = { refund: 0.9, partial: 0.05, decline: 0.05 };
    if (prompt.includes('o-457')) Object.assign(p, { refund: 0.35, partial: 0.55, decline: 0.1 });
    return options.map(option => p[JSON.parse(option)]);
  });
}

test('examples/refunds: eligibility, a calibrated verdict under each shop\'s policy, escalation of the unsure, refined replies', async () => {
  const load = await build('refunds');
  const { settle } = await load('desk.js');
  const local = refundsModel();
  const asked = [];
  const teacher = async request => { asked.push(text(request)); return { text: '"partial"', calls: [] }; };
  const runtime = createNatlangRuntime({ model: local.driver, models: { teacher } });
  const outcomes = await runtime.run(() => settle(requests, { policies }));
  assert.deepEqual(outcomes.map(outcome => [outcome.id, outcome.verdict, outcome.escalated]),
    [['r1', 'refund', false], ['r2', 'partial', true], ['r3', 'decline', false]]);
  assert.ok(outcomes.every(outcome => outcome.reply.length > 0));
  // r3 is outside the crisp 30-day window (40 days): it is declined without a verdict call.
  const verdicts = local.decisions.filter(decision => decision.options.length === 3);
  assert.equal(verdicts.length, 2, 'one scored verdict per eligible request');
  // .in(context): each verdict call reads its own shop's policy.
  assert.ok(verdicts.every(decision => decision.prompt.includes('half its price')), 'both eligible requests are outdoor orders');
  assert.ok(!verdicts.some(decision => decision.prompt.includes('e-books')), 'the books policy is not in an outdoor call');
  // Only the unsure request (r2, top probability 0.55 < 0.8) went to the stronger model, under the same policy.
  assert.equal(asked.length, 1);
  assert.match(asked[0], /o-457/);
  assert.match(asked[0], /half its price/);
});

test('examples/refunds: shadow eligibility serves the crisp rule and traces where the judgment disagrees', async () => {
  const load = await build('refunds');
  const { settle } = await load('desk.js');
  const local = refundsModel();
  const traces = [];
  const runtime = createNatlangRuntime({ model: local.driver, models: { teacher: local.driver }, trace: trace => traces.push(trace) });
  const outcomes = await runtime.run(() => settle(requests, { policies, eligibility: 'shadow' }));
  assert.equal(outcomes.find(outcome => outcome.id === 'r3').verdict, 'decline', 'shadow serves the crisp side');
  const shadow = traces.flatMap(trace => trace.events ?? []).filter(event => event.kind === 'pluggable_shadow');
  assert.equal(shadow.length, 3);
  assert.ok(shadow.every(event => event.name === 'refunds.eligible' && event.served === 'crisp'));
  // The scripted judgment says yes to every request; the crisp window says no to r3 (40 days).
  assert.deepEqual(shadow.map(event => event.agree).sort(), [false, true, true]);
});

test('examples/refunds: the ledger is one folder transaction, checked before it is accepted', async () => {
  const load = await build('refunds');
  const { record } = await load('ledger.js');
  const local = refundsModel();
  const folder = Folder.fromFiles({ 'ledger.json': '{ "refund": 0, "partial": 0, "decline": 0 }\n' }, 'write');
  const outcomes = [
    { id: 'r1', verdict: 'refund', confidence: 0.9, escalated: false, reply: 'Refunded.' },
    { id: 'r2', verdict: 'partial', confidence: 0.55, escalated: true, reply: 'Half refunded.' },
  ];
  const ledger = await createNatlangRuntime({ model: local.driver }).run(() => record(folder, outcomes));
  assert.deepEqual(ledger, { refund: 1, partial: 1, decline: 0 });
  assert.deepEqual(JSON.parse(await folder.readText('ledger.json')), ledger);
  assert.match(await folder.readText('cases/r2.md'), /^# r2: partial\n\nThe request was settled/);
});

test('examples/triage: scored labels, the unsure ones on the stronger model, a refined summary type', async () => {
  const load = await build('triage');
  const { triage } = await load('triage.js');
  const tickets = JSON.parse(readFileSync(join(repo, 'examples/triage/tickets.json'), 'utf8'));
  const local = localModel(opening => {
    if (opening.includes('going wrong across tickets')) return 'return (tickets.join(" ") + " ").repeat(8).trim()';
    if (opening.includes('noticeably shorter')) return 'return text.split(/\\s+/).slice(0, Math.ceil(text.split(/\\s+/).length / 2)).join(" ")';
    return null;
  }, (options, prompt) => {
    if (options.length === 2) return /500 error/.test(prompt) ? [0.95, 0.05] : [0.05, 0.95];
    const label = /charged twice/.test(prompt) ? 'billing' : /WATCHES/.test(prompt) ? 'spam' : 'technical';
    // The cosmetic Safari ticket is the unsure one.
    const top = /Safari/.test(prompt) ? 0.6 : 0.95;
    return options.map(option => JSON.parse(option) === label ? top : (1 - top) / 2);
  });
  const asked = [];
  const teacher = async request => { asked.push(text(request)); return { text: '"technical"', calls: [] }; };
  const report = await createNatlangRuntime({ model: local.driver, models: { teacher } }).run(() => triage(tickets, 'billing, technical, spam'));
  assert.deepEqual(report.by_label, { technical: 2, billing: 1, spam: 1 });
  assert.equal(report.urgent, 1);
  assert.match(report.summary, /500 error/);
  assert.ok(report.summary.split(/\s+/).length <= 60, report.summary);
  assert.ok(local.openings.some(opening => opening.includes('noticeably shorter')), 'iterateOn shortened the draft');
  assert.equal(asked.length, 1);
  assert.match(asked[0], /Safari/);
});

test('examples/screening: checks bound with partial and a judgment written for the role, applied to every application', async () => {
  const load = await build('screening');
  const { screen } = await load('screen.js');
  const { role, requirements, applications } = JSON.parse(readFileSync(join(repo, 'examples/screening/applications.json'), 'utf8'));
  const local = localModel(opening => {
    // fit_for.nl: the call's result is a function, so the model writes the judgment and finishes with it.
    if (opening.includes('Write the judgment that decides whether an application fits role'))
      return 'return nl`Is application from someone who has kept a live payments service running?`';
    if (opening.includes('kept a live payments service running')) return 'return !application.text.includes("todo app")';
    return null;
  }, (options, prompt) => {
    // meets.nl, scored: a1 shows both requirements, a2 neither, a3 both.
    const shows = /requirement[^]*PostgreSQL/.test(prompt) ? !/happy to learn/.test(prompt) : !/todo app/.test(prompt);
    return shows ? [0.9, 0.1] : [0.1, 0.9];
  });
  const screening = await createNatlangRuntime({ model: local.driver }).run(() => screen(applications, role, requirements));
  assert.deepEqual(screening.shortlisted, ['a1', 'a3']);
  assert.deepEqual(screening.missing, { [requirements[0]]: ['a2'], [requirements[1]]: ['a2'] });
  assert.ok(local.openings.some(opening => /This call's result is a function of type \(application: Application\) => boolean/.test(opening)),
    'fit_for.nl is told how to return a function');
  // Each requirement's check is meets with the requirement bound: the scored prompts name it.
  for (const requirement of requirements) assert.ok(local.decisions.some(decision => decision.prompt.includes(requirement)), requirement);
});

test('examples/outline: a recursive .nl digests each section and every digest is held to its postconditions', async () => {
  const load = await build('outline');
  const { outline } = await load('outline.js');
  const marked = section => ({ ...section, text: untrusted(section.text, `section ${section.title}`), sections: section.sections.map(marked) });
  const document = marked(JSON.parse(readFileSync(join(repo, 'examples/outline/document.json'), 'utf8')));
  const local = localModel(opening => opening.includes('Write the gist of section') ?
    'const parts = await Promise.all(section.sections.map(child => digest(child)));\n' +
    'return { title: section.title, gist: section.text.split(". ")[0], parts };' : null, () => [0.5, 0.5]);
  // Its own call store: judged verdicts of earlier runs would answer the judge from the machine's store.
  const calls = CallStore.open(mkdtempSync(join(tmpdir(), 'natlang-outline-calls-')));
  const { contents } = await createNatlangRuntime({ model: local.driver, calls }).run(() => outline(document));
  assert.deepEqual(contents.map(line => line.split(':')[0]), ['Incident review', '  Timeline', '    Detection', '    Recovery', '  Follow-ups']);
  assert.equal(local.openings.length, 5, 'one call per section: digest called itself on each subsection');
  // Each digest's judged postcondition saw its section beside it, with the section's text fenced as outside data.
  const judged = local.decisions.filter(decision => /Property: the value is a digest whose gist/.test(decision.prompt));
  assert.equal(judged.length, 5);
  assert.ok(judged.every(decision => /Context, the values the property refers to by name:\n<<<section\n/.test(decision.prompt)));
  assert.ok(judged.some(decision => decision.prompt.includes('```untrusted data from section Detection')));
});

/**
 * A fenced block preceded by `<!-- from: PATH -->` quotes PATH: its lines appear in that file in order, contiguously
 * (indentation and `...` elision lines aside). The documents then cannot drift from the programs they teach.
 */
function quotedBlocks(document) {
  const source = readFileSync(join(repo, document), 'utf8');
  return [...source.matchAll(/<!-- from: ([^ ]+) -->\n```[a-z]*\n([^]*?)```/g)].map(match => ({ path: match[1], body: match[2] }));
}
for (const document of ['README.md', 'skills/natlang-authoring/SKILL.md', 'skills/natlang-authoring/references/composition.md',
  'skills/natlang-authoring/references/patterns.md']) {
  test(`${document} quotes its examples verbatim`, () => {
    const blocks = quotedBlocks(document);
    if (document.endsWith('composition.md')) assert.ok(blocks.length >= 6, 'composition.md teaches from quoted programs');
    for (const { path, body } of blocks) {
      const file = readFileSync(join(repo, path), 'utf8').split('\n').map(line => line.trim());
      const chunks = body.trimEnd().split(/\n\s*(?:\/\/ )?\.\.\.\n/);
      let from = 0;
      for (const chunk of chunks) {
        const lines = chunk.split('\n').map(line => line.trim());
        const at = file.findIndex((_, index) => index >= from && lines.every((line, offset) => file[index + offset] === line));
        assert.ok(at >= 0, `${document}: this block is not in ${path} as written:\n${chunk}`);
        from = at + lines.length;
      }
    }
  });
}
