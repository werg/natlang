// Adapters for AI2 reasoning datasets: ProofWriter (open-world entailment with a counterfactual missing fact),
// EntailmentBank (premise selection checked by a verifier), αNLI (per-story abductive judgments in inline
// children), and CommaQA (questions answered by combining a table specialist and a text specialist).
import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Random, curriculumCase, evalCall, literal, nlFile, returnCall } from './lib.mjs';
import { READ_ALL, factStore } from './logic.mjs';
import { SOURCES, cachePath } from './acquire.mjs';

const CACHE = process.env.NATLANG_DATASETS ?? fileURLToPath(new URL('../../../vendor/datasets', import.meta.url));
const loaded = new Map();
const once = (key, load) => { if (!loaded.has(key)) loaded.set(key, load()); return loaded.get(key); };
const missing = source => new Error(`${SOURCES[source].name} is not in the dataset cache; run node scripts/inline-curriculum/acquire.mjs --source ${source}`);
/** The first directory under root (searching depth-first) that contains file. */
function findDir(root, file) {
  if (!existsSync(root)) return;
  if (existsSync(join(root, file))) return root;
  for (const entry of readdirSync(root, { withFileTypes: true })) if (entry.isDirectory()) {
    const found = findDir(join(root, entry.name), file);
    if (found) return found;
  }
}
const jsonl = path => readFileSync(path, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
const digest = text => createHash('sha256').update(text).digest('hex').slice(0, 10);
const VERDICTS = 'export type Evaluation = { verdict: "true" | "false" | "unknown" };\n';

// ProofWriter ----------------------------------------------------------------------------------------------

function proofwriterRows() {
  return once('proofwriter', () => {
    const root = cachePath(CACHE, 'proofwriter', SOURCES.proofwriter.revision, 'proofwriter-dataset-V2020.12.3');
    const dir = findDir(root, 'OWA');
    if (!dir) throw missing('proofwriter');
    const rows = [];
    for (const [file, split] of [['meta-train.jsonl', 'train'], ['meta-dev.jsonl', 'test'], ['meta-test.jsonl', 'test']]) {
      for (const theory of jsonl(join(dir, 'OWA', 'depth-5', file))) {
        // Deep questions with exactly one proof: removing a fact that proof uses leaves the statement unknown.
        const questions = Object.entries(theory.questions).filter(([, q]) => q.QDep >= 2 && String(q.answer) !== 'Unknown' &&
          (q.proofsWithIntermediates ?? []).length === 1);
        for (const [key, question] of questions) rows.push({ theory, key, question, split });
      }
    }
    return rows;
  });
}

/**
 * ProofWriter (open world): is a statement true, false, or unknown given the theory? The counterpart removes a
 * fact the statement's only proof uses, which leaves it unknown.
 */
export function proofwriterQuestion(seed, index) {
  const rows = proofwriterRows();
  const { theory, key, question, split } = rows[(index * 7919) % rows.length];
  const rng = new Random(seed, `proofwriter:${theory.id}:${key}`);
  const used = [...new Set(question.proofs.match(/triple\d+/g) ?? [])].filter(id => theory.triples[id]);
  if (!used.length) throw new Error(`ProofWriter ${theory.id} ${key}: the proof uses no fact`);
  const removed = rng.pick(used);
  const sentences = removedId => [...Object.entries(theory.triples).filter(([id]) => id !== removedId).map(([, t]) => t.text),
    ...Object.values(theory.rules).map(r => r.text)];
  const verdict = question.answer === true || question.answer === 'True' ? 'true' : 'false';
  const shape = `${theory.id}_${key}`;
  return [['proved', null, verdict], ['fact_removed', removed, 'unknown']].map(([variant, removedId, value]) => {
    const facts = rng.shuffle(sentences(removedId)).map((text, i) => ({ id: `S${i + 1}`, text }));
    const expected = { verdict: value };
    return curriculumCase({ family: 'proofwriter_question', shape, variant, pairGroup: `proofwriter:${shape}`,
      splitGroup: `proofwriter:${theory.id}`, split, slice: 'observation_followup', domain: 'logic', mode: 'single_call',
      worldSemantics: 'open_world', inline: 'avoid',
      evidence: { world: removedId ? [`removed: ${theory.triples[removedId].text}`] : [], retrieved: [],
        background: [`source: ProofWriter ${SOURCES.proofwriter.revision} OWA depth-5 ${theory.id} ${key}`, `proof: ${question.proofs}`, `depth ${question.QDep}`] },
      assumptions: ['Open world: a statement the theory neither proves nor refutes is unknown; "not" must be proved like anything else.'],
      minimumSequence: ['read the theory', 'chain rules from the facts toward the statement or its negation'],
      reference: { root: [evalCall(READ_ALL), returnCall(expected)] },
      root: { name: 'evaluate_statement', args: { statement: 'string' }, returns: 'Evaluation',
        instructions: 'Decide whether statement is true, false, or unknown given the theory in the store (facts), using only that theory. Treat the theory as incomplete: "true" when it proves statement, "false" when it proves the opposite, and "unknown" when it proves neither. A statement with "not" must be proved like any other.' },
      files: { 'evaluate_statement/facts.ts': factStore(facts, 'The theory: facts and rules.'), 'types.ts': VERDICTS },
      inputs: { statement: question.question }, expected });
  });
}

// EntailmentBank -------------------------------------------------------------------------------------------

function entailmentRows() {
  return once('entailmentbank', () => SOURCES.entailmentbank.files.flatMap(file => {
    const path = cachePath(CACHE, 'entailmentbank', SOURCES.entailmentbank.revision, file.path);
    if (!existsSync(path)) throw missing('entailmentbank');
    return jsonl(path).map(row => ({ ...row, split: file.split === 'train' ? 'train' : 'test' }));
  }));
}

/**
 * EntailmentBank: select the stored facts that together support a hypothesis, from a store with distractors;
 * a verifier accepts the gold premises with at most two extra. The counterpart removes a needed premise.
 */
export function entailmentPremises(seed, index) {
  const rows = entailmentRows();
  // Proofs with at least two premises, tried from a spread-out position for this index.
  const leavesOf = candidate => [...new Set(candidate.proof.match(/sent\d+/g) ?? [])];
  let row;
  for (let offset = 0; !row && offset < 50; offset++) {
    const candidate = rows[(index * 7919 + offset * 104729) % rows.length];
    if (leavesOf(candidate).length >= 2) row = candidate;
  }
  if (!row) throw new Error(`EntailmentBank: no multi-premise proof near index ${index}`);
  const rng = new Random(seed, `entailment:${row.id}`);
  const triples = row.meta.triples;
  const leaves = leavesOf(row);
  const removed = rng.pick(leaves);
  const certificate = `support-${digest(`${row.id}:${row.hypothesis}`)}`;
  const shape = row.id.replace(/[^A-Za-z0-9]+/g, '_');
  return [['supported', null], ['premise_removed', removed]].map(([variant, removedId]) => {
    const kept = Object.entries(triples).filter(([id]) => id !== removedId);
    const facts = rng.shuffle(kept).map(([source, text], i) => ({ id: `F${i + 1}`, text, source }));
    const idOf = Object.fromEntries(facts.map(f => [f.source, f.id]));
    const gold = removedId ? [] : leaves.map(id => idOf[id]);
    const expected = removedId ? { supported: false, certificate: null } : { supported: true, certificate };
    const checker = `const GOLD: string[] = ${literal(gold)};
/**
 * Check a selection of fact ids as the premises that together support the hypothesis. Accepts a selection that
 * contains every needed premise and at most two others, and then returns a certificate; it says nothing else.
 */
export function check(ids: string[]): { ok: boolean, certificate: string | null } {
  const chosen = [...new Set(ids)];
  const ok = GOLD.length > 0 && GOLD.every(id => chosen.includes(id)) && chosen.length <= GOLD.length + 2;
  return { ok, certificate: ok ? ${JSON.stringify(certificate)} : null };
}
`;
    const reference = removedId ? [evalCall(READ_ALL), returnCall(expected)] :
      [evalCall(READ_ALL), evalCall(`premises.check(${JSON.stringify(gold)})`), returnCall(expected)];
    return curriculumCase({ family: 'entailment_premises', shape, variant, pairGroup: `entailment:${shape}`,
      splitGroup: `entailmentbank:${row.id}`, split: row.split, slice: 'observation_followup', domain: 'logic', mode: 'single_call',
      worldSemantics: 'open_world', inline: 'avoid',
      evidence: { world: leaves.map(id => triples[id]), retrieved: [],
        background: [`source: EntailmentBank task 2 ${SOURCES.entailmentbank.revision} ${row.id}`, `proof: ${row.proof}`, removedId ? `removed: ${triples[removedId]}` : 'complete'] },
      assumptions: ['Everyday science knowledge may bridge wording, but every step must rest on a stored fact.'],
      minimumSequence: ['read the fact store', 'find the facts that chain to the hypothesis', 'check them, or conclude a needed fact is missing'],
      reference: { root: reference },
      root: { name: 'support_hypothesis', args: { hypothesis: 'string' }, returns: 'Support',
        instructions: 'Find the facts in the store (facts) that together support hypothesis: each fact needed for a short chain of reasoning to it, and no more. Check your selection with premises.check(ids); if it accepts, return supported true with its certificate. If the store lacks a fact the reasoning needs, return supported false with a null certificate.' },
      files: { 'support_hypothesis/facts.ts': factStore(facts.map(({ id, text }) => ({ id, text })), 'Science facts, most of them unrelated to any one question.'),
        'support_hypothesis/premises.ts': checker,
        'types.ts': 'export type Support = { supported: boolean, certificate: string | null };\n' },
      inputs: { hypothesis: row.hypothesis }, expected });
  });
}

// αNLI -----------------------------------------------------------------------------------------------------

function anliRows() {
  return once('anli', () => {
    const dir = findDir(cachePath(CACHE, 'anli', SOURCES.anli.revision, 'anli'), 'train.jsonl');
    if (!dir) throw missing('anli');
    const rows = [];
    for (const [name, split] of [['train', 'train'], ['dev', 'test'], ['test', 'test']]) {
      const items = jsonl(join(dir, `${name}.jsonl`));
      const labels = readFileSync(join(dir, `${name}-labels.lst`), 'utf8').split('\n').filter(Boolean);
      // One hypothesis pair per story: training stories repeat with many crowd-written pairs.
      const seen = new Set();
      items.forEach((item, i) => {
        const story = item.story_id.replace(/-\d+$/, '');
        if (seen.has(story)) return;
        seen.add(story);
        rows.push({ ...item, story, label: labels[i], split });
      });
    }
    return rows;
  });
}

/** αNLI: for each story, which of two hypotheses better explains how the beginning led to the ending. */
export function anliBatch(seed, index) {
  const rows = anliRows();
  const rng = new Random(seed, `anli:${index}`);
  const start = (index * 5 * 7919) % rows.length;
  const split = rows[start].split;
  const stories = [];
  for (let i = start; stories.length < 5 && i < start + 50; i++) if (rows[i % rows.length].split === split) stories.push(rows[i % rows.length]);
  const items = stories.map((row, i) => ({ id: `N${i + 1}`, beginning: row.obs1, ending: row.obs2, a: row.hyp1, b: row.hyp2 }));
  const expected = Object.fromEntries(stories.map((row, i) => [`N${i + 1}`, row.label === '1' ? 'a' : 'b']));
  void rng;
  return [curriculumCase({ family: 'anli_batch', shape: `stories${index}`, variant: 'a', splitGroup: `anli:${stories[0].story}`, split,
    slice: 'inline_placement', domain: 'logic', mode: 'single_call', inline: 'required', worldSemantics: 'defeasible',
    evidence: { world: stories.map(row => `${row.story}: ${row.label}`), retrieved: [], background: [`source: αNLI ${SOURCES.anli.revision}`] },
    assumptions: ['Everyday knowledge decides which explanation is more plausible; neither is certain.'],
    minimumSequence: ['judge each story in its own child', 'collect the choices by id'],
    reference: { root: [evalCall(`const choices = await Promise.all(stories.map(story => nl<'a' | 'b'>\`Which hypothesis, a or b, better explains how story's beginning led to its ending?\`(story)));
return Object.fromEntries(stories.map((story, i) => [story.id, choices[i]]));`), returnCall(expected)],
      children: items.map(item => ({ match: JSON.stringify(item.beginning), value: expected[item.id] })) },
    root: { name: 'explain_stories', args: { stories: 'Story[]' }, returns: 'Record<string, "a" | "b">',
      instructions: 'For each of stories, decide which hypothesis, a or b, better explains how its beginning led to its ending. Judge each story separately. Return a record from story id to "a" or "b".' },
    files: { 'types.ts': 'export type Story = { id: string, beginning: string, ending: string, a: string, b: string };\n' },
    inputs: { stories: items }, expected })];
}

// CommaQA --------------------------------------------------------------------------------------------------

function commaqaRows() {
  return once('commaqa', () => {
    const dir = findDir(cachePath(CACHE, 'commaqa', SOURCES.commaqa.revision, 'commaqa_explicit'), 'train.json');
    if (!dir) throw missing('commaqa');
    const rows = [];
    for (const [name, split] of [['train', 'train'], ['dev', 'test'], ['test', 'test']]) {
      JSON.parse(readFileSync(join(dir, `${name}.json`), 'utf8')).forEach((world, w) => {
        // per_fact_context maps each knowledge-base fact to the sentence that renders it.
        const rendered = Object.entries(world.per_fact_context);
        const text = rendered.filter(([fact]) => fact.startsWith('text_')).map(([, sentence]) => sentence);
        const tables = rendered.filter(([fact]) => fact.startsWith('table_')).map(([, sentence]) => sentence);
        world.qa_pairs.forEach((qa, q) => rows.push({ id: `${name}:${w}:${q}`, world: `${name}:${w}`, split, qa, text, tables }));
      });
    }
    return rows;
  });
}

/**
 * CommaQA: a question answered by combining two specialists that each see half the evidence: one reads the
 * tables, the other the text passages. The decomposition's per-step answers are the specialists' reference
 * answers; the model never sees the decomposition.
 */
export function commaqaQuestion(seed, index) {
  const rows = commaqaRows();
  const row = rows[(index * 7919) % rows.length];
  const rng = new Random(seed, `commaqa:${row.id}`);
  const steps = row.qa.decomposition;
  const results = [];
  const lines = [], children = [];
  steps.forEach((step, i) => {
    // `#k` refers to an earlier step's answers.
    const question = step.q.replace(/#(\d+)/g, (_, k) => results[Number(k) - 1].join(' and '));
    const expert = step.m === 'table' ? 'table_expert' : 'text_expert';
    lines.push(`const step${i + 1} = await ${expert}(${JSON.stringify(question)});`);
    children.push({ match: [`You are inside this call: ${expert}`, JSON.stringify(question)], value: step.a });
    results.push(step.a);
  });
  const answer = [...row.qa.answer].sort();
  lines.push(`return [...new Set(step${steps.length})].sort();`);
  const store = (facts, what) => factStore(rng.shuffle(facts).map((text, i) => ({ id: `${what[0].toUpperCase()}${i + 1}`, text })), `The ${what}.`);
  const expertFile = (what, source) => nlFile({ args: { question: 'string' }, returns: 'string[]',
    description: `Answer a question from the ${what} alone.`,
    instructions: `Answer question using only the ${what} in ${source}.page(n). Return every name that answers it (an empty list when none does).` });
  return [curriculumCase({ family: 'commaqa_question', shape: row.id.replace(/:/g, '_'), variant: 'q', splitGroup: `commaqa:${row.world}`,
    split: row.split, slice: 'nested_scoped', domain: 'relational', mode: 'single_call', inline: 'avoid', named: 'required',
    worldSemantics: 'closed_world',
    evidence: { world: row.qa.facts_used, retrieved: [], background: [`source: CommaQA explicit ${SOURCES.commaqa.revision} ${row.id}`,
      `decomposition: ${steps.map(s => `[${s.m}] ${s.q} => ${s.a.join(', ')}`).join(' | ')}`] },
    minimumSequence: ['split the question into steps', 'ask the specialist that holds each step\'s evidence', 'feed each answer into the next step'],
    reference: { root: [evalCall(lines.join('\n')), returnCall(answer)], children },
    root: { name: 'answer_question', args: { question: 'string' }, returns: 'string[]',
      instructions: 'Answer question about the movie world. The evidence is split between two specialists: table_expert answers questions from the tables and text_expert from the text passages; neither sees the other\'s evidence. Ask them the steps the question needs, feeding each answer into the next. Return the answer names, sorted alphabetically, without duplicates.' },
    files: {
      'answer_question/table_expert.nl': expertFile('tables', 'tables'),
      'answer_question/table_expert/tables.ts': store(row.tables, 'table rows'),
      'answer_question/text_expert.nl': expertFile('text passages', 'passages'),
      'answer_question/text_expert/passages.ts': store(row.text, 'text passages'),
    },
    inputs: { question: row.qa.question }, expected: answer })];
}
