// Edit stream: a document (news paragraphs of different topics, ag_news) and a stream of natural-language edit
// instructions, applied one at a time with iterateOn and an nl step that takes the current text and the next
// instruction. Some instructions locate their target by meaning ("delete the paragraph about sport"), others are
// mechanical (rename a name everywhere, add a title, append a closing line, lowercase, drop commas). The reference
// applies every edit exactly, so each intermediate state is known; the oracle is the set of code-checked constraints
// that hold on the final text (evaluation/constraints.ts), so teacher outputs are judged too. (Owner 2026-10-05:
// iterate over a stream of instructions doing natural-language modifications.)
import { checkConstraints } from '../../dist/evaluation/constraints.js';
import { rowsOf } from './labeled.mjs';
import { Random, curriculumCase, evalCall, literal, returnCall } from './lib.mjs';

const TOPIC = { World: 'world affairs', Sports: 'sport', Business: 'business', 'Sci/Tech': 'science or technology' };
const RENAMES = ['Alder', 'Brennan', 'Castell', 'Dorrance', 'Elmsworth', 'Fairley', 'Galloway', 'Hartwell'];
const CLOSINGS = ['More updates will follow next week.', 'Send corrections to the editors.', 'This digest is published every Monday.'];
const TITLES = ['Weekly Digest', 'News in Brief', 'The Monday Roundup', 'Headlines'];

const paragraphs = text => text.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
const join = list => list.join('\n\n');
const words = text => text.match(/[\p{L}\p{N}'’-]+/gu) ?? [];

/** Edits: (rng, state) -> { instruction, apply(text) -> text, constraint(final) -> WritingConstraint | null } or null. */
const EDITS = {
  delete_topic(rng, s) {
    const live = s.paras.filter(p => !p.deleted);
    if (live.length < 3) return null;
    const target = rng.pick(live);
    const marker = words(target.text).filter(w => w.length >= 7 && !s.paras.some(p => p !== target && p.text.includes(w)))[0];
    if (!marker) return null;
    target.deleted = true;
    return { instruction: `Delete the paragraph about ${TOPIC[target.label]}.`, kind: 'semantic',
      apply: text => join(paragraphs(text).filter(p => !p.includes(marker))),
      constraint: () => ({ kind: 'exclude_words', words: [marker] }) };
  },
  move_topic_first(rng, s) {
    const live = s.paras.filter(p => !p.deleted);
    if (live.length < 2) return null;
    const target = rng.pick(live.slice(1));
    const marker = words(target.text).filter(w => w.length >= 7 && !s.paras.some(p => p !== target && p.text.includes(w)))[0];
    if (!marker || s.titled) return null;
    return { instruction: `Move the paragraph about ${TOPIC[target.label]} to the top.`, kind: 'semantic',
      apply: text => { const list = paragraphs(text), i = list.findIndex(p => p.includes(marker)); return join([list[i], ...list.filter((_, j) => j !== i)]); },
      constraint: final => paragraphs(final)[0].includes(marker) ? null : null };
  },
  rename(rng, s) {
    const names = [...new Set(s.paras.filter(p => !p.deleted).flatMap(p => words(p.text)))].filter(w => /^[A-Z][a-z]{3,}$/.test(w) && !s.renamed.has(w));
    if (!names.length) return null;
    const from = rng.pick(names), to = rng.pick(RENAMES.filter(r => !s.renamed.has(r)));
    s.renamed.add(from); s.renamed.add(to);
    return { instruction: `Replace "${from}" with "${to}" everywhere.`, kind: 'mechanical',
      apply: text => text.replace(new RegExp(`\\b${from}\\b`, 'g'), to),
      constraint: () => ({ kind: 'exclude_words', words: [from] }) };
  },
  title(rng, s) {
    if (s.titled) return null;
    s.titled = true;
    const title = rng.pick(TITLES);
    return { instruction: `Add the title "${title}" on its own first line, in double angle brackets.`, kind: 'mechanical',
      apply: text => `<<${title}>>\n\n${text}`, constraint: () => ({ kind: 'title' }) };
  },
  closing(rng, s) {
    if (s.closed) return null;
    s.closed = true;
    const line = rng.pick(CLOSINGS);
    return { instruction: `End the document with the sentence "${line}" as its own paragraph.`, kind: 'mechanical',
      apply: text => `${text}\n\n${line}`, constraint: final => final.trimEnd().endsWith(line) ? { kind: 'ends_with', text: line } : null };
  },
  no_commas(rng, s) {
    if (s.noCommas) return null;
    s.noCommas = true;
    return { instruction: 'Remove every comma.', kind: 'mechanical', apply: text => text.replace(/,/g, ''), constraint: () => ({ kind: 'no_commas' }) };
  },
  lowercase(rng, s) {
    if (s.lower) return null;
    s.lower = true;
    return { instruction: 'Rewrite everything in lowercase.', kind: 'mechanical', apply: text => text.toLowerCase(), constraint: () => ({ kind: 'all_lowercase' }) };
  },
};

export function editStream(seed, index) {
  const rng = new Random(seed, `edit-stream:${index}`);
  for (let attempt = 0; attempt < 40; attempt++) {
    const labels = rng.sample(Object.keys(TOPIC), rng.int(3, 4));
    const pool = rowsOf('ag_news');
    const paras = labels.map(label => ({ label, text: rng.pick(pool.filter(r => r.label === label && r.text.length > 120)).text }));
    const state = { paras, renamed: new Set(), titled: false, closed: false, noCommas: false, lower: false };
    const steps = [];
    const want = rng.int(3, 5);
    for (let tries = 0; steps.length < want && tries < 30; tries++) {
      const name = rng.pick(Object.keys(EDITS));
      // Lowercasing must come last among case-sensitive edits; keep it as the final step only.
      if (name === 'lowercase' && steps.length < want - 1) continue;
      const edit = EDITS[name](rng, state);
      if (edit) steps.push({ name, ...edit });
    }
    if (!steps.some(s => s.kind === 'semantic') || steps.length < 3) continue;
    const states = [join(paras.map(p => p.text))];
    for (const step of steps) states.push(step.apply(states.at(-1)));
    const final = states.at(-1);
    const constraints = steps.map(step => step.constraint(final)).filter(Boolean);
    // The reference's lowercasing makes earlier exact-case constraints vacuous only if they are case-insensitive: they are.
    if (!constraints.length || !checkConstraints(final, constraints).passed) continue;
    const instructions = steps.map(step => step.instruction);
    const code = `const edit = nl<(text: string, instruction: string) => Promise<string>>\`Apply instruction to text and return the whole edited text, changing nothing else.\`;
const queue = docs.instructions();
const final = await iterateOn(async (s: { text: string, next: number }) => ({ text: await edit(s.text, queue[s.next]), next: s.next + 1 }),
  { text: docs.document(), next: 0 }).withLimit({maxSteps: 32}).until(s => s.next >= queue.length);
return final.text;`;
    const record = curriculumCase({ family: 'edit_stream', shape: `edits${index}`, variant: 'v0', slice: 'inline_placement', domain: 'writing',
      mode: 'single_call', inline: 'required', iterate: 'required',
      evidence: { world: steps.map(s => `${s.name}: ${s.instruction}`), retrieved: [], background: [] },
      minimumSequence: ['read the document and the instruction queue', 'apply one instruction per step with iterateOn and an nl edit',
        'return the final text'],
      reference: { root: [evalCall(code), returnCall(final)],
        children: steps.map((step, i) => ({ match: ['Apply instruction to text', JSON.stringify(step.instruction).slice(1, -1).slice(0, 40)], value: states[i + 1] })) },
      root: { name: 'apply_edits', args: {}, returns: 'string',
        instructions: 'Apply the editing instructions from docs.instructions() to the document from docs.document(), one after another and in order, each to the result of the previous one. Change nothing an instruction does not ask for. Return the final text.' },
      files: { 'apply_edits/docs.ts': `const DOCUMENT = ${JSON.stringify(states[0])};\nconst INSTRUCTIONS = ${literal(instructions)};\n` +
        '/** The document to edit. */\nexport function document(): string { return DOCUMENT; }\n' +
        '/** The editing instructions, in the order they must be applied. */\nexport function instructions(): string[] { return INSTRUCTIONS; }\n' },
      inputs: {}, expected: constraints });
    record.semantics.oracle = { level: 'constraints' };
    record.license = 'AG News (academic use) via labeled curriculum sources';
    record.gold_sources = ['ag_news-labels', 'deterministic-edit-reference'];
    record.dataset = 'ag_news';
    return [record];
  }
  throw new Error(`no edit stream for ${seed}:${index}`);
}
