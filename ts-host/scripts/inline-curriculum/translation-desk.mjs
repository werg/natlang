// Translation desk: messages from the translation source pools (translation-sources-20261004: the static candidate
// pool and NLLB-Seed), each addressed to a recipient whose language comes from a crisp directory. Every message not
// already in its recipient's language is translated by an inline nl call; the others pass through unchanged. The
// pools' host-only reference translations answer the children and give the expected record.
//
// Demonstrations only (replay-demonstrations.mjs): a teacher's translation would need a per-item similarity oracle
// that the answer comparison does not have yet, and the pools are held pending source-policy review, so cases carry
// that hold. (Owner 2026-10-05: static inline-lambda data from labeled source pools.)
import { closeSync, existsSync, openSync, readSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Random, curriculumCase, evalCall, literal, returnCall } from './lib.mjs';

const POOLS = fileURLToPath(new URL('../../../data/neuralese/corpora/translation-sources-20261004/self-improvement-expansion-20261004/', import.meta.url));
const NAMES = ['translation-static-candidate-v2', 'nllb-seed-static-full-v2'];
const LIMIT = Number(process.env.NATLANG_TRANSLATION_POOL_LIMIT ?? 60000);
const title = name => name.replace(/(^|[\s(])\w/g, c => c.toUpperCase());

/** The lines of a file, read in chunks (the NLLB pool is larger than the longest string V8 allows). */
function* lines(path) {
  const fd = openSync(path, 'r'), buffer = Buffer.alloc(16 << 20);
  let rest = '';
  try {
    for (let n; (n = readSync(fd, buffer, 0, buffer.length, null)) > 0;) {
      const parts = (rest + buffer.toString('utf8', 0, n)).split('\n');
      rest = parts.pop();
      yield* parts;
    }
    if (rest) yield rest;
  } finally { closeSync(fd); }
}

let pairs = null;
/** Reference pairs (source and target language and text), train split only, at most LIMIT per pool. */
function loadPairs() {
  if (pairs) return pairs;
  pairs = [];
  for (const name of NAMES) {
    const dir = POOLS + name;
    if (!existsSync(`${dir}/translation-source-ir.jsonl`)) continue;
    const references = new Map();
    for (const line of lines(`${dir}/translation-references.host-only.jsonl`)) {
      if (!line) continue;
      const r = JSON.parse(line);
      if ((r.split ?? r.original_split ?? 'train') !== 'train' || r.role === 'heldout') continue;
      references.set(r.ir_id, r);
    }
    let kept = 0;
    for (const line of lines(`${dir}/translation-source-ir.jsonl`)) {
      if (kept >= LIMIT) break;
      if (!line) continue;
      const ir = JSON.parse(line), ref = references.get(ir.id);
      if (!ref) continue;
      const input = ir.semantics.inputs;
      if (input.source_text.length > 400 || ref.target_text.length > 400) continue;
      pairs.push({ pool: name, from: title(input.source_language), to: title(input.target_language), text: input.source_text,
        translation: ref.target_text, group: ref.source_group ?? ir.source_groups?.[0] ?? ir.id, irId: ir.id });
      kept++;
    }
  }
  if (!pairs.length) throw new Error('translation pools are not present; sync translation-sources-20261004');
  return pairs;
}

export function translationDesk(seed, index) {
  const rng = new Random(seed, `translation-desk:${index}`);
  const all = loadPairs();
  const pool = rng.pick(NAMES.filter(name => all.some(p => p.pool === name)));
  const candidates = all.filter(p => p.pool === pool);
  const n = rng.int(4, 7);
  const chosen = rng.sample(candidates, n);
  // One or two messages already in their recipient's language: pass through.
  const passCount = rng.int(1, 2);
  const messages = chosen.map((p, i) => ({ id: `M${i + 1}`, recipient: `R${i + 1}`, ...p, pass: i < passCount }));
  for (const m of messages) if (m.pass) { m.to = m.from; m.translation = m.text; }
  const shuffled = rng.shuffle(messages).map((m, i) => ({ ...m, id: `M${i + 1}`, recipient: `R${i + 1}` }));
  const expected = Object.fromEntries(shuffled.map(m => [m.id, m.translation]));
  const plainMessages = shuffled.map(m => ({ id: m.id, recipient: m.recipient, language: m.from, text: m.text }));
  const directory = shuffled.map(m => ({ id: m.recipient, language: m.to }));
  const code = `const all = desk.messages();
const languages = Object.fromEntries(desk.recipients().map(r => [r.id, r.language]));
const translate = nl<(message: Message, language: string) => Promise<string>>\`Translate message's text from its language into language faithfully, keeping names and register. Return only the translation.\`;
const out = await Promise.all(all.map(message => message.language === languages[message.recipient]
  ? message.text : translate(message, languages[message.recipient])));
return Object.fromEntries(all.map((message, i) => [message.id, out[i]]));`;
  const shape = `translation${index}`;
  const record = curriculumCase({ family: 'translation_desk', shape, variant: 'v0', splitGroup: `translation:${shuffled.map(m => m.group).sort()[0]}`,
    slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'required',
    evidence: { world: shuffled.map(m => `${m.id}: ${m.from} -> ${m.to}`), retrieved: [], background: [] },
    minimumSequence: ['look up each recipient\'s language in the directory', 'pass through messages already in that language',
      'translate every other message in its own nl call'],
    reference: { root: [evalCall(code), returnCall(expected)],
      children: shuffled.filter(m => !m.pass).map(m => ({ match: ['Translate message', JSON.stringify(m.id)], value: m.translation })) },
    root: { name: 'translation_desk', args: {}, returns: 'Record<string, string>',
      instructions: 'Deliver every message in desk.messages() to its recipient in the recipient\'s language (desk.recipients()). A message already in that language goes out unchanged; translate the others faithfully. Return a record from message id to the text to deliver.' },
    files: { 'translation_desk/desk.ts': `const MESSAGES = ${literal(plainMessages)};\nconst RECIPIENTS = ${literal(directory)};\n` +
      '/** The outgoing messages, each with its language and recipient. */\nexport function messages(): Message[] { return MESSAGES; }\n' +
      '/** The recipient directory: the language each recipient reads. */\nexport function recipients(): { id: string, language: string }[] { return RECIPIENTS; }\n',
      'types.ts': 'export type Message = { id: string, recipient: string, language: string, text: string };\n' },
    inputs: {}, expected });
  record.license = pool.startsWith('nllb') ? 'CC-BY-SA-4.0' : 'see translation-sources-20261004 manifest';
  record.gold_sources = [`${pool}-host-references`];
  record.dataset = pool;
  record.dataset_records = shuffled.map(m => m.irId);
  record.curriculum.hold = 'demonstration-only: no per-item translation oracle; source pools held pending source-policy review';
  return [record];
}
