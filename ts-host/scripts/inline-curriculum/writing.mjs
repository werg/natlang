// Constrained writing: a natural-language function whose answer is open text, admitted by code-checked constraints
// (evaluation/constraints.ts, oracle level "constraints"). The first family of the chat-and-writing domain
// (plans/neuralese/MAPLE_NESTED.md §4a, owner 2026-10-05: chat and writing tasks become natlang domains).
import { checkConstraints, describeConstraint } from '../../dist/evaluation/constraints.js';
import { Random, curriculumCase, returnCall } from './lib.mjs';

const BRIEFS = [
  { kind: 'email', what: 'an email to the team announcing that the {topic} review moves to {day}', subject: 'review', words: ['review', 'team'] },
  { kind: 'announcement', what: 'a short announcement that the {topic} workshop opens for sign-ups on {day}', subject: 'workshop', words: ['workshop', 'sign'] },
  { kind: 'product', what: 'a product description for a {topic} kit aimed at beginners', subject: 'kit', words: ['kit', 'beginners'] },
  { kind: 'explanation', what: 'an explanation for a new colleague of why the {topic} checklist exists', subject: 'checklist', words: ['checklist', 'colleague'] },
  { kind: 'apology', what: 'an apology to customers for the delayed {topic} shipment, with the new date {day}', subject: 'shipment', words: ['shipment', 'customers'] },
  { kind: 'summary', what: 'a summary of a meeting that decided to pilot the {topic} plan on {day}', subject: 'meeting', words: ['meeting', 'pilot'] },
  { kind: 'tip', what: 'a practical tip for keeping a {topic} log up to date', subject: 'log', words: ['log', 'update'] },
  { kind: 'invitation', what: 'an invitation to a {topic} tasting on {day}', subject: 'tasting', words: ['tasting', 'invite'] },
];
const TOPICS = ['garden', 'budget', 'safety', 'onboarding', 'pottery', 'cycling', 'archive', 'recycling', 'coffee', 'robotics',
  'library', 'kitchen', 'volunteer', 'telescope', 'ceramics', 'harvest'];
const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
const FILLER = ['Please read the details below.', 'Everything else stays the same.', 'Thank you for your patience.',
  'Questions are welcome at any time.', 'The plan is simple and clear.', 'We will share more soon.',
  'Each step has an owner.', 'Small changes add up over time.', 'This keeps our work steady.', 'Bring a friend if you like.'];

function constraintSet(rng, brief) {
  const pool = [];
  const lo = rng.int(3, 6) * 10;
  pool.push({ kind: 'word_count', min: lo, max: lo + rng.int(2, 5) * 10 });
  pool.push({ kind: 'include_words', words: rng.sample(brief.words, rng.int(1, 2)) });
  pool.push({ kind: 'exclude_words', words: rng.sample(['very', 'really', 'basically', 'just', 'actually', 'literally'], rng.int(1, 2)) });
  pool.push(rng.next() < 0.5 ? { kind: 'bullet_count', count: rng.int(2, 4) } : { kind: 'paragraph_count', count: rng.int(2, 3) });
  pool.push(rng.next() < 0.5 ? { kind: 'no_commas' } : { kind: 'all_lowercase' });
  pool.push(rng.next() < 0.5 ? { kind: 'title' } : { kind: 'placeholders', min: rng.int(1, 2) });
  pool.push(rng.next() < 0.5 ? { kind: 'ends_with', text: 'Is there anything else I can help with?' } : { kind: 'starts_with', text: 'Hello everyone' });
  const chosen = rng.sample(pool, rng.int(2, 4));
  // word_count and paragraph/bullet counts interact; keep at most one structure constraint with a word range.
  return chosen;
}

/** A reference answer that satisfies the constraints, built from the brief (verified before use). */
function reference(rng, brief, topic, day, constraints) {
  const has = kind => constraints.find(c => c.kind === kind);
  const lower = Boolean(has('all_lowercase'));
  const include = has('include_words')?.words ?? [];
  const exclude = new Set((has('exclude_words')?.words ?? []).map(w => w.toLowerCase()));
  const opening = `This note is about the ${topic} ${brief.subject} on ${day}.`;
  let body = [opening, `This matters for the ${include.length ? include.join(' and the ') : brief.subject}.`];
  const range = has('word_count');
  const target = range ? Math.floor(((range.min ?? 40) + (range.max ?? (range.min ?? 40) + 20)) / 2) : 50;
  const fillers = rng.shuffle([...FILLER]);
  let i = 0;
  const count = text => (text.match(/[\p{L}\p{N}'’-]+/gu) ?? []).length;
  const framing = (has('starts_with') ? 2 : 0) + (has('ends_with') ? 8 : 0) + (has('title') ? 3 : 0) + (has('placeholders') ? 4 : 0) + 6;
  while (count(body.join(' ')) + framing < target && i < 200) body.push(fillers[i++ % fillers.length]);
  const placeholders = has('placeholders') ? ` Reply to [name] at [email].` : '';
  let paragraphs;
  const bullets = has('bullet_count');
  const paraCount = has('paragraph_count')?.count;
  if (bullets) {
    const items = Array.from({ length: bullets.count }, (_, k) => `- Point ${k + 1} about the ${topic} ${brief.subject}.`);
    paragraphs = [body.join(' ') + placeholders, items.join('\n')];
  } else if (paraCount) {
    const per = Math.ceil(body.length / paraCount);
    paragraphs = Array.from({ length: paraCount }, (_, k) => body.slice(k * per, (k + 1) * per).join(' ') || 'More follows.');
    paragraphs[0] += placeholders;
  } else paragraphs = [body.join(' ') + placeholders];
  let text = paragraphs.join('\n\n');
  if (has('title')) text = `<<The ${topic} ${brief.subject}>>\n\n${text}`;
  if (has('starts_with')) text = `${has('starts_with').text}. ${text}`;
  if (has('ends_with')) text = `${text}\n\n${has('ends_with').text}`;
  if (has('no_commas')) text = text.replace(/,/g, '');
  if (lower) text = text.toLowerCase();
  for (const word of exclude) text = text.replace(new RegExp(`\\b${word}\\b`, 'gi'), '');
  return text;
}

export function constrainedWriting(seed, index) {
  for (let attempt = 0; attempt < 20; attempt++) {
    const rng = new Random(seed, `writing:${index}:${attempt}`);
    const brief = BRIEFS[rng.int(0, BRIEFS.length - 1)];
    const topic = TOPICS[rng.int(0, TOPICS.length - 1)];
    const day = DAYS[rng.int(0, DAYS.length - 1)];
    const constraints = constraintSet(rng, brief);
    // starts_with and all_lowercase contradict each other ("Hello" is capitalized); so do title brackets and nothing else.
    if (constraints.some(c => c.kind === 'all_lowercase') && constraints.some(c => c.kind === 'starts_with' || c.kind === 'ends_with')) continue;
    const text = reference(rng, brief, topic, day, constraints);
    if (!checkConstraints(text, constraints).passed) continue;
    const what = brief.what.replace('{topic}', topic).replace('{day}', day);
    const rules = constraints.map(describeConstraint).join(' ');
    const shape = `writing${index}`;
    const record = curriculumCase({ family: 'constrained_writing', shape, variant: brief.kind, slice: 'writing', domain: 'writing',
      mode: 'single_call', inline: 'optional',
      evidence: { world: [], retrieved: [], background: [] },
      plausibleActions: [], minimumSequence: ['write the text to the brief', 'check every constraint before returning'],
      reference: { root: [returnCall(text)] },
      root: { name: `write_${brief.kind}`, args: {}, returns: 'string',
        instructions: `Write ${what}. ${rules} Return only the text.` },
      inputs: {}, expected: constraints });
    record.semantics.oracle = { level: 'constraints' };
    return [record];
  }
  throw new Error(`no consistent constrained-writing case for ${seed}:${index}`);
}
