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

const PEOPLE = ['Amara', 'Bruno', 'Chen', 'Dalia', 'Emeka', 'Farah', 'Goran', 'Hana', 'Ivo', 'Jun', 'Kofi', 'Lena'];
const PLACES = ['Lisbon', 'Nairobi', 'Oslo', 'Quito', 'Hanoi', 'Tbilisi', 'Perth', 'Dakar', 'Cusco', 'Riga'];

/** A source passage with facts (names, a place, numbers) that a rewrite must keep. */
function passage(rng) {
  const [a, b] = rng.sample(PEOPLE, 2);
  const place = PLACES[rng.int(0, PLACES.length - 1)];
  const topic = TOPICS[rng.int(0, TOPICS.length - 1)];
  const count = rng.int(12, 480);
  const year = rng.int(2019, 2026);
  const percent = rng.int(5, 95);
  const sentences = [
    `In ${year} ${a} started a ${topic} project in ${place} with a small group of neighbours.`,
    `At first the group met once a week in a borrowed room and kept notes by hand.`,
    `By the end of the second season the project had ${count} regular members.`,
    `${b} joined to handle the schedule and introduced a shared calendar that everyone could edit.`,
    `Attendance rose by ${percent} percent after the calendar went live.`,
    `The group now plans to open a second site and is looking for volunteers who can commit a few hours a month.`,
  ];
  return { text: sentences.join(' '), facts: [a, b, place, String(count), String(year)], topic };
}

/** Rewriting and summarizing: keep the facts, meet a budget and a register; checked by code. */
export function constrainedRewrite(seed, index) {
  for (let attempt = 0; attempt < 20; attempt++) {
    const rng = new Random(seed, `rewrite:${index}:${attempt}`);
    const source = passage(rng);
    const task = rng.sample(['summarize', 'bullets', 'plain'], 1)[0];
    const keep = rng.sample(source.facts, rng.int(2, 4));
    const constraints = [{ kind: 'include_words', words: keep }];
    let instruction, text;
    const s = source.text.split(/(?<=\.)\s+/);
    if (task === 'summarize') {
      const max = rng.int(3, 5) * 10;
      constraints.push({ kind: 'word_count', max }, { kind: 'sentence_count', max: 3 });
      instruction = `Summarize the passage for a newsletter.`;
      text = `${s[0]} ${s[2]} ${s[3].replace(/ and introduced.*$/, '.')}`;
    } else if (task === 'bullets') {
      const n = rng.int(3, 4);
      constraints.push({ kind: 'bullet_count', count: n }, { kind: 'no_commas' });
      instruction = `Rewrite the passage as a list of key facts.`;
      text = s.slice(0, n).map(line => `- ${line.replace(/,/g, '').replace(/\.$/, '')}`).join('\n');
    } else {
      constraints.push({ kind: 'exclude_words', words: ['project'] }, { kind: 'word_count', max: 70 });
      instruction = `Rewrite the passage in plain words for a ten-year-old, without the word "project".`;
      text = `${s[0].replace('project', 'club')} ${s[2].replace('project', 'club')} ${s[3]} ${s[4]}`;
    }
    if (!checkConstraints(text, constraints).passed) continue;
    const rules = constraints.map(describeConstraint).join(' ');
    const record = curriculumCase({ family: 'constrained_rewrite', shape: `rewrite${index}`, variant: task, slice: 'writing',
      domain: 'writing', mode: 'single_call', inline: 'optional',
      evidence: { world: [], retrieved: [], background: [] }, plausibleActions: [],
      minimumSequence: ['read the passage', 'rewrite it keeping the named facts', 'check every constraint before returning'],
      reference: { root: [returnCall(text)] },
      root: { name: `rewrite_${task}`, args: { passage: 'string' }, returns: 'string',
        instructions: `${instruction} Keep these facts: ${keep.join(', ')}. ${rules} Return only the text.` },
      inputs: { passage: source.text }, expected: constraints });
    record.semantics.oracle = { level: 'constraints' };
    return [record];
  }
  throw new Error(`no consistent rewrite case for ${seed}:${index}`);
}

const ITEMS = ['notebooks', 'lamps', 'mugs', 'chairs', 'plants', 'cables', 'posters', 'boxes'];
const STYLES = [
  { ask: 'From now on please keep every answer under 25 words.', constraint: { kind: 'word_count', max: 25 } },
  { ask: 'Please answer everything in lowercase from now on, it is easier on my eyes.', constraint: { kind: 'all_lowercase' } },
  { ask: 'Could you start every reply with "Sure:" from now on?', constraint: { kind: 'starts_with', text: 'Sure:' } },
  { ask: 'Please never use commas in your replies, my screen reader stumbles on them.', constraint: { kind: 'no_commas' } },
  { ask: 'Answer in at most two sentences from now on, please.', constraint: { kind: 'sentence_count', max: 2 } },
];

/** A chat turn: reply to the last message of a conversation. The reply must keep a format preference the user set
 * earlier and answer from facts stated earlier in the conversation; both are checked by code. */
export function chatReply(seed, index) {
  for (let attempt = 0; attempt < 20; attempt++) {
    const rng = new Random(seed, `chat:${index}:${attempt}`);
    const name = PEOPLE[rng.int(0, PEOPLE.length - 1)];
    const city = PLACES[rng.int(0, PLACES.length - 1)];
    const [itemA, itemB] = rng.sample(ITEMS, 2);
    const a = rng.int(2, 19), b = rng.int(2, 19);
    const style = STYLES[rng.int(0, STYLES.length - 1)];
    const question = rng.sample(['total', 'name', 'city'], 1)[0];
    const turns = [
      { role: 'user', content: `Hi, I'm ${name}. I'm setting up a small studio in ${city}.` },
      { role: 'assistant', content: `Nice to meet you, ${name}. A studio in ${city} sounds like a great project. How can I help?` },
      { role: 'user', content: `${style.ask} I have ${a} ${itemA} already and I just ordered ${b} ${itemB}.` },
      { role: 'assistant', content: 'Noted.' },
    ];
    let ask, fact, answer;
    if (question === 'total') {
      ask = `How many ${itemA} and ${itemB} will I have in total once the order arrives?`;
      fact = String(a + b);
      answer = `You will have ${a + b} in total: ${a} ${itemA} and ${b} ${itemB}.`;
    } else if (question === 'name') {
      ask = 'Quick check that you were listening: what is my name?';
      fact = name;
      answer = `Your name is ${name}.`;
    } else {
      ask = 'Remind me which city my studio is in?';
      fact = city;
      answer = `Your studio is in ${city}.`;
    }
    turns.push({ role: 'user', content: ask });
    const constraints = [style.constraint, { kind: 'include_words', words: [fact] }];
    let text = answer;
    if (style.constraint.kind === 'all_lowercase') text = text.toLowerCase();
    if (style.constraint.kind === 'starts_with') text = `Sure: ${text}`;
    if (style.constraint.kind === 'no_commas') text = text.replace(/,/g, '');
    if (style.constraint.kind === 'sentence_count') text = text.replace(/:/, '.');
    // all_lowercase lowercases the fact too; the include check is case-insensitive.
    if (!checkConstraints(text, constraints).passed) continue;
    const transcript = turns.map(t => `${t.role === 'user' ? 'User' : 'Assistant'}: ${t.content}`).join('\n');
    const record = curriculumCase({ family: 'chat_reply', shape: `chat${index}`, variant: `${question}-${style.constraint.kind}`,
      slice: 'writing', domain: 'writing', mode: 'single_call', inline: 'optional',
      evidence: { world: [], retrieved: [], background: [] }, plausibleActions: [],
      minimumSequence: ['read the whole conversation', 'answer the last message from what was said earlier',
        'keep every preference the user stated'],
      reference: { root: [returnCall(text)] },
      root: { name: 'reply_to_chat', args: { conversation: 'string' }, returns: 'string',
        instructions: 'Write the assistant\'s next reply to this conversation. Answer the last user message using what was said earlier, and follow every preference the user has stated about how you reply. Return only the reply text.' },
      inputs: { conversation: transcript }, expected: constraints });
    record.semantics.oracle = { level: 'constraints' };
    return [record];
  }
  throw new Error(`no consistent chat case for ${seed}:${index}`);
}
