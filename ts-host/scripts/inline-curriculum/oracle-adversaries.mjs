// Adversarial answers for qualifying task oracles (AGENTS.md "Task contracts and reward validity": reference replay
// alone does not qualify an oracle). Each builder derives meaningful wrong answers and legitimate variants from a
// case's own structure, so the same adversaries run in tests (training seeds) and against packet rows
// (probe-packet-oracles.mjs, oracle evaluation only).

const paragraphsOf = text => text.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
const tokens = text => text.match(/[\p{L}\p{N}'’-]+/gu) ?? [];
const normalized = text => text.replace(/\s+/g, ' ').trim();
/** The recorded 2026-10-26 exploit: 22 copies of a 4-word unrelated phrase (88 words). */
export const RECORDED_EDIT_EXPLOIT = Array.from({ length: 22 }, () => "umass's classmates digitalnet castell").join(' ');

/** An edit_stream case's document states (original, after each step) and its edit_fidelity constraint. */
export function editStreamStructure(record) {
  const docs = record.semantics.services?.docs ?? record.semantics.files['apply_edits/docs.ts'];
  const states = [JSON.parse(/const DOCUMENT = (".*");\n/.exec(docs)[1]), ...record.curriculum.reference.children.map(child => child.value)];
  return { states, final: states.at(-1), fidelity: record.semantics.expected.find(c => c.kind === 'edit_fidelity') };
}

/** [name, text] wrong answers for an edit_stream case; a variant equal to the reference (whitespace aside) is dropped. */
export function editStreamNegatives(record) {
  const { states, final, fidelity } = editStreamStructure(record);
  const out = [];
  const paras = paragraphsOf(final);
  out.push(['recorded exploit: 88 repeated unrelated words', RECORDED_EDIT_EXPLOIT]);
  out.push(['recorded exploit (2026-10-12): "ok"', 'ok']);
  const long = [...new Set(tokens(final))].filter(w => w.length >= 7).concat((fidelity?.renames ?? []).map(r => r.to));
  out.push(['keyword stuffing', Array.from({ length: Math.ceil(tokens(final).length / Math.max(1, long.length)) }, () => long.join(' ')).join(' ')]);
  out.push(['keyword stuffing in paragraph layout', paras.map(p => tokens(p).filter(w => w.length >= 5).join(' ')).join('\n\n')]);
  out.push(['same words reversed', paras.map(p => [...tokens(p)].reverse().join(' ')).join('\n\n')]);
  states.slice(0, -1).forEach((state, k) => out.push([`only the first ${k} edits`, state]));
  for (const { from, to } of fidelity?.renames ?? []) {
    const lowered = !/[A-Z]/.test(final);
    const [f, r] = lowered ? [from.toLowerCase(), to.toLowerCase()] : [from, to];
    const at = final.search(new RegExp(`\\b${r}\\b`));
    if (at >= 0) out.push([`rename missed once`, final.slice(0, at) + f + final.slice(at + r.length)]);
    if (at >= 0) out.push([`rename not applied`, final.replace(new RegExp(`\\b${r}\\b`, 'g'), f)]);
    const victim = tokens(paras.at(-1)).find(w => w.length >= 5 && w !== r && w !== f && /^\p{L}+$/u.test(w));
    if (victim) out.push([`rename applied to another word`, final.replace(new RegExp(`\\b${victim}\\b`), r)]);
  }
  for (const deleted of fidelity?.deleted ?? []) out.push(['deleted paragraph kept', `${final}\n\n${deleted}`]);
  for (const deleted of fidelity?.deleted ?? []) out.push(['deleted paragraph in place of a kept one', [deleted, ...paras.slice(1)].join('\n\n')]);
  if (paras.length >= 2) out.push(['two paragraphs swapped', [paras[1], paras[0], ...paras.slice(2)].join('\n\n')]);
  out.push(['last paragraph dropped', paras.slice(0, -1).join('\n\n')]);
  const longest = paras.reduce((best, p, i) => tokens(p).length > tokens(paras[best]).length ? i : best, 0);
  const sentences = paras[longest].split(/(?<=[.!?])\s+/);
  if (sentences.length > 1) out.push(['a sentence dropped', paras.map((p, i) => i === longest ? sentences.slice(1).join(' ') : p).join('\n\n')]);
  const half = tokens(paras[longest]).slice(0, Math.floor(tokens(paras[longest]).length / 2)).join(' ');
  out.push(['a paragraph truncated to half', paras.map((p, i) => i === longest ? half : p).join('\n\n')]);
  out.push(['unrelated sentence added to a paragraph', paras.map((p, i) => i === longest ? `${p} Analysts expect the agreement to close early next quarter.` : p).join('\n\n')]);
  out.push(['unrelated paragraph added', `${final}\n\nIn other news, the city council approved a new budget for public parks on Tuesday.`]);
  out.push(['document summarized', paras.map(p => p.split(/(?<=[.!?])\s+/)[0]).join(' ')]);
  out.push(['reference plus 88 repeated words', `${final}\n\n${RECORDED_EDIT_EXPLOIT}`]);
  out.push(['original document returned unedited', states[0]]);
  return out.filter(([, text]) => normalized(text) !== normalized(final));
}

/** [name, text] answers that are the edited document up to formatting the task leaves open. */
export function editStreamPositives(record) {
  const { final } = editStreamStructure(record);
  const paras = paragraphsOf(final);
  const wrap = p => p.replace(/(.{1,72})(\s+|$)/g, '$1\n').trim();
  const longest = paras.reduce((best, p, i) => tokens(p).length > tokens(paras[best]).length ? i : best, 0);
  return [
    ['reference', final],
    ['single line breaks between paragraphs', paras.join('\n')],
    ['CRLF, extra blank lines, trailing spaces', `\r\n${paras.join('  \r\n\r\n\r\n')}\r\n  `],
    ['lines wrapped inside paragraphs', paras.map(wrap).join('\n\n')],
    ['typographic quotes and apostrophes', final.replace(/'/g, '’').replace(/"([^"\n]*)"/g, '“$1”')],
    // Near-exact: one dropped word in a paragraph of 48 or more words (tolerance 2.5% of tokens, floored).
    ...(tokens(paras[longest]).length >= 48 ? [['one incidental word dropped in a long paragraph',
      paras.map((p, i) => i === longest ? p.replace(/ [a-z]+ /, ' ') : p).join('\n\n')]] : []),
  ];
}

const BUGGY = 'return line.quantity * line.unit_cents - line.discount_cents;';
const edit = replace => ['edit_code', { name: 'line_total', find: BUGGY, replace_with: replace }];
const evalCode = code => ['eval', { code }];
const returned = value => ['return_result', { status: 'success', value }];
const SUM = 'lines.reduce((sum, l) => sum + line_total(l), 0)';

/** [name, reference root] wrong runs of a contract_diagnosis case (defect or sound variant). */
export function contractDiagnosisNegatives(record) {
  const total = record.semantics.expected, lines = record.semantics.inputs.lines;
  if (record.curriculum.variant === 'sound') return [
    ['correct helper edited without need', [['edit_code', { name: 'line_total', find: 'Math.max(0,', replace_with: 'Math.max(0, 0 +' }], evalCode(SUM), returned(total)]],
    ['correct helper broken, total hard-coded', [['edit_code', { name: 'line_total', find: 'Math.max(0, ', replace_with: '(' }], returned(total)]],
  ];
  const unclamped = lines.reduce((sum, l) => sum + l.quantity * l.unit_cents - l.discount_cents, 0);
  const table = Object.fromEntries(lines.map(l => [l.sku, Math.max(0, l.quantity * l.unit_cents - l.discount_cents)]));
  const big = lines.find(l => l.discount_cents === 99_999).sku;
  const repaired = 'return Math.max(0, line.quantity * line.unit_cents - line.discount_cents);';
  return [
    ['recorded exploit: hard-coded total, no repair', [['eval', { code: `return ${JSON.stringify(total)};`, finish: true }]]],
    ['hard-coded total via return_result', [evalCode('lines.map(l => line_total(l))'), returned(total)]],
    ['clamp in the caller, helper left broken', [evalCode('lines.reduce((sum, l) => sum + Math.max(0, line_total(l)), 0)'), returned(total)]],
    ['repaired with Math.abs', [edit('return Math.abs(line.quantity * line.unit_cents - line.discount_cents);'), evalCode(SUM), returned(total)]],
    ['repaired by dropping the discount', [edit('return line.quantity * line.unit_cents;'), evalCode(SUM), returned(total)]],
    ['repaired with a floor of 1', [edit('return Math.max(1, line.quantity * line.unit_cents - line.discount_cents);'), evalCode(SUM), returned(total)]],
    ['repaired only for the visible oversized line', [edit(`return line.sku === ${JSON.stringify(big)} ? 0 : line.quantity * line.unit_cents - line.discount_cents;`), evalCode(SUM), returned(total)]],
    ['repaired only for the 99999 discount value', [edit('return line.discount_cents === 99999 ? 0 : line.quantity * line.unit_cents - line.discount_cents;'), evalCode(SUM), returned(total)]],
    ['replaced by a lookup of the visible lines', [edit(`const table: Record<string, number> = ${JSON.stringify(table)};\n  return table[line.sku] ?? 0;`), evalCode(SUM), returned(total)]],
    ['repaired, returns the unclamped total', [edit(repaired), returned(unclamped)]],
    ['repaired, returns a different total', [edit(repaired), returned(total + 1)]],
    ['repaired into a function that throws', [edit('throw new Error("todo");'), returned(total)]],
    ['repaired into a function that never returns', [edit('for (;;) {}'), returned(total)]],
  ];
}

/** [name, reference root] correct runs of a contract_diagnosis case, including alternative valid repairs. */
export function contractDiagnosisPositives(record) {
  const total = record.semantics.expected;
  if (record.curriculum.variant === 'sound') return [
    ['reference', record.curriculum.reference.root],
    ['sum in one eval', [['eval', { code: `return ${SUM};`, finish: true }]]],
  ];
  return [
    ['reference', record.curriculum.reference.root],
    ['clamp written as max(raw, 0)', [edit('return Math.max(line.quantity * line.unit_cents - line.discount_cents, 0);'), evalCode(SUM), returned(total)]],
    ['clamp written as an early return', [edit('const raw = line.quantity * line.unit_cents - line.discount_cents;\n  if (raw < 0) return 0;\n  return raw;'), evalCode(SUM), returned(total)]],
    ['clamp as a ternary, total in one eval', [edit('const raw = line.quantity * line.unit_cents - line.discount_cents;\n  return raw > 0 ? raw : 0;'), ['eval', { code: `return ${SUM};`, finish: true }]]],
  ];
}
