/**
 * Find the editable prose in a hand-written HTML page and its exact source ranges, so an edit made in the browser can
 * be written back into the file without re-serializing the rest of the document. The page's scripts rewrite much of
 * its DOM at run time, so only the inner HTML of these elements is ever replaced.
 *
 * A target is the outermost element that matches the selection rules, has no `id` (ids mark elements a script fills
 * in), contains no element with an `id`, has visible text, and is not inside an excluded element.
 */

const VOID = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'source', 'track', 'wbr']);
const RAW_TEXT = new Set(['script', 'style', 'textarea', 'title']);
const EXCLUDED = new Set(['pre', 'svg', 'button', 'select', 'noscript', 'template', 'nav']);

export const DEFAULT_RULES = {
  tags: ['h1', 'h2', 'h3', 'h4', 'p', 'li', 'figcaption', 'blockquote', 'dt', 'dd'],
  classes: ['plate-title', 'label'],
  childrenOf: ['facts', 'panel-h', 'exercise'],
};

const TOKEN = /<!--[\s\S]*?-->|<!doctype[^>]*>|<\/([a-zA-Z][\w:-]*)\s*>|<([a-zA-Z][\w:-]*)((?:\s+[^\s"'>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'>]+))?)*)\s*(\/?)>/gi;
const ATTR = /([^\s"'>\/=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+)))?/g;

function parseAttrs(text) {
  const attrs = {};
  for (const m of text.matchAll(ATTR)) attrs[m[1].toLowerCase()] = m[2] ?? m[3] ?? m[4] ?? '';
  return attrs;
}

const classesOf = attrs => (attrs.class || '').split(/\s+/).filter(Boolean);
const textOf = html => html.replace(/<[^>]*>/g, '').replace(/&nbsp;|&#160;/g, ' ').trim();

/** Walk the document and return the elements as { tag, attrs, openStart, openEnd, innerStart, innerEnd, parent }. */
export function scanElements(html) {
  const out = [];
  const stack = [];
  TOKEN.lastIndex = 0;
  let m;
  while ((m = TOKEN.exec(html))) {
    if (m[0].startsWith('<!')) continue;
    if (m[1]) {
      const name = m[1].toLowerCase();
      let k = stack.length - 1;
      while (k >= 0 && stack[k].tag !== name) k--;
      if (k < 0) continue; // a stray close tag: ignore it, as browsers do
      while (stack.length > k) {
        const el = stack.pop();
        el.innerEnd = m.index;
        el.closeEnd = TOKEN.lastIndex;
      }
      continue;
    }
    const tag = m[2].toLowerCase();
    const el = { tag, attrs: parseAttrs(m[3] || ''), openStart: m.index, openEnd: TOKEN.lastIndex, innerStart: TOKEN.lastIndex,
      innerEnd: TOKEN.lastIndex, parent: stack[stack.length - 1] || null };
    out.push(el);
    if (RAW_TEXT.has(tag)) {
      const close = html.toLowerCase().indexOf(`</${tag}`, TOKEN.lastIndex);
      el.innerEnd = close < 0 ? html.length : close;
      const gt = close < 0 ? html.length : html.indexOf('>', close) + 1;
      el.closeEnd = gt;
      TOKEN.lastIndex = gt;
      continue;
    }
    if (VOID.has(tag) || m[4] === '/') { el.closeEnd = el.openEnd; continue; }
    stack.push(el);
  }
  return out;
}

function isExcluded(el) {
  for (let p = el; p; p = p.parent) {
    if (EXCLUDED.has(p.tag) || 'data-no-edit' in p.attrs) return true;
    if (p.tag === 'header' && classesOf(p.attrs).includes('bar')) return true;
  }
  return false;
}

function matches(el, rules) {
  if (rules.tags.includes(el.tag)) return true;
  const cls = classesOf(el.attrs);
  if (cls.some(c => rules.classes.includes(c))) return true;
  if (el.parent && classesOf(el.parent.attrs).some(c => rules.childrenOf.includes(c))) return true;
  return 'data-edit' in el.attrs;
}

/**
 * The editable elements of `html`, in document order. Each has `id` (its position in this list), `tag`, the source
 * range of its opening tag (`openStart`, `openEnd`) and of its inner HTML (`innerStart`, `innerEnd`).
 */
export function findEditableTargets(html, rules = DEFAULT_RULES) {
  const els = scanElements(html);
  const hasIdInside = el => els.some(o => o !== el && 'id' in o.attrs && o.openStart >= el.innerStart && o.openStart < el.innerEnd);
  const chosen = [];
  for (const el of els) {
    if (!matches(el, rules) || 'id' in el.attrs || isExcluded(el)) continue;
    if (chosen.some(c => el.openStart >= c.innerStart && el.openStart < c.innerEnd)) continue; // outermost wins
    if (!textOf(html.slice(el.innerStart, el.innerEnd)) || hasIdInside(el)) continue;
    chosen.push(el);
  }
  return chosen.map((el, id) => ({ id, tag: el.tag, openStart: el.openStart, openEnd: el.openEnd, innerStart: el.innerStart, innerEnd: el.innerEnd }));
}

/** The page with `data-nl-edit="<id>"` added to the opening tag of every target. */
export function annotate(html, targets) {
  let out = html;
  for (const t of [...targets].sort((a, b) => b.openEnd - a.openEnd)) {
    const close = out[t.openEnd - 2] === '/' ? t.openEnd - 2 : t.openEnd - 1;
    out = out.slice(0, close) + ` data-nl-edit="${t.id}"` + out.slice(close);
  }
  return out;
}

/**
 * Why `fragment` cannot replace an element's inner HTML, or null when it can: its tags must balance, and it may not
 * contain scripts, styles, frames or event-handler attributes.
 */
export function fragmentProblem(fragment) {
  if (/<\s*(script|style|iframe|object|embed|link|meta)\b/i.test(fragment)) return 'contains a script, style or embedded document';
  if (/<[^>]*\son\w+\s*=/i.test(fragment)) return 'contains an event-handler attribute';
  const stack = [];
  TOKEN.lastIndex = 0;
  let m;
  while ((m = TOKEN.exec(fragment))) {
    if (m[0].startsWith('<!')) continue;
    if (m[1]) {
      const name = m[1].toLowerCase();
      if (stack.pop() !== name) return `has an unmatched </${name}>`;
    } else if (!VOID.has(m[2].toLowerCase()) && m[4] !== '/') stack.push(m[2].toLowerCase());
  }
  return stack.length ? `leaves <${stack[stack.length - 1]}> open` : null;
}

/** Apply `edits` ([{ id, html }]) to the source; throws when an id is unknown or a fragment is unsafe. */
export function applyEdits(html, edits, rules = DEFAULT_RULES) {
  const targets = findEditableTargets(html, rules);
  const byId = new Map(targets.map(t => [t.id, t]));
  const seen = new Set();
  const planned = edits.map(e => {
    const t = byId.get(Number(e.id));
    if (!t) throw new Error(`no editable element ${e.id}`);
    if (seen.has(t.id)) throw new Error(`element ${e.id} is edited twice`);
    seen.add(t.id);
    if (typeof e.html !== 'string') throw new Error(`element ${e.id} has no html`);
    const problem = fragmentProblem(e.html);
    if (problem) throw new Error(`the new text of element ${e.id} ${problem}`);
    return { t, html: e.html };
  });
  let out = html;
  for (const { t, html: inner } of planned.sort((a, b) => b.t.innerStart - a.t.innerStart)) {
    out = out.slice(0, t.innerStart) + inner + out.slice(t.innerEnd);
  }
  return out;
}
