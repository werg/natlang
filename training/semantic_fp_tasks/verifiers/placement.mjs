// Trusted verifier (ts-host/src/teacher/trusted-exact-verifier.ts): the folder a directory-reducer run left, judged file
// by file against where each file's meaning allows it to go. Any placement meeting every file's options passes.
// oracle: { files: { <starting path>: [<allowed place>, ...] }   a place is a directory ("keep/"), a full path
//                                                                ("archive/2026/r-12.md") or "removed" (the file is gone)
//           written?: { <path>: { includes?: [<text>, ...], excludes?: [<text>, ...], lines?: number } }
//                     files the run must write (a summary, a ledger), checked by exact pieces of text it must or must not
//                     contain and its exact number of non-empty lines; nothing semantic is judged from free text
//           others?: 'forbid' (default) | 'allow'    any file not starting there and not in written fails the run
//           value?: the returned value, compared exactly after canonicalisation (strings trimmed and lowercased) }
// A directory place keeps the file's name. A moved or kept file must keep its starting text (trailing space ignored).
// Paths compare exactly. score = (files placed acceptably that had to move + written files right + value right) /
// (files that had to move + written files + strays + (value ? 1 : 0)); a file allowed to stay that stays earns nothing.
const text = value => String(value ?? '').replace(/\s+$/, '');
const norm = value => typeof value === 'string' ? value.trim().replace(/\s+/g, ' ').toLowerCase() : value;
function canon(value) {
  if (Array.isArray(value)) return value.map(canon);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(key => [key, canon(value[key])]));
  return norm(value);
}
const name = path => path.split('/').at(-1);
const placeOf = (place, start) => place === 'removed' ? null : place.endsWith('/') ? place + name(start) : place;

export function verifyCandidate({ value, oracle, fixture, files }) {
  if (!files || typeof files !== 'object') return { ok: false, issues: ['the run left no folder'], score: 0 };
  const issues = [];
  const left = new Map(Object.entries(files).map(([path, body]) => [path, text(body)]));
  const claimed = new Set();
  let right = 0, counted = 0;
  for (const [start, places] of Object.entries(oracle.files)) {
    const body = text(fixture?.[start]);
    const targets = places.map(place => placeOf(place, start));
    const mustMove = !targets.includes(start);
    const at = targets.filter(target => target !== null && left.get(target) === body);
    const removedOk = targets.includes(null) && ![...left.values()].includes(body);
    const ok = at.length === 1 || (at.length === 0 && removedOk);
    if (mustMove) counted += 1;
    if (ok) { if (mustMove) right += 1; for (const target of at) claimed.add(target); }
    else issues.push(`${start}: should be at ${places.join(' or ')}`);
    if (at.length === 0 && !removedOk && !mustMove && left.get(start) !== undefined && left.get(start) !== body)
      issues.push(`${start}: its text changed`);
  }
  for (const [path, spec] of Object.entries(oracle.written ?? {})) {
    counted += 1;
    const have = left.get(path);
    claimed.add(path);
    if (have === undefined) { issues.push(`${path}: not written`); continue; }
    const problems = [];
    for (const piece of spec.includes ?? []) if (!have.includes(piece)) problems.push(`lacks ${JSON.stringify(piece)}`);
    for (const piece of spec.excludes ?? []) if (have.includes(piece)) problems.push(`must not contain ${JSON.stringify(piece)}`);
    if (spec.lines !== undefined && have.split('\n').filter(line => line.trim()).length !== spec.lines)
      problems.push(`should have ${spec.lines} non-empty lines`);
    if (problems.length) issues.push(`${path}: ${problems.join(', ')}`); else right += 1;
  }
  let strays = 0;
  if ((oracle.others ?? 'forbid') === 'forbid') {
    const starts = new Set(Object.keys(oracle.files));
    for (const path of left.keys()) if (!claimed.has(path) && !(starts.has(path) && oracle.files[path].some(place => placeOf(place, path) === path))) {
      if (starts.has(path) && left.get(path) === text(fixture?.[path]) && issues.some(issue => issue.startsWith(path + ':'))) continue;
      strays += 1; issues.push(`unexpected file: ${path}`);
    }
  }
  if (oracle.value !== undefined) {
    counted += 1;
    if (JSON.stringify(canon(value)) === JSON.stringify(canon(oracle.value))) right += 1;
    else issues.push(`returned ${JSON.stringify(value)}`);
  }
  const score = counted + strays ? right / (counted + strays) : 1;
  return { ok: issues.length === 0, issues, score: issues.length ? Math.min(score, 0.999) : 1 };
}
