// Argument-count check of root eval scripts against the declared arguments of the root's helpers
// (used by structure_smoke.mjs; the runtime drops extra arguments silently, so a scripted run alone would pass).
// Declared argument counts of the root's direct helpers: name -> [required, total].
export function helperArities(row, { readNatlangFrontmatter }) {
  const stem = row.semantics.root.replace(/\.nl$/, '') + '/', out = {};
  for (const [file, text] of Object.entries(row.semantics.files)) {
    if (!file.startsWith(stem) || !file.endsWith('.nl') || file.slice(stem.length).includes('/')) continue;
    const match = /^---\r?\n([\s\S]*?)\r?\n---/.exec(text);
    let args = [];
    try { args = Object.keys(readNatlangFrontmatter(match?.[1] ?? '').args ?? {}); } catch { continue; }
    out[file.slice(stem.length, -3)] = [args.filter(a => !a.endsWith('?')).length, args.length];
  }
  return out;
}
// Calls with the wrong number of arguments in a root eval script (parsed as an async function body).
export function arityProblems(code, arities, { ts }) {
  const file = ts.createSourceFile('script.ts', `async function __run() {\n${code}\n}`, ts.ScriptTarget.Latest, true);
  const problems = [];
  const visit = node => {
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression)) {
      let name = node.expression.text, count = node.arguments.length;
      if (name === 'decide' && node.arguments[0] && ts.isIdentifier(node.arguments[0])) { name = node.arguments[0].text; count -= 1; }
      const declared = arities[name];
      if (declared && (count < declared[0] || count > declared[1]))
        problems.push(`${name} is called with ${count} argument(s) but declares ${declared[0] === declared[1] ? declared[1] : declared[0] + ' to ' + declared[1]}`);
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return problems;
}
