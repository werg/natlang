/** Runtime verdict for eval code, one JSON request per line: the TypeScript scope compiler's own diagnostics. */
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
import { compileScopeSnippet } from '../../ts-host/dist/scope-compiler.js';

export function evalCodeVerdict(code) {
  if (typeof code !== 'string' || code.length > 4 * 1024 * 1024) throw new TypeError('expected bounded eval code');
  let result;
  try { result = compileScopeSnippet(code, { allowModules: true }); }
  catch (error) { return { ok: false, diagnostics: [{ code: 'compiler-exception', message: String(error?.message ?? error).slice(0, 300) }] }; }
  return { ok: !!result.ok, diagnostics: (result.diagnostics ?? []).slice(0, 3).map(d => ({ code: d.code, message: String(d.message).slice(0, 300) })) };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
    if (!line.trim()) continue;
    process.stdout.write(JSON.stringify(evalCodeVerdict(JSON.parse(line).code)) + '\n');
  }
}
