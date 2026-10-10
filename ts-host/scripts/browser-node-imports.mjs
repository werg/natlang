// Node built-ins in the browser bundles. A browser build may reach a `node:*` module only through a rule below, which
// names the stand-in it gets and why that is safe; any other `node:*` import reachable from a browser entry point fails
// the build, naming the chain of modules that reaches it. Node-only code stays out of the browser graph (behind a Node
// entry point or an injected hook) rather than being stubbed into code that breaks when it runs.
import { relative, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const unavailable = specifier => `throw new Error(${JSON.stringify(`${specifier} is not available in a browser`)});`;

/**
 * Tolerated imports: `specifiers`, optionally only from importers matching `importer` and only of `kind` (an esbuild
 * import kind), get `contents(specifier)` as their module.
 */
export const TOLERATED_NODE_IMPORTS = [
  { specifiers: ['node:zlib'], reason: 'gzip for archives and .nz files; fflate implements the two functions the shared code calls',
    contents: () => 'import { gunzipSync, gzipSync } from "fflate"; export { gunzipSync, gzipSync }; export const constants = {}; ' +
      'export default { gunzipSync, gzipSync, constants };' },
  { specifiers: ['node:fs'], importer: /\/@earendil-works\/pi-ai\/dist\/utils\/provider-env\.js$/, kind: 'require-call',
    reason: 'pi-ai reads /proc/self/environ only under Bun (process.versions.bun), when process.env is empty; a browser has no process, so the require never runs (if it did, it throws)',
    contents: unavailable },
  { specifiers: ['node:child_process', 'node:crypto', 'node:fs', 'node:fs/promises', 'node:path', 'node:url', 'node:vm'],
    importer: /\/node_modules\/pyodide\//, kind: 'dynamic-import',
    reason: 'Pyodide imports its Node loader only after detecting Node; in a browser these imports never run (if one did, it throws)',
    contents: unavailable },
];

const matches = (rule, args) => rule.specifiers.includes(args.path) && (!rule.kind || rule.kind === args.kind) &&
  (!rule.importer || rule.importer.test(args.importer.split('\\').join('/')));

/**
 * The esbuild plugin enforcing the policy (`tolerated` defaults to TOLERATED_NODE_IMPORTS). Forbidden imports resolve
 * to an empty placeholder so the build can finish and report every offending chain at once; then it fails.
 */
export function browserNodeImports({ tolerated = TOLERATED_NODE_IMPORTS } = {}) {
  return { name: 'browser-node-imports', setup(build) {
    build.initialOptions.metafile = true;
    const workingDir = build.initialOptions.absWorkingDir ?? process.cwd();
    const forbidden = new Set();
    build.onStart(() => { forbidden.clear(); });
    build.onResolve({ filter: /^node:/ }, args => {
      const rule = tolerated.findIndex(candidate => matches(candidate, args));
      if (rule >= 0) return { path: args.path, namespace: 'browser-node-tolerated', pluginData: { rule } };
      forbidden.add(args.path);
      return { path: args.path, namespace: 'browser-node-forbidden' };
    });
    build.onLoad({ filter: /.*/, namespace: 'browser-node-tolerated' }, args =>
      ({ contents: tolerated[args.pluginData.rule].contents(args.path), loader: 'js', resolveDir: root }));
    build.onLoad({ filter: /.*/, namespace: 'browser-node-forbidden' }, () => ({ contents: 'module.exports = {};', loader: 'js' }));
    build.onEnd(result => {
      if (!forbidden.size || !result.metafile) return;
      const entries = entryPoints(build.initialOptions.entryPoints).map(path => relative(workingDir, resolve(workingDir, path)).split('\\').join('/'));
      const found = [...forbidden].sort().map(specifier => `${specifier}, reached by\n` +
        importChains(result.metafile, `browser-node-forbidden:${specifier}`, entries).map(chain => '  ' + [...chain, specifier].join('\n    -> ')).join('\n'));
      return { errors: [{ text: `Node built-ins reachable from the browser bundle (${entries.join(', ')}):\n${found.join('\n')}\n` +
        'Keep the Node-only module out of the browser graph (import it from a Node entry point such as src/runtime/node.ts, ' +
        'or inject the Node behavior through a hook), or, when a browser stand-in is genuinely correct, add a rule to ' +
        'TOLERATED_NODE_IMPORTS in scripts/browser-node-imports.mjs.' }] };
    });
  } };
}

const entryPoints = option => Array.isArray(option) ? option.map(entry => typeof entry === 'string' ? entry : entry.in)
  : Object.values(option ?? {});

/** The shortest import chain from `entries` to each module that imports `target` (metafile input keys). */
export function importChains(metafile, target, entries) {
  const previous = new Map(entries.map(entry => [entry, undefined]));
  const queue = [...entries];
  while (queue.length) {
    const at = queue.shift();
    for (const edge of metafile.inputs[at]?.imports ?? []) {
      if (previous.has(edge.path) || edge.external) continue;
      previous.set(edge.path, { from: at, kind: edge.kind });
      queue.push(edge.path);
    }
  }
  const kindNote = kind => kind === 'dynamic-import' ? ' (dynamic import)' : '';
  return Object.entries(metafile.inputs).filter(([path, input]) => previous.has(path) && input.imports.some(edge => edge.path === target))
    .map(([importer, input]) => {
      const chain = [`${importer}${kindNote(input.imports.find(edge => edge.path === target).kind)}`];
      for (let at = importer, step = previous.get(at); step; at = step.from, step = previous.get(at))
        chain.unshift(step.from + kindNote(step.kind));
      return chain;
    });
}
