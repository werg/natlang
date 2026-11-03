import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { checkCodebase, positionalParameters } from '../dist/teacher/codebase-check.js';

const ROOT = new URL('../../codebases/design_briefs/017-functional-substitution-sourcing-desk', import.meta.url).pathname;
const sha = text => createHash('sha256').update(text).digest('hex');
function packageFiles(dir = ROOT, out = {}) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) packageFiles(path, out);
    else if (/\.(nl|ts)$/.test(name) && !/\.test\.|\.d\.ts$/.test(name) || name === 'natlang.json') out[relative(ROOT, path)] = readFileSync(path, 'utf8');
  }
  return out;
}

test('a recreated function is held to the types its callers pass by position, not to unstated names (H2)', () => {
  const original = packageFiles();
  const path = 'sourcingDesk/scopeUseApproval.nl';
  const declared = /^args: (.*)$/m.exec(original[path])[1], returns = /^returns: (.*)$/m.exec(original[path])[1];
  assert.deepEqual(positionalParameters(declared), [[false, 'SubstituteEvaluationRequest'], [false, 'EvidencePage[]']]);
  const folder = { ...original }; delete folder[path];
  const spec = { load: true, require: [path], allow_new_prefixes: [path],
    preserve: Object.fromEntries(Object.entries(original).filter(([name]) => name !== path).map(([name, text]) => [name, sha(text)])),
    signatures: [{ path, args: declared, returns }] };
  const write = (args, ret = returns) => ({ ...folder, [path]: `---\nargs: ${args}\nreturns: ${ret}\n---\nBind each approval.\n` });
  const check = files => checkCodebase(files, spec, folder);
  // Reference: the original function, and the same parameters under other names (the failing samples' `useApprovals`).
  assert.deepEqual(check({ ...folder, [path]: original[path] }).problems, []);
  assert.deepEqual(check(write('{ request: SubstituteEvaluationRequest, useApprovals: EvidencePage[] }')).problems, []);
  assert.deepEqual(check(write('{request:SubstituteEvaluationRequest,pages:EvidencePage[]}')).problems, []);
  // Negatives: the whole container as the second parameter, the wrong order, a missing or extra parameter, a wrong result.
  assert.match(check(write('{ request: SubstituteEvaluationRequest, catalog: SupplierCatalogSnapshot }')).problems.join(), /types, in order/);
  assert.match(check(write('{ approvals: EvidencePage[], request: SubstituteEvaluationRequest }')).problems.join(), /types, in order/);
  assert.match(check(write('{ request: SubstituteEvaluationRequest }')).problems.join(), /types, in order/);
  assert.match(check(write('{ request: SubstituteEvaluationRequest, approvals: EvidencePage[], extra: string }')).problems.join(), /types, in order/);
  assert.match(check(write('{ request: SubstituteEvaluationRequest, approvals?: EvidencePage[] }')).problems.join(), /types, in order/);
  assert.match(check(write(declared, 'string')).problems.join(), /returns differs/);
  assert.match(check(folder).problems.join(), /is missing/);
});

test('a function repaired in place keeps its declared parameter names', () => {
  const original = packageFiles();
  const path = 'sourcingDesk/scopeUseApproval.nl';
  const declared = /^args: (.*)$/m.exec(original[path])[1];
  const spec = { load: true, require: [path], signatures: [{ path, args: declared }] };
  const renamed = { ...original, [path]: original[path].replace('approvals:', 'useApprovals:') };
  assert.match(checkCodebase(renamed, spec, original).problems.join(), /args differ/);
  assert.deepEqual(checkCodebase(original, spec, original).problems, []);
});
