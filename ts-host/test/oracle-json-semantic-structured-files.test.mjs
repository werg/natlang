import test from 'node:test';
import assert from 'node:assert/strict';
import { checkFiles, checkFilesWithJudge } from '../dist/teacher/oracle.js';

const spec = {
  compare: 'json-semantic-structured',
  json_report_paths: ['reports/ledger.json'],
  rationale_context_paths: ['source/clauses.json'],
  comparison_spec: {
    schema_version: 'semantic-structured-comparison/1',
    unordered_paths: [],
    ordered_paths: [],
    prose_paths: ['/findings/*/explanation'],
    prose_policy: 'Judge each explanation against the supplied clause source. Wording may vary; do not waive any mechanically checked report field.',
  },
  rubric: 'Compare the report using exact structural fields and source-grounded prose.',
};
const input = { 'source/clauses.json': '{"6":"All information, technical and non-technical, is confidential."}\n' };
const expectedReport = {
  schema: 'natlang.contract-obligation-review/2', agreementId: '146', sourceDigest: 'abc',
  findings: [{ hypothesisId: 'nda-2', choice: 'Contradiction', evidenceSpanIds: ['6'],
    explanation: 'The agreement includes non-technical information.' }],
  counts: { Entailment: 0, Contradiction: 1, NotMentioned: 0 },
  sharedEvidence: [],
};
const actualReport = (explanation = 'Span 6 includes non-technical material, so the “only technical” hypothesis is contradicted.') =>
  ({ ...expectedReport, findings: [{ ...expectedReport.findings[0], explanation }] });
const file = report => JSON.stringify(report) + '\n';

test('semantic JSON file oracle holds every non-prose field exact and judges prose with bound source', async () => {
  const actual = { 'reports/ledger.json': file(actualReport()) };
  const expected = { 'reports/ledger.json': file(expectedReport) };
  assert.equal(checkFiles(actual, expected, input, spec).accepted, false);
  assert.deepEqual(checkFiles(actual, expected, input, spec).pending, ['unjudged_json_rationale:reports/ledger.json']);
  let called = 0;
  const verdict = await checkFilesWithJudge(actual, expected, input, spec, async request => {
    called++;
    assert.deepEqual(request.actual.report, actualReport());
    assert.deepEqual(request.expected, { checked_non_prose_structure: true,
      prose_paths: ['/findings/*/explanation'] });
    assert.deepEqual(request.actual.source_files, input);
    assert.equal(request.actual.output_path, 'reports/ledger.json');
    return { accepted: true, verdict: 'The explanation accurately grounds the decision in span 6.' };
  });
  assert.equal(called, 1);
  assert.equal(verdict.accepted, true);
});

test('structural JSON mutations cannot be waived by the prose judge', async () => {
  const expected = { 'reports/ledger.json': file(expectedReport) };
  for (const report of [
    { ...actualReport(), extra: true },
    { ...actualReport(), findings: [{ ...actualReport().findings[0], hypothesisId: 'nda-3' }] },
    { ...actualReport(), findings: [{ ...actualReport().findings[0], choice: 'Entailment' }] },
    { ...actualReport(), findings: [{ ...actualReport().findings[0], evidenceSpanIds: ['10'] }] },
    { ...actualReport(), findings: [{ ...actualReport().findings[0], explanation: undefined }] },
  ]) {
    let called = 0;
    const result = await checkFilesWithJudge({ 'reports/ledger.json': file(report) }, expected, input, spec,
      async () => { called++; return { accepted: true, verdict: 'incorrectly permissive' }; });
    assert.equal(result.accepted, false);
    assert.equal(called, 0);
  }
});

test('an independent judge can reject unsupported prose and missing bound source fails closed', async () => {
  const expected = { 'reports/ledger.json': file(expectedReport) };
  const actual = { 'reports/ledger.json': file(actualReport('The cited span says the opposite, but I call it entailment.')) };
  assert.equal((await checkFilesWithJudge(actual, expected, input, spec,
    async () => ({ accepted: false, verdict: 'Explanation contradicts source.' }))).accepted, false);
  const noSource = await checkFilesWithJudge(actual, expected, {}, spec,
    async () => ({ accepted: true, verdict: 'cannot accept without source' }));
  assert.equal(noSource.accepted, false);
  assert.ok(noSource.errors.some(error => error === 'missing_rationale_context:source/clauses.json'));
});

test('duplicate JSON keys and malformed rationale specs fail closed', () => {
  const expected = { 'reports/ledger.json': file(expectedReport) };
  assert.equal(checkFiles({ 'reports/ledger.json': '{"schema":"x","schema":"y"}' }, expected, input, spec).accepted, false);
  assert.equal(checkFiles(expected, expected, input, { ...spec, json_report_paths: ['../ledger.json'] }).accepted, false);
  assert.equal(checkFiles(expected, expected, input, { ...spec, comparison_spec: { ...spec.comparison_spec,
    prose_paths: ['/findings/*/missing'] } }).accepted, false);
});
