import test from 'node:test';
import assert from 'node:assert/strict';
import ts from 'typescript';
import { buildResearchClassificationEpisodes, domains, metric } from '../scripts/skills/build-research-classification-episodes.mjs';
import { authorView, validateEpisode } from '../dist/skills/episode.js';
import { RESEARCH_OBJECTIVE_KIND, scoreResearchObjective } from '../src/skills/research-objective.ts';

const exactAnswer = expected => ({ label: expected.label, unresolved: expected.unresolved,
  citations: expected.requiredEvidence.map(row => ({ sourceId: row.sourceId, evidence: row.text })) });

test('fictional research packets use four held-out domains and host-only service implementations', () => {
  const episodes = buildResearchClassificationEpisodes();
  assert.equal(domains.length, 4);
  assert.equal(episodes.length, 4);
  for (const episode of episodes) {
    assert.deepEqual(validateEpisode(episode), [], episode.id);
    assert.equal(episode.support.cases.length, 2);
    assert.equal(episode.query.cases.length, 2);
    assert.ok(episode.support.cases.every(row => !episode.query.cases.some(query => query.group === row.group)));
    assert.equal(episode.provenance.metric.kind, RESEARCH_OBJECTIVE_KIND);
    const visible = JSON.stringify(authorView(episode));
    assert.ok(!visible.includes('research-classification-episodes/1'));
    assert.equal(authorView(episode).query, undefined);
    assert.ok(authorView(episode).support.cases.every(row=>row.services===undefined));
    assert.ok(!visible.includes('PRIVATE_DOCUMENTS'));
    for (const row of [...episode.support.cases, ...episode.query.cases]) {
      const packet = JSON.parse(row.args[0]);
      assert.ok(packet.disclaimer.includes('Fictional'));
      assert.ok(packet.question.length > 80);
      assert.ok(packet.catalog.length >= 3);
      assert.ok(packet.catalog.every(entry => !('text' in entry)));
      assert.ok(row.services.research.includes('PRIVATE_DOCUMENTS'));
      assert.ok(row.expected.documents.every(document => packet.catalog.some(entry => entry.id === document.id)));
      assert.ok(row.expected.requiredEvidence.length >= 2, 'gold requires joined-source evidence');
      assert.equal(row.expected.kind, 'research-classification');
      assert.equal(scoreResearchObjective(packet, exactAnswer(row.expected), row.expected).quality, 1, row.id);

      const declaration = ts.transpileDeclaration(row.services.research, { fileName: 'research.ts', compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
      assert.match(declaration, /search\(query: string\)/);
      assert.match(declaration, /read\(sourceId: string\)/);
      assert.ok(!declaration.includes('PRIVATE_DOCUMENTS'));
      for (const document of row.expected.documents) assert.ok(!declaration.includes(document.text), 'public declaration must not contain document bodies');
    }
  }
  assert.equal(metric.kind, 'research-classification');
});

test('retrieval service searches the private corpus and reads documents only by catalog ID', () => {
  const row = buildResearchClassificationEpisodes()[0].support.cases[0];
  const output = ts.transpileModule(row.services.research, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
  const exports = {};
  new Function('exports', output)(exports);
  const packet = JSON.parse(row.args[0]);
  const hits = exports.search('pressure release occupied staffed hours');
  assert.ok(hits.length >= 2);
  assert.ok(hits.every(hit => packet.catalog.some(item => item.id === hit.id)));
  const source = exports.read(hits[0].id);
  assert.ok(row.expected.documents.some(document => document.id === hits[0].id && document.text === source));
  assert.throws(() => exports.read('made-up-source'), /Unknown research source ID/);
});

test('research scorer requires exact real-source citations and explicit handling of contradictions', () => {
  const episode = buildResearchClassificationEpisodes().find(row => row.family.endsWith('incident-triage'));
  const clear = episode.support.cases[0], packet = JSON.parse(clear.args[0]);
  const perfect = scoreResearchObjective(packet, exactAnswer(clear.expected), clear.expected);
  assert.equal(perfect.quality, 1);
  assert.equal(perfect.gates.classification_correct, true);
  const wrongLabel = scoreResearchObjective(packet, { ...exactAnswer(clear.expected), label: 'monitor' }, clear.expected);
  assert.equal(wrongLabel.quality, 0.5);
  assert.equal(wrongLabel.gates.classification_correct, false);
  const partial = scoreResearchObjective(packet, { ...exactAnswer(clear.expected), citations: exactAnswer(clear.expected).citations.slice(0, 1) }, clear.expected);
  assert.equal(partial.quality, 0.5);
  assert.equal(partial.gates.required_evidence_supported, false);
  const firstRequired = clear.expected.requiredEvidence[0];
  const containingQuote = clear.expected.documents.find(document => document.id === firstRequired.sourceId).text;
  const longerVerifiedQuote = scoreResearchObjective(packet, { ...exactAnswer(clear.expected), citations: [
    { sourceId: firstRequired.sourceId, evidence: containingQuote }, ...exactAnswer(clear.expected).citations.slice(1),
  ] }, clear.expected);
  assert.equal(longerVerifiedQuote.quality, 1, 'a longer quote earns credit when it is fully present in the cited source');
  const whitespaceQuote = scoreResearchObjective(packet, { ...exactAnswer(clear.expected), citations: [
    { sourceId: firstRequired.sourceId, evidence: containingQuote.replace(/ /gu, '  ') }, ...exactAnswer(clear.expected).citations.slice(1),
  ] }, clear.expected);
  assert.equal(whitespaceQuote.quality, 1, 'whitespace changes are normalized while the full quote remains contiguous');
  const fabricated = scoreResearchObjective(packet, { ...exactAnswer(clear.expected), citations: [{ sourceId: clear.expected.requiredEvidence[0].sourceId, evidence: 'invented policy sentence' }] }, clear.expected);
  assert.equal(fabricated.quality, 0);
  assert.equal(fabricated.gates.citations_verified, false);
  const brokenGold = structuredClone(clear.expected); brokenGold.requiredEvidence[0].text = 'not present in the pinned source';
  assert.throws(() => scoreResearchObjective(packet, exactAnswer(clear.expected), brokenGold), /absent from its pinned document/);
  const blankGold = structuredClone(clear.expected); blankGold.requiredEvidence[0].text = '  ';
  assert.throws(() => scoreResearchObjective(packet, exactAnswer(clear.expected), blankGold), /absent from its pinned document/);

  const ambiguous = episode.query.cases.find(row => row.expected.unresolved);
  assert.ok(ambiguous);
  const ambiguousPacket = JSON.parse(ambiguous.args[0]);
  const explicit = scoreResearchObjective(ambiguousPacket, exactAnswer(ambiguous.expected), ambiguous.expected);
  assert.equal(explicit.quality, 1);
  const forced = scoreResearchObjective(ambiguousPacket,
    { ...exactAnswer(ambiguous.expected), label: 'critical', unresolved: false }, ambiguous.expected);
  assert.equal(forced.quality, 0.5);
  assert.equal(forced.gates.unresolved_status_correct, false);
});

test('allowed classification IDs accept only unambiguous casing, whitespace, and hyphen aliases', () => {
  const episode = buildResearchClassificationEpisodes().find(row => row.family.endsWith('jurisdiction-applicability'));
  const row = episode.support.cases[1];
  const packet = JSON.parse(row.args[0]);
  const answer = exactAnswer(row.expected);
  const alias = scoreResearchObjective(packet, { ...answer, label: '  NOT   applicable ' }, row.expected);
  assert.equal(alias.quality, 1);
  const unknown = scoreResearchObjective(packet, { ...answer, label: 'outside-policy' }, row.expected);
  assert.equal(unknown.quality, 0);
  assert.equal(unknown.gates.classification_label, false);
  assert.throws(() => scoreResearchObjective({ ...packet, allowedLabels: ['not-applicable', 'not applicable'] }, answer, row.expected),
    /invalid host research allowed-label catalog/);
});

test('research dossiers exercise exceptions, scope, supersession, and ambiguous authorities', () => {
  const episodes = buildResearchClassificationEpisodes();
  const labels = episodes.flatMap(episode => [...episode.support.cases, ...episode.query.cases].map(row => row.expected.label));
  assert.ok(labels.includes(null));
  const corpus = JSON.stringify(episodes.map(episode => [...episode.support.cases, ...episode.query.cases].map(row => row.expected.documents)));
  for (const phrase of ['overrides the general response matrix', 'both signed and marked effective', 'not randomized',
    'Regardless of processing location', 'no precedence or supersession entry', 'supersedes the supplier-address test']) assert.ok(corpus.includes(phrase));
});
