#!/usr/bin/env node
/** Fictional, source-grounded research classification packets. No external texts or model calls. */
import { createHash } from 'node:crypto';
import { mkdirSync, openSync, writeFileSync, closeSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const metric = { schema: 'natlang.skill-research/1', kind: 'research-classification' };
const digest = value => createHash('sha256').update(typeof value === 'string' ? value : JSON.stringify(value)).digest('hex');
const ID = (prefix, caseId, suffix) => `${prefix}-${caseId}-${suffix}`;
const doc = (id, title, text, kind = 'policy') => ({ id, title, text, kind });

function incidentCases() {
  return [
    { id: 'incident-staffed-release', group: 'incident-rule/occupied-hours', brief: 'At 10:20 local time, a sensor confirms a pressure release in an occupied pump room. The isolation valve is open and the trend is not stable. Determine the response tier.', facts: { hour: 10, occupied: true, confirmedRelease: true, isolated: false, stable: false }, docs: [
      doc('IR-A1', 'Pressure response matrix', 'A confirmed pressure release in an occupied zone during staffed hours is Critical. A confirmed release outside those conditions is High.'),
      doc('IR-A2', 'Control-room staffing schedule', 'Pump rooms are staffed from 07:00 through 19:00 local time, including both endpoint hours.'),
      doc('IR-A3', 'Sensor status record', 'The release sensor is confirmed and the isolation valve remains open.')],
      required: [['IR-A1', 'A confirmed pressure release in an occupied zone during staffed hours is Critical.'], ['IR-A2', 'Pump rooms are staffed from 07:00 through 19:00 local time, including both endpoint hours.']], resolve: f => f.confirmedRelease && f.occupied && f.hour >= 7 && f.hour <= 19 ? 'critical' : 'high' },
    { id: 'incident-isolation-exception', group: 'incident-rule/isolation-override', brief: 'At 11:00, a historic release alarm remains on the dashboard. The isolation valve is now closed and the last three sensor samples are stable. Determine the current response tier.', facts: { hour: 11, occupied: true, confirmedRelease: true, isolated: true, stable: true }, docs: [
      doc('IR-B1', 'Pressure response matrix', 'A confirmed pressure release in an occupied zone during staffed hours is Critical.'),
      doc('IR-B2', 'Isolation exception bulletin', 'When the isolation valve is closed and three consecutive samples are stable, classify the event as Monitor even if an earlier release alarm remains recorded. This exception overrides the general response matrix.'),
      doc('IR-B3', 'Sensor status record', 'The valve is closed. The latest three samples are stable; the earlier alarm is historical.')],
      required: [['IR-B2', 'When the isolation valve is closed and three consecutive samples are stable, classify the event as Monitor even if an earlier release alarm remains recorded.'], ['IR-B2', 'This exception overrides the general response matrix.'], ['IR-B3', 'The valve is closed. The latest three samples are stable; the earlier alarm is historical.']], resolve: f => f.isolated && f.stable ? 'monitor' : f.confirmedRelease && f.occupied && f.hour >= 7 && f.hour <= 19 ? 'critical' : 'high' },
    { id: 'incident-after-hours', group: 'incident-rule/after-hours-scope', brief: 'At 22:10 local time, a confirmed release is reported in an unoccupied pump room. Determine the response tier.', facts: { hour: 22, occupied: false, confirmedRelease: true, isolated: false, stable: false }, docs: [
      doc('IR-C1', 'Pressure response matrix', 'For a confirmed release, assign Critical only when the zone is occupied during staffed hours. At all other times assign High.'),
      doc('IR-C2', 'Control-room staffing schedule', 'Pump rooms are staffed from 07:00 through 19:00 local time, including both endpoint hours.'),
      doc('IR-C3', 'Sensor status record', 'The release is confirmed; the pump room is unoccupied at the reported time.')],
      required: [['IR-C1', 'At all other times assign High.'], ['IR-C2', 'Pump rooms are staffed from 07:00 through 19:00 local time, including both endpoint hours.']], resolve: f => f.confirmedRelease && f.occupied && f.hour >= 7 && f.hour <= 19 ? 'critical' : 'high' },
    { id: 'incident-unresolved-shift-boundary', group: 'incident-rule/unresolved-scope-conflict', brief: 'At 19:30 local time, a confirmed release occurs in an occupied pump room. Two currently effective bulletins disagree about whether staffed hours end at 19:00 or 20:00, and neither supersedes the other. Determine whether a tier can be assigned.', facts: { hour: 19.5, occupied: true, confirmedRelease: true, unresolvedScopeConflict: true }, docs: [
      doc('IR-D1', 'Operations schedule revision 4', 'For response classification, staffed hours end at 19:00 local time.'),
      doc('IR-D2', 'Operations schedule revision 5', 'For response classification, staffed hours end at 20:00 local time. This revision is marked effective, but its approval ledger is missing.'),
      doc('IR-D3', 'Pressure response matrix', 'A confirmed pressure release in an occupied zone during staffed hours is Critical; outside staffed hours it is High.')],
      required: [['IR-D1', 'staffed hours end at 19:00 local time.'], ['IR-D2', 'staffed hours end at 20:00 local time.']], resolve: f => f.unresolvedScopeConflict ? null : 'high' },
  ];
}

function procurementCases() {
  return [
    { id: 'procurement-local-fast-track', group: 'procurement-rule/local-threshold', brief: 'A local Northmere supplier bids 18,400 credits for standard replacement filters. The application is complete and no exception is requested. Determine eligibility for the fast-track procedure.', facts: { jurisdiction: 'Northmere', localSupplier: true, amount: 18400, complete: true, soleSourceSafety: false }, docs: [
      doc('PR-A1', 'Northmere fast-track rule', 'A complete application from a Northmere supplier is eligible for fast-track review when the total is at most 20,000 credits.'),
      doc('PR-A2', 'Supplier registry extract', 'The supplier address and registration are both in Northmere.'),
      doc('PR-A3', 'Procurement intake record', 'The application is complete. The submitted total is 18,400 credits.')],
      required: [['PR-A1', 'eligible for fast-track review when the total is at most 20,000 credits.'], ['PR-A2', 'The supplier address and registration are both in Northmere.']], resolve: f => f.complete && f.localSupplier && f.amount <= 20000 ? 'eligible' : 'ineligible' },
    { id: 'procurement-superseded-address-rule', group: 'procurement-rule/signed-supersession', brief: 'On 10 June, a complete 12,000-credit warehouse-delivery contract is submitted by a supplier registered in Southmere with an operating depot in Northmere. A signed addendum effective on 1 June changes the address test. Determine eligibility under the current rule.', facts: { jurisdiction: 'Northmere', intakeDate: '10 June', localSupplier: false, operatingDepotNorthmere: true, warehouseContract: true, amount: 12000, complete: true, soleSourceSafety: false }, docs: [
      doc('PR-B1', 'Northmere fast-track rule', 'A complete application from a Northmere supplier is eligible for fast-track review when the total is at most 20,000 credits.'),
      doc('PR-B2', 'Signed warehouse-contract addendum', 'Effective 1 June, this signed addendum supersedes the supplier-address test for warehouse-delivery contracts. A supplier with an operating depot in Northmere qualifies for fast-track review up to 15,000 credits; registered office is not controlling.'),
      doc('PR-B3', 'Supplier registry extract', 'The registered office is in Southmere and the operating depot is in Northmere.')],
      required: [['PR-B1', 'eligible for fast-track review when the total is at most 20,000 credits.'], ['PR-B2', 'this signed addendum supersedes the supplier-address test for warehouse-delivery contracts.'], ['PR-B2', 'A supplier with an operating depot in Northmere qualifies for fast-track review up to 15,000 credits; registered office is not controlling.'], ['PR-B3', 'The registered office is in Southmere and the operating depot is in Northmere.']], resolve: f => f.complete && (f.warehouseContract && f.operatingDepotNorthmere && f.amount <= 15000 || f.localSupplier && f.amount <= 20000) ? 'eligible' : 'ineligible' },
    { id: 'procurement-safety-exception', group: 'procurement-rule/sole-source-exception', brief: 'A 54,000-credit order has one certified supplier for a safety-critical replacement part. The signed waiver names the part and states that no equivalent is available. Determine eligibility for direct award.', facts: { jurisdiction: 'Northmere', localSupplier: false, amount: 54000, complete: true, soleSourceSafety: true }, docs: [
      doc('PR-C1', 'Ordinary procurement threshold', 'Competitive fast-track review is limited to complete bids no greater than 20,000 credits from registered Northmere suppliers.'),
      doc('PR-C2', 'Safety-critical sole-source exception', 'A direct award may exceed the ordinary threshold when a signed waiver identifies a safety-critical part and certifies that no equivalent supplier is available.'),
      doc('PR-C3', 'Signed waiver and certification', 'The waiver is signed, identifies the replacement part as safety-critical, and certifies that no equivalent supplier is available.')],
      required: [['PR-C2', 'A direct award may exceed the ordinary threshold when a signed waiver identifies a safety-critical part'], ['PR-C3', 'The waiver is signed, identifies the replacement part as safety-critical, and certifies that no equivalent supplier is available.']], resolve: f => f.soleSourceSafety && f.complete ? 'eligible' : f.complete && f.localSupplier && f.amount <= 20000 ? 'eligible' : 'ineligible' },
    { id: 'procurement-coeffective-amendments', group: 'procurement-rule/unresolved-amendment-conflict', brief: 'A 19,000-credit application is complete. The supplier has a Southmere office and a Northmere operating address. Two signed amendments are both listed as effective and disagree about which address controls; no later amendment resolves the conflict. Determine whether eligibility can be decided.', facts: { localSupplier: null, amount: 19000, complete: true, coeffectiveConflict: true }, docs: [
      doc('PR-D1', 'Supplier qualification amendment 8', 'For eligibility, the registered office must be in Northmere; the operating address is not controlling.'),
      doc('PR-D2', 'Supplier qualification amendment 9', 'For eligibility, the principal operating address must be in Northmere; the registered office is not controlling.'),
      doc('PR-D3', 'Amendment register', 'Amendments 8 and 9 are both signed and marked effective on the same date. The register contains no precedence or supersession entry.')],
      required: [['PR-D1', 'the registered office must be in Northmere; the operating address is not controlling.'], ['PR-D2', 'the principal operating address must be in Northmere; the registered office is not controlling.'], ['PR-D3', 'contains no precedence or supersession entry.']], resolve: f => f.coeffectiveConflict ? null : f.localSupplier && f.amount <= 20000 && f.complete ? 'eligible' : 'ineligible' },
  ];
}

function studyCases() {
  return [
    { id: 'study-randomized', group: 'study-rule/random-allocation', brief: 'A team assigns participants by a computer-generated random sequence to the new protocol or usual care. The protocol is assigned by the study rather than chosen by clinicians. Classify the study design.', facts: { allocation: 'random', interventionAssigned: true, clinicianChoice: false, recordsRetrospective: false, recordsConflict: false }, docs: [
      doc('ST-A1', 'Methods taxonomy', 'A study that assigns an intervention using random allocation to comparison groups is randomized-controlled.'),
      doc('ST-A2', 'Allocation protocol', 'A computer-generated random sequence assigns each enrolled participant to the new protocol or usual care.'),
      doc('ST-A3', 'Enrollment log', 'The study team assigned the protocol; participants and clinicians did not select their group.')],
      required: [['ST-A1', 'assigns an intervention using random allocation to comparison groups is randomized-controlled.'], ['ST-A2', 'A computer-generated random sequence assigns each enrolled participant to the new protocol or usual care.']], resolve: f => f.recordsConflict ? 'ambiguous' : f.interventionAssigned ? (f.allocation === 'random' ? 'randomized-controlled' : 'quasi-experimental') : 'observational' },
    { id: 'study-observational', group: 'study-rule/no-assigned-intervention', brief: 'Researchers review existing appointment records and compare outcomes for people who independently selected different clinics. The investigators assign no treatment or exposure. Classify the study design.', facts: { allocation: 'none', interventionAssigned: false, clinicianChoice: false, recordsRetrospective: true, recordsConflict: false }, docs: [
      doc('ST-B1', 'Methods taxonomy', 'An analysis of existing records with no intervention assigned by the investigators is observational.'),
      doc('ST-B2', 'Data provenance statement', 'The dataset contains appointments and outcomes recorded before this analysis began.'),
      doc('ST-B3', 'Clinic assignment note', 'Patients selected their clinics independently. The research team assigned no treatment, clinic, or exposure.')],
      required: [['ST-B1', 'with no intervention assigned by the investigators is observational.'], ['ST-B3', 'The research team assigned no treatment, clinic, or exposure.']], resolve: f => f.recordsConflict ? 'ambiguous' : f.interventionAssigned ? (f.allocation === 'random' ? 'randomized-controlled' : 'quasi-experimental') : 'observational' },
    { id: 'study-clinic-phasein', group: 'study-rule/nonrandom-intervention', brief: 'Clinics receive the new protocol in a fixed calendar order chosen by the program office. Each clinic changes protocol, but the order is not randomized. Classify the design.', facts: { allocation: 'calendar-order', interventionAssigned: true, clinicianChoice: false, recordsRetrospective: false, recordsConflict: false }, docs: [
      doc('ST-C1', 'Methods taxonomy', 'A study that assigns an intervention without random allocation is quasi-experimental.'),
      doc('ST-C2', 'Phase-in schedule', 'Clinics switch to the new protocol in the fixed order listed by the program office.'),
      doc('ST-C3', 'Schedule approval', 'The phase-in order was selected for staffing convenience and was not randomized.')],
      required: [['ST-C1', 'assigns an intervention without random allocation is quasi-experimental.'], ['ST-C3', 'was not randomized.']], resolve: f => f.recordsConflict ? 'ambiguous' : f.interventionAssigned ? (f.allocation === 'random' ? 'randomized-controlled' : 'quasi-experimental') : 'observational' },
    { id: 'study-conflicting-allocation-records', group: 'study-rule/superseded-protocol-conflict', brief: 'The protocol says computer randomization was planned. The signed enrollment ledger says clinicians selected each participant’s group. Neither record says whether randomization actually ran, and no correction is filed. Classify only if the evidence supports one design.', facts: { allocation: null, interventionAssigned: true, clinicianChoice: true, recordsRetrospective: false, recordsConflict: true }, docs: [
      doc('ST-D1', 'Study protocol', 'The study will assign participants to two groups using a computer-generated random sequence.'),
      doc('ST-D2', 'Signed enrollment ledger', 'Clinicians selected the group for each participant. This ledger is signed for the completed enrollment period.'),
      doc('ST-D3', 'Document register', 'The protocol and signed ledger are both current records. No execution log or superseding correction resolves their disagreement.')],
      required: [['ST-D1', 'using a computer-generated random sequence.'], ['ST-D2', 'Clinicians selected the group for each participant.'], ['ST-D3', 'No execution log or superseding correction resolves their disagreement.']], resolve: f => f.recordsConflict ? null : f.interventionAssigned ? (f.allocation === 'random' ? 'randomized-controlled' : 'quasi-experimental') : 'observational' },
  ];
}

function jurisdictionCases() {
  return [
    { id: 'jurisdiction-local-processing', group: 'jurisdiction-rule/territorial-trigger', brief: 'A Southmere company performs regulated archive processing on a server physically located in Northmere. The records fall in the named protected category. Decide whether the fictional Northmere policy applies.', facts: { processingPlace: 'Northmere', entityRegistration: 'Southmere', recordClass: 'protected', remoteException: false, boundaryConflict: false }, docs: [
      doc('JU-A1', 'Northmere territorial rule', 'This fictional policy applies to protected records processed on infrastructure physically located in Northmere, regardless of the operator’s registered office.'),
      doc('JU-A2', 'Record classification schedule', 'Archive category P is a protected record class under this policy.'),
      doc('JU-A3', 'Infrastructure inventory', 'The processing server is physically located in Northmere. The operator is registered in Southmere.')],
      required: [['JU-A1', 'applies to protected records processed on infrastructure physically located in Northmere'], ['JU-A2', 'Archive category P is a protected record class under this policy.']], resolve: f => f.boundaryConflict ? 'ambiguous' : f.recordClass === 'protected' && f.processingPlace === 'Northmere' ? 'applicable' : 'not-applicable' },
    { id: 'jurisdiction-registration-only', group: 'jurisdiction-rule/registration-not-trigger', brief: 'A Northmere company processes ordinary archive records on infrastructure physically located in Southmere. No Northmere processing or listed exception is present. Decide whether the fictional Northmere policy applies.', facts: { processingPlace: 'Southmere', entityRegistration: 'Northmere', recordClass: 'ordinary', remoteException: false, boundaryConflict: false }, docs: [
      doc('JU-B1', 'Northmere territorial rule', 'This fictional policy applies to protected records processed on infrastructure physically located in Northmere, regardless of the operator’s registered office.'),
      doc('JU-B2', 'Record classification schedule', 'Archive category O is ordinary and is not within the protected record class.'),
      doc('JU-B3', 'Infrastructure inventory', 'The processing server is physically located in Southmere. The operator is registered in Northmere.')],
      required: [['JU-B1', 'regardless of the operator’s registered office.'], ['JU-B3', 'The processing server is physically located in Southmere.']], resolve: f => f.boundaryConflict ? 'ambiguous' : f.recordClass === 'protected' && f.processingPlace === 'Northmere' ? 'applicable' : 'not-applicable' },
    { id: 'jurisdiction-remote-exception', group: 'jurisdiction-rule/remote-processing-exception', brief: 'A foreign operator processes protected Northmere records in a Southmere cloud region. The signed exception states that a Northmere-controlled encryption key makes this processing subject to the fictional policy. The key is controlled in Northmere. Decide applicability.', facts: { processingPlace: 'Southmere', entityRegistration: 'Eastmere', recordClass: 'protected', remoteException: true, keyControl: 'Northmere', boundaryConflict: false }, docs: [
      doc('JU-C1', 'Northmere territorial rule', 'The ordinary rule applies to protected records processed on infrastructure physically located in Northmere.'),
      doc('JU-C2', 'Remote-key exception', 'Regardless of processing location or operator registration, the policy also applies when protected records are processed using an encryption key controlled in Northmere.'),
      doc('JU-C3', 'Key-control audit', 'The encryption key used for this processing is controlled by the Northmere records office.')],
      required: [['JU-C2', 'the policy also applies when protected records are processed using an encryption key controlled in Northmere.'], ['JU-C3', 'The encryption key used for this processing is controlled by the Northmere records office.']], resolve: f => f.boundaryConflict ? 'ambiguous' : f.recordClass === 'protected' && (f.processingPlace === 'Northmere' || f.remoteException && f.keyControl === 'Northmere') ? 'applicable' : 'not-applicable' },
    { id: 'jurisdiction-boundary-conflict', group: 'jurisdiction-rule/overlapping-boundary', brief: 'A protected archive is processed at a facility on the Northmere–Eastmere boundary. The location audit assigns the facility to both jurisdictions under two current maps. No controlling boundary order is available. Decide whether applicability is determinate.', facts: { processingPlace: null, entityRegistration: 'Eastmere', recordClass: 'protected', remoteException: false, boundaryConflict: true }, docs: [
      doc('JU-D1', 'Northmere map annex', 'The boundary facility is within Northmere for purposes of the protected-record processing rule.'),
      doc('JU-D2', 'Eastmere map annex', 'The same boundary facility is within Eastmere and outside Northmere for purposes of archive processing.'),
      doc('JU-D3', 'Boundary review register', 'Both map annexes are marked current. The register contains no controlling order or corrected facility coordinate.')],
      required: [['JU-D1', 'within Northmere for purposes of the protected-record processing rule.'], ['JU-D2', 'within Eastmere and outside Northmere for purposes of archive processing.'], ['JU-D3', 'contains no controlling order or corrected facility coordinate.']], resolve: f => f.boundaryConflict ? null : f.recordClass === 'protected' && f.processingPlace === 'Northmere' ? 'applicable' : 'not-applicable' },
  ];
}

const domains = [
  { id: 'incident-triage', title: 'Fictional industrial incident triage', allowedLabels: ['critical', 'high', 'monitor'], cases: incidentCases() },
  { id: 'procurement-eligibility', title: 'Fictional procurement eligibility', allowedLabels: ['eligible', 'ineligible'], cases: procurementCases() },
  { id: 'study-method', title: 'Fictional study-method classification', allowedLabels: ['randomized-controlled', 'quasi-experimental', 'observational', 'ambiguous'], cases: studyCases() },
  { id: 'jurisdiction-applicability', title: 'Fictional jurisdiction and policy applicability', allowedLabels: ['applicable', 'not-applicable', 'ambiguous'], cases: jurisdictionCases() },
];

function serviceSource(documents) {
  const data = JSON.stringify(documents.map(({ id, title, text, kind }) => ({ id, title, text, kind })));
  return `type ResearchHit = { id: string; title: string; kind: string };
const PRIVATE_DOCUMENTS = ${data};
export function search(query: string): ResearchHit[] {
  const words = query.toLowerCase().split(/[^\\p{L}\\p{N}]+/u).filter(word => word.length > 2);
  const ranked = PRIVATE_DOCUMENTS.map(document => ({ document, score: words.filter(word => (document.title + ' ' + document.text).toLowerCase().includes(word)).length }))
    .filter(row => row.score > 0).sort((a, b) => b.score - a.score || a.document.id.localeCompare(b.document.id));
  return ranked.slice(0, 12).map(row => ({ id: row.document.id, title: row.document.title, kind: row.document.kind }));
}
export function read(sourceId: string): string {
  const document = PRIVATE_DOCUMENTS.find(row => row.id === sourceId);
  if (!document) throw new Error('Unknown research source ID');
  return document.text;
}`;
}

const target = { kind: 'improvement-case', entry: 'solve.nl', source: { schema: 'natlang.research-classification/1', id: 'fictional-policy-dossiers-v1' },
  files: { 'solve.nl': '---\nargs: { packet: string }\nreturns: string\n---\nYou receive a JSON research packet with a fictional case brief, an allowedLabels array of exact classification IDs, and a catalog containing source IDs, titles, and kinds. The document bodies are not in the packet. Use research.search(query) to locate relevant sources and research.read(sourceId) to retrieve each document. Join the governing rule, scope/exception/supersession documents, and case facts before deciding. Your label must be one exact ID from allowedLabels, or null only when the evidence is unresolved. Return JSON only: {"label":"exact allowed ID or null","unresolved":true|false,"citations":[{"sourceId":"...","evidence":"a contiguous quote from that retrieved document"}]}. Cite source text that supports the decision. If currently effective sources conflict and no precedence resolves them, return label null and mark unresolved true rather than choosing a substantive classification. Do not treat catalog titles as evidence.\n' } };

function buildResearchClassificationEpisodes() {
  return domains.map(domain => {
    const support = [], query = [], sourceGroups = [];
    for (const [index, item] of domain.cases.entries()) {
      const documents = item.docs.map(row => ({ id: row.id, title: row.title, kind: row.kind, text: row.text }));
      const catalog = documents.map(({ id, title, kind }) => ({ id, title, kind }));
      const packet = { disclaimer: 'Fictional training exercise. These rules apply only to the fictional setting described here.',
        domain: domain.title, question: item.brief, allowedLabels: domain.allowedLabels, catalog };
      const label = item.resolve(item.facts);
      const expected = { kind: 'research-classification', label, unresolved: label === null,
        requiredEvidence: item.required.map(([sourceId, text]) => ({ sourceId, text })), documents };
      const row = { id: `research-${digest(domain.id + ':' + item.id).slice(0, 20)}`,
        group: `research-template/${domain.id}/${item.group}`, args: [JSON.stringify(packet)], expected,
        services: { research: serviceSource(documents) } };
      sourceGroups.push(row.group);
      (index < 2 ? support : query).push(row);
    }
    return { version: 'natlang.skill-episode/1', id: `research-${domain.id}-v1`, family: `research:${domain.id}`, split: 'train',
      source_groups: sourceGroups, license: 'project-generated', target, library: { kind: 'empty', skills: {} },
      support: { cases: support }, query: { cases: query }, operations: ['create', 'revise'], limits: { maxSteps: 6 },
      provenance: { generator: 'natlang.research-classification-episodes/1', metric,
        query_holdout: domain.cases.slice(2).map(item => item.group), fictional_domain: domain.title,
        scoring_contract: 'exact source-grounded citations; classification and explicit ambiguity are host-scored' } };
  });
}

export { buildResearchClassificationEpisodes, domains, metric, serviceSource };

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const at = process.argv.indexOf('--out'), argument = at >= 0 ? process.argv[at + 1] : undefined;
  if (!argument || argument.startsWith('--')) throw Error('Usage: build-research-classification-episodes.mjs --out DIR');
  const out = resolve(argument), episodes = buildResearchClassificationEpisodes();
  mkdirSync(out, { recursive: true });
  const body = episodes.map(row => JSON.stringify(row)).join('\n') + '\n';
  const fd = openSync(join(out, 'research-classification-episodes.jsonl'), 'wx');
  try { writeFileSync(fd, body); } finally { closeSync(fd); }
  const manifest = { schema: 'natlang.research-classification-episodes/1', episodes: episodes.length,
    cases: episodes.reduce((sum,row)=>sum+row.support.cases.length+row.query.cases.length,0), support_cases: episodes.reduce((n, row) => n + row.support.cases.length, 0),
    query_cases: episodes.reduce((n, row) => n + row.query.cases.length, 0), domains: domains.map(row => row.id),
    metric, model_calls: 0, sha256: digest(body) };
  const mf = openSync(join(out, 'research-classification-episodes.manifest.json'), 'wx');
  try { writeFileSync(mf, JSON.stringify(manifest, null, 2) + '\n'); } finally { closeSync(mf); }
  process.stdout.write(`${JSON.stringify(manifest)}\n`);
}
