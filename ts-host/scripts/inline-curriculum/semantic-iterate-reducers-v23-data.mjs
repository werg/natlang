import { caseFrom } from './semantic-iterate-reducers-v18-novel-data.mjs';
import { reducerDomainSpecs } from './semantic-iterate-reducers-v22-novel-data.mjs';
import { selectionContract } from './semantic-iterate-reducer-selection-contract.mjs';

const measureNames = {
  'accessible-procurement': 'total bid cost in credits',
  'vendor-api-migration': 'verified endpoint coverage points',
  'research-cohort-classification': 'enrolled participants',
  'maintenance-work-order-priority': 'verified risk-reduction points',
  'document-claim-scope': 'evidence grade on the stated 1–5 scale',
  'community-service-eligibility': 'documented need-score points',
  'records-retention-disposition': 'eligible records in the series',
  'incident-mitigation-selection': 'verified user-impact minutes avoided',
};
const requests = {
  'accessible-procurement': 'Review candidate public-service portal bids against accessibility, interoperability, and the 180000-credit cap, then rank compliant bids by total cost.',
  'vendor-api-migration': 'Review connector releases for the next deployment window against contract compatibility, rollback readiness, and support coverage; rank qualifying releases by verified migration coverage.',
  'research-cohort-classification': 'Review cohorts against the stated population, preregistration, and missing-data requirements; rank qualifying cohorts by enrolled sample.',
  'maintenance-work-order-priority': 'Review work orders for verified hazards, crew certification, and parts readiness; rank qualifying work orders by risk reduction.',
  'document-claim-scope': 'Review report claims for direct source support within the reporting period; rank qualifying claims by evidence grade.',
  'community-service-eligibility': 'Review applicants for verified residency, need threshold, and duplicate enrollment; rank qualifying applicants by documented need score.',
  'records-retention-disposition': 'Review record series for expired retention and absence of legal holds or audit freezes; rank qualifying series by record count.',
  'incident-mitigation-selection': 'Review mitigations for safety approval, tested rollback, and backed-up data; rank qualifying mitigations by user-impact minutes avoided.',
};
const eligibilityRules = {
  'accessible-procurement': 'A bid qualifies only if its accessibility audit passes, identity protocol is compatible, and total is within the stated 180000-credit cap.',
  'vendor-api-migration': 'A release qualifies only when contract tests pass, rollback is verified, and vendor support covers deployment.',
  'research-cohort-classification': 'A cohort qualifies only when the population matches, the protocol is preregistered, and missing outcome data is at most 5 percent.',
  'maintenance-work-order-priority': 'A work order qualifies only when the hazard is verified, the assigned crew has listed certification, and required parts are on site.',
  'document-claim-scope': 'A claim qualifies only when its cited source directly supports it and the source date falls in the reporting period.',
  'community-service-eligibility': 'An applicant qualifies only when residency is verified, documented need meets the program threshold, and duplicate enrollment is absent.',
  'records-retention-disposition': 'A record series qualifies only when retention has expired and no legal hold or audit freeze applies.',
  'incident-mitigation-selection': 'A mitigation qualifies only when its safety review passes, rollback is tested, and affected data is backed up.',
};

const eligibleRows = base => base.candidates.filter(base.eligible);

function makeVariant(base, domainIndex, count) {
  const action = base.approvedAction;
  const originalEligible = eligibleRows(base);
  if (originalEligible.length < 2)
    throw new Error(`${base.slug}: V23 requires at least two fact-supported eligible rows for both cardinalities`);
  const tieSingle = domainIndex % 2 === 0;
  const candidates = base.candidates.map((row, index) => ({
    ...row,
    ...(count === 1 && tieSingle && base.eligible(row) ? { metric: originalEligible[0].metric } : {}),
    id: `${row.id}-V23${count === 1 ? 'S' : 'M'}${index + 1}`,
  }));
  const requestId = `${base.requestId}-V23-${count === 1 ? 'S' : 'M'}`;
  const contract = selectionContract({ count, direction: base.direction ?? 'desc', measureName: measureNames[base.slug] });
  const idCompare = (a, b) => a.id.localeCompare(b.id);
  const compare = (a, b) => (base.direction === 'asc' ? a.metric - b.metric : b.metric - a.metric) || idCompare(a, b);
  const eligible = base.eligible;
  const actionScope = count === 1 ? 'one item' : `up to ${count} selected items`;
  const authorityText = base.authorized
    ? `AUTHORITY STATUS: approved. The approving authority recorded ${action} authorization for ${actionScope} under request ${requestId}.`
    : `AUTHORITY STATUS: not approved. The approving authority has not recorded ${action} authorization for ${actionScope} under request ${requestId}; approval is pending.`;
  const cardinalityRule = `Selection cardinality: ${contract.instruction}`;
  const requestCount = count === 1
    ? 'one item when any item qualifies, and none if no item qualifies'
    : `up to ${count} items, selecting exactly ${count} whenever at least ${count} qualify and all qualifying items when fewer do`;
  const spec = {
    slug: `${base.slug}-${count === 1 ? 'single' : 'pair'}`,
    sourceGroup: `v22:${base.slug}:world`,
    domain: base.domain,
    owner: `${base.domain} operations desk`,
    requestId,
    requestText: `${requests[base.slug]} Request ${requestId} selects ${requestCount}.`,
    ruleText: `${eligibilityRules[base.slug]} ${cardinalityRule} Apply the authority record only after determining the complete selected set.`,
    instruction: `${requests[base.slug]} Request ${requestId}. ${cardinalityRule}`,
    registerHeading: base.registerHeading,
    auditHeading: base.auditHeading,
    authorityHeading: base.authorityHeading,
    candidates,
    compare,
    provisional: rows => contract.select([...rows].sort(compare)),
    eligible,
    select: rows => contract.select([...rows].sort(compare)),
    format: contract.format,
    measure: contract.formatMeasure,
    registerLine: base.registerLine,
    auditLine: base.auditLine,
    exceptionText: base.exceptionText,
    authorityText,
    authorized: base.authorized,
    selectionFormat: contract.selectionFormat,
    measureFormat: contract.measureFormat,
    finalDecisions: [action, 'hold', base.noAction ?? 'no_action'],
    approvedAction: action,
    noAction: base.noAction ?? 'no_action',
    selectionCardinality: count,
    tieSingle,
    domainIndex,
  };
  const row = caseFrom(spec);
  row.group = spec.sourceGroup;
  row.source_summary = { requestId, candidates: candidates.map(candidate => ({ ...candidate })),
    selectedItems: row.source_summary.selectedItems, measure: row.source_summary.measure,
    decision: row.source_summary.decision, authorized: base.authorized,
    parent_group: spec.sourceGroup, domain_index: domainIndex, selection_cardinality: count };
  row.decision_rule = spec.ruleText;
  row.instruction = spec.instruction;
  row.selection_format = spec.selectionFormat;
  row.measure_format = spec.measureFormat;
  return { spec, row };
}

export const reducerVariants = reducerDomainSpecs.flatMap((base, domainIndex) => [1, 2].map(count =>
  makeVariant(base, domainIndex, count)));

export const sourceDomains = reducerDomainSpecs.map(({ slug, domain }, index) => ({
  slug, domain, inherited_group: `v22:${slug}:world`, inherited_split: index % 2 === 0 ? 'train' : 'test',
  variants: ['exact-one', 'exact-two-ranked'],
}));
