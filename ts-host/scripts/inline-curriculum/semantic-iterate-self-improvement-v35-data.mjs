import { caseFrom } from './semantic-iterate-reducers-v18-novel-data.mjs';

// V35 source retains the authored requests while assigning each complete domain
// family to one split. These are fictional task families, not externally
// grounded independent domains.
const domains = [
  {
    slug: 'safety-corrective-action', label: 'workplace safety corrective-action review', owner: 'Safety assurance office', action: 'schedule',
    metricName: 'residual risk points after correction', unit: 'risk points', direction: 'asc', count: 1,
    requirement: 'a safety director release',
    rule: 'A corrective action qualifies only when its hazard-control owner accepted the action, the control was field-verified, and the action does not transfer the hazard to an unassessed work area. Rank qualifying actions by residual risk points after correction, lowest first; ties use ascending complete action ID. Scheduling requires a safety director release.',
    conditions: [
      ['Owner acceptance', 'the named hazard-control owner accepted this action', 'the named hazard-control owner has not accepted this action'],
      ['Field verification', 'the control was verified at the work location', 'the control has not been verified at the work location'],
      ['Hazard transfer', 'no unassessed work area receives the hazard', 'the action transfers the hazard to an unassessed work area'],
    ],
    scenarios: [
      { id:'SAFE-510', authority:true, rows:[['ACT-A',8,[1,1,1]],['ACT-B',3,[1,1,1]],['ACT-C',1,[1,0,1]],['ACT-D',5,[1,1,0]]] },
      { id:'SAFE-511', authority:false, rows:[['ACT-E',4,[1,1,1]],['ACT-F',4,[1,1,1]],['ACT-G',2,[0,1,1]],['ACT-H',6,[1,1,1]]] },
      { id:'SAFE-512', authority:true, rows:[['ACT-J',1,[1,0,1]],['ACT-K',2,[0,1,1]],['ACT-L',3,[1,1,0]],['ACT-M',4,[0,0,1]]] },
      { id:'SAFE-513', authority:false, rows:[['ACT-N',9,[1,1,1]],['ACT-P',5,[1,1,1]],['ACT-Q',2,[1,0,1]],['ACT-R',7,[1,1,0]]] },
    ], authorityText: id => `Safety director release for ${id} is ${id.endsWith('510') || id.endsWith('512') ? 'signed and current' : 'not recorded'}.`,
    exception: 'A lower residual-risk score cannot replace owner acceptance or field verification.',
  },
  {
    slug: 'provenance-reconciliation', label: 'conflicting-record provenance reconciliation', owner: 'Records integrity team', action: 'reconcile',
    metricName: 'independent provenance links', unit: 'verified links', direction: 'desc', count: 1,
    requirement: 'a records custodian approval',
    rule: 'A source record qualifies as the reconciliation basis only when its original source is identified, its custody transfer is documented, and an independent corroborating record agrees on the key date. Rank qualifying records by independent provenance links, greatest first; ties use ascending complete record ID. Reconciliation requires a records custodian approval.',
    conditions: [
      ['Original source', 'the original issuing source is identified', 'the original issuing source is not identified'],
      ['Custody transfer', 'every transfer is documented in the custody log', 'a transfer is missing from the custody log'],
      ['Independent corroboration', 'an independent record confirms the key date', 'no independent record confirms the key date'],
    ],
    scenarios: [
      { id:'PROV-520', authority:true, rows:[['REC-A',7,[1,1,1]],['REC-B',11,[1,1,1]],['REC-C',15,[1,0,1]],['REC-D',20,[0,1,1]]] },
      { id:'PROV-521', authority:false, rows:[['REC-E',9,[1,1,1]],['REC-F',8,[1,1,1]],['REC-G',21,[1,0,1]],['REC-H',13,[1,1,1]]] },
      { id:'PROV-522', authority:true, rows:[['REC-J',8,[1,0,1]],['REC-K',6,[0,1,1]],['REC-L',5,[1,1,0]],['REC-M',4,[0,0,1]]] },
      { id:'PROV-523', authority:false, rows:[['REC-N',17,[1,1,1]],['REC-P',15,[1,1,1]],['REC-Q',22,[1,0,1]],['REC-R',18,[1,1,1]]] },
    ], authorityText: id => `Records custodian approval for ${id} is ${id.endsWith('520') || id.endsWith('522') ? 'signed' : 'pending'}.`,
    exception: 'A high link count cannot establish an unknown original source or fill a custody gap.',
  },
  {
    slug: 'experimental-plan-revision', label: 'experimental protocol revision selection', owner: 'Study design review group', action: 'adopt',
    metricName: 'unresolved design deviations', unit: 'deviations', direction: 'asc', count: 2,
    requirement: 'principal investigator approval',
    rule: 'A protocol revision qualifies only when its primary outcome is preregistered, its control arm is retained, and its analysis plan handles the stated missing-data mechanism. Select up to two qualifying revisions with the fewest unresolved design deviations, ties by ascending complete revision ID. Adoption requires principal investigator approval.',
    conditions: [
      ['Outcome preregistration', 'the primary outcome is preregistered before enrollment', 'the primary outcome was not preregistered before enrollment'],
      ['Control arm', 'the planned control arm is retained', 'the planned control arm is removed'],
      ['Missing-data plan', 'the analysis plan addresses the observed missing-data mechanism', 'the analysis plan does not address the observed missing-data mechanism'],
    ],
    scenarios: [
      { id:'EXPT-530', authority:true, rows:[['REV-A',6,[1,1,1]],['REV-B',3,[1,1,1]],['REV-C',1,[1,0,1]],['REV-D',4,[1,1,0]]] },
      { id:'EXPT-531', authority:false, rows:[['REV-E',2,[1,1,1]],['REV-F',2,[1,1,1]],['REV-G',1,[0,1,1]],['REV-H',5,[1,1,1]]] },
      { id:'EXPT-532', authority:true, rows:[['REV-J',1,[1,0,1]],['REV-K',2,[0,1,1]],['REV-L',3,[1,1,0]],['REV-M',4,[1,1,1]]] },
      { id:'EXPT-533', authority:false, rows:[['REV-N',5,[1,1,1]],['REV-P',7,[1,1,1]],['REV-Q',2,[1,0,1]],['REV-R',9,[1,1,0]]] },
    ], authorityText: id => `Principal investigator approval for ${id} is ${id.endsWith('530') || id.endsWith('532') ? 'signed' : 'not recorded'}.`,
    exception: 'Fewer deviations do not cure a missing preregistration, removed control arm, or unhandled missing-data mechanism.',
  },
  {
    slug: 'resource-window-scheduling', label: 'constrained resource-window scheduling', owner: 'Operations scheduling desk', action: 'assign',
    metricName: 'completion time in minutes', unit: 'minutes', direction: 'asc', count: 2,
    requirement: 'an operations lead release',
    rule: 'A work window qualifies only when the required skill is staffed, the resource is available for the full interval, and the assignment respects the worker rest rule. Select up to two qualifying windows with the earliest completion time, ties by ascending complete window ID. Assignment requires an operations lead release.',
    conditions: [
      ['Skill coverage', 'a currently qualified worker covers the required skill', 'no currently qualified worker covers the required skill'],
      ['Resource availability', 'the resource is available for the complete interval', 'the resource is unavailable during part of the interval'],
      ['Rest compliance', 'the worker meets the required rest interval', 'the worker does not meet the required rest interval'],
    ],
    scenarios: [
      { id:'SCHED-540', authority:true, rows:[['WIN-A',95,[1,1,1]],['WIN-B',80,[1,1,1]],['WIN-C',65,[1,0,1]],['WIN-D',75,[1,1,0]]] },
      { id:'SCHED-541', authority:false, rows:[['WIN-E',72,[1,1,1]],['WIN-F',88,[1,1,1]],['WIN-G',60,[0,1,1]],['WIN-H',93,[1,1,1]]] },
      { id:'SCHED-542', authority:true, rows:[['WIN-J',55,[1,0,1]],['WIN-K',58,[0,1,1]],['WIN-L',70,[1,1,0]],['WIN-M',83,[1,1,1]]] },
      { id:'SCHED-543', authority:false, rows:[['WIN-N',99,[1,1,1]],['WIN-P',102,[1,1,1]],['WIN-Q',65,[1,0,1]],['WIN-R',77,[1,1,0]]] },
    ], authorityText: id => `Operations lead release for ${id} is ${id.endsWith('540') || id.endsWith('542') ? 'recorded' : 'absent'}.`,
    exception: 'An earlier completion time cannot waive skill coverage, full-interval availability, or worker rest.',
  },
  {
    slug: 'compliance-document-revision', label: 'compliance document revision prioritization', owner: 'Compliance control office', action: 'revise',
    metricName: 'confirmed control gaps closed', unit: 'control gaps', direction: 'desc', count: 1,
    requirement: 'compliance officer authorization',
    rule: 'A proposed revision qualifies only when the cited requirement is currently in force, the gap is confirmed against the current document, and the proposed wording preserves the control owner’s responsibility. Rank qualifying revisions by confirmed control gaps closed, greatest first; ties use ascending complete revision ID. Publication requires compliance officer authorization.',
    conditions: [
      ['Current requirement', 'the cited requirement is in force for this document', 'the cited requirement is superseded'],
      ['Confirmed gap', 'the gap is confirmed in the current document', 'the gap is not present in the current document'],
      ['Owner responsibility', 'the proposed wording preserves the control owner responsibility', 'the proposed wording removes the control owner responsibility'],
    ],
    scenarios: [
      { id:'COMP-550', authority:true, rows:[['REV-A',5,[1,1,1]],['REV-B',9,[1,1,1]],['REV-C',12,[1,0,1]],['REV-D',8,[1,1,0]]] },
      { id:'COMP-551', authority:false, rows:[['REV-E',7,[1,1,1]],['REV-F',6,[1,1,1]],['REV-G',13,[1,0,1]],['REV-H',11,[1,1,1]]] },
      { id:'COMP-552', authority:true, rows:[['REV-J',14,[1,0,1]],['REV-K',12,[0,1,1]],['REV-L',9,[1,1,0]],['REV-M',8,[0,0,1]]] },
      { id:'COMP-553', authority:false, rows:[['REV-N',4,[1,1,1]],['REV-P',10,[1,1,1]],['REV-Q',16,[1,0,1]],['REV-R',10,[1,1,1]]] },
    ], authorityText: id => `Compliance officer authorization for ${id} is ${id.endsWith('550') || id.endsWith('552') ? 'signed' : 'awaiting signature'}.`,
    exception: 'A large gap count cannot justify an outdated requirement or transfer/remove control ownership.',
  },
  {
    slug: 'incident-response-mitigation', label: 'incident response mitigation sequencing', owner: 'Incident coordination cell', action: 'execute',
    metricName: 'systems restored within the approved window', unit: 'systems', direction: 'desc', count: 2,
    requirement: 'incident commander order',
    rule: 'A mitigation action qualifies only when its prerequisites are complete, its rollback has been tested, and its blast radius is within the stated incident limit. Select up to two qualifying actions restoring the most systems within the approved window, ties by ascending complete action ID. Execution requires an incident commander order.',
    conditions: [
      ['Prerequisites', 'all documented prerequisites are complete', 'at least one documented prerequisite is incomplete'],
      ['Rollback test', 'the rollback was tested in the incident environment', 'the rollback has not been tested in the incident environment'],
      ['Blast radius', 'the affected-system count stays within the incident limit', 'the affected-system count exceeds the incident limit'],
    ],
    scenarios: [
      { id:'IR-560', authority:true, rows:[['MIT-A',7,[1,1,1]],['MIT-B',9,[1,1,1]],['MIT-C',13,[1,0,1]],['MIT-D',11,[1,1,0]]] },
      { id:'IR-561', authority:false, rows:[['MIT-E',8,[1,1,1]],['MIT-F',6,[1,1,1]],['MIT-G',16,[1,0,1]],['MIT-H',12,[1,1,1]]] },
      { id:'IR-562', authority:true, rows:[['MIT-J',18,[0,1,1]],['MIT-K',15,[1,0,1]],['MIT-L',12,[1,1,0]],['MIT-M',10,[1,1,1]]] },
      { id:'IR-563', authority:false, rows:[['MIT-N',4,[1,1,1]],['MIT-P',7,[1,1,1]],['MIT-Q',19,[1,0,1]],['MIT-R',13,[1,1,0]]] },
    ], authorityText: id => `Incident commander order for ${id} is ${id.endsWith('560') || id.endsWith('562') ? 'signed' : 'not issued'}.`,
    exception: 'A high restoration count cannot waive incomplete prerequisites, an untested rollback, or an exceeded blast limit.',
  },
  {
    slug: 'evidence-claim-triage', label: 'evidence-backed claim investigation triage', owner: 'Review and assurance unit', action: 'investigate',
    metricName: 'independent corroborating sources', unit: 'sources', direction: 'desc', count: 2,
    requirement: 'review chair assignment',
    rule: 'A claim qualifies for investigation only when it is in the current scope, at least one cited source is primary, and no material contradiction remains unresolved. Select up to two qualifying claims with the most independent corroborating sources, ties by ascending complete claim ID. Opening an investigation requires a review chair assignment.',
    conditions: [
      ['Current scope', 'the claim is within the current review scope', 'the claim is outside the current review scope'],
      ['Primary source', 'at least one cited source is primary', 'all cited sources are secondary or unattributed'],
      ['Contradiction status', 'no material contradiction remains unresolved', 'a material contradiction remains unresolved'],
    ],
    scenarios: [
      { id:'TRIAGE-570', authority:true, rows:[['CLM-A',3,[1,1,1]],['CLM-B',5,[1,1,1]],['CLM-C',9,[1,0,1]],['CLM-D',8,[1,1,0]]] },
      { id:'TRIAGE-571', authority:false, rows:[['CLM-E',7,[1,1,1]],['CLM-F',7,[1,1,1]],['CLM-G',12,[1,0,1]],['CLM-H',2,[1,1,1]]] },
      { id:'TRIAGE-572', authority:true, rows:[['CLM-J',9,[0,1,1]],['CLM-K',8,[1,0,1]],['CLM-L',7,[1,1,0]],['CLM-M',3,[0,0,1]]] },
      { id:'TRIAGE-573', authority:false, rows:[['CLM-N',6,[1,1,1]],['CLM-P',4,[1,1,1]],['CLM-Q',11,[1,0,1]],['CLM-R',9,[1,1,0]]] },
    ], authorityText: id => `Review chair assignment for ${id} is ${id.endsWith('570') || id.endsWith('572') ? 'recorded' : 'not assigned'}.`,
    exception: 'A large source count cannot make an out-of-scope claim, a claim without a primary source, or an unresolved contradiction eligible.',
  },
  {
    slug: 'maintenance-work-order', label: 'maintenance work-order sequencing', owner: 'Facilities reliability team', action: 'dispatch',
    metricName: 'verified downtime hours avoided', unit: 'hours', direction: 'desc', count: 2,
    requirement: 'facilities duty manager approval',
    rule: 'A work order qualifies only when the fault was confirmed by inspection, required parts are on site, and the repair can be completed without bypassing a safety interlock. Select up to two qualifying work orders avoiding the most verified downtime hours, ties by ascending complete work-order ID. Dispatch requires facilities duty manager approval.',
    conditions: [
      ['Inspection', 'the fault was confirmed by an inspection record', 'the fault has not been confirmed by inspection'],
      ['Parts', 'all required parts are on site', 'at least one required part is not on site'],
      ['Safety interlock', 'the repair keeps every safety interlock active', 'the repair would bypass a safety interlock'],
    ],
    scenarios: [
      { id:'MAINT-580', authority:true, rows:[['WO-A',12,[1,1,1]],['WO-B',18,[1,1,1]],['WO-C',22,[1,0,1]],['WO-D',20,[1,1,0]]] },
      { id:'MAINT-581', authority:false, rows:[['WO-E',14,[1,1,1]],['WO-F',14,[1,1,1]],['WO-G',29,[1,0,1]],['WO-H',25,[1,1,1]]] },
      { id:'MAINT-582', authority:true, rows:[['WO-J',30,[0,1,1]],['WO-K',28,[1,0,1]],['WO-L',26,[1,1,0]],['WO-M',19,[1,1,1]]] },
      { id:'MAINT-583', authority:false, rows:[['WO-N',10,[1,1,1]],['WO-P',9,[1,1,1]],['WO-Q',31,[1,0,1]],['WO-R',21,[1,1,0]]] },
    ], authorityText: id => `Facilities duty manager approval for ${id} is ${id.endsWith('580') || id.endsWith('582') ? 'recorded' : 'pending'}.`,
    exception: 'A downtime estimate cannot replace inspection, parts availability, or a repair that preserves safety interlocks.',
  },
];

const worlds = [];
for (const [domainIndex, domain] of domains.entries()) {
  for (const [scenarioIndex, scenario] of domain.scenarios.entries()) {
    const count = scenarioIndex % 2 === 0 ? domain.count : 1;
    const candidates = scenario.rows.map(([id, metric, flags]) => ({
      id, metric,
      flags: Object.fromEntries(domain.conditions.map(([label], i) => [label, Boolean(flags[i])])),
      facts: domain.conditions.map(([label, yes, no], i) => `${label}: ${flags[i] ? yes : no}.`).join(' '),
    }));
    const compare = domain.direction === 'asc'
      ? (a, b) => a.metric - b.metric || a.id.localeCompare(b.id)
      : (a, b) => b.metric - a.metric || a.id.localeCompare(b.id);
    const eligible = candidate => domain.conditions.every(([label]) => candidate.flags[label]);
    const selected = candidates.filter(eligible).sort(compare).slice(0, count);
    // Some scenarios intentionally reduce a two-item domain to a one-item
    // request. Keep the authored ranking clause aligned with that scenario's
    // request and output cardinality instead of retaining the domain's
    // two-item default.
    const ruleText = count === 1
      ? domain.rule.replace('Select up to two qualifying ', 'Select exactly one qualifying ')
      : domain.rule;
    const request = `${domain.action[0].toUpperCase()}${domain.action.slice(1)} items for ${scenario.id}.`;
    const format = rows => rows.length ? rows.map(row => row.id).join('; ') : 'none';
    const measure = rows => rows.length ? rows.map(row => String(row.metric)).join('; ') : 'none';
    const instruction = `${request} Apply the stated evidence conditions before ranking. Selection is a recommendation; authority controls only the final action. If no item qualifies, use no_action.`;
    const world = caseFrom({
      slug: `${domain.slug}-${scenario.id.toLowerCase()}`,
      sourceGroup: `v35:${scenario.id}:world`, domain: domain.label,
      owner: domain.owner, requestId: scenario.id,
      requestText: `${request} ${count === 1 ? 'Select exactly one eligible item if any qualify.' : `Select up to ${count} eligible items.`}`,
      ruleText,
      selectionInstruction: `${count === 1 ? 'exactly one if any qualify' : `up to ${count}`}; ${domain.direction === 'asc' ? 'ascending' : 'descending'} ${domain.metricName}; ties by ascending complete ID`,
      authorizationRule: { requirement: domain.requirement, scope: 'request' },
      instruction, registerHeading: 'Measured candidate register', auditHeading: 'Evidence condition review', authorityHeading: 'Action authority record',
      candidates, compare, provisional: rows => rows.slice(0, count), eligible, select: rows => rows.slice(0, count), format,
      measure,
      registerLine: row => `${row.id}: ${domain.metricName} ${row.metric} ${domain.unit}.`,
      auditLine: row => `${row.id}: ${row.facts}`,
      exceptionText: domain.exception,
      authorityText: domain.authorityText(scenario.id), authorized: scenario.authority,
      selectionFormat: count === 1 ? 'One complete item ID, or none if no item qualifies.' : `Up to ${count} complete item IDs joined by exactly semicolon and one space; in ranked order; none if no item qualifies.`,
      measureFormat: count === 1 ? `${domain.metricName} for the selected item, as digits; none if no item qualifies. Unit: ${domain.unit}.` : `Corresponding ${domain.metricName} values as digits in the selected-item order, joined by exactly semicolon and one space; none if no item qualifies. Unit: ${domain.unit}.`,
      finalDecisions: [domain.action, 'hold', 'no_action'], approvedAction: domain.action, noAction: 'no_action',
    });
    world.domainIndex = domainIndex;
    world.domainGroup = `v35:${domain.slug}:domain`;
    world.split = domainIndex < 4 ? 'train' : 'test';
    world.selectionCardinality = count;
    world.domainDefaultCardinality = domain.count;
    world.scenarioFacts = candidates;
    world.scenarioAuthority = scenario.authority;
    world.authorityEvidence = domain.authorityText(scenario.id);
    world.conditionSchema = domain.conditions.map(([label, yes, no]) => ({ label, yes, no }));
    world.metricUnit = domain.unit;
    world.authorityRequirement = domain.requirement;
    const ranked = candidates.filter(eligible).sort(compare).slice(0, count);
    world.expectedFactDerivation = {
      selected: ranked.map(row => row.id),
      measure: measure(ranked),
      decision: ranked.length ? (scenario.authority ? domain.action : 'hold') : 'no_action',
    };
    worlds.push(world);
  }
}

if (worlds.length !== 32 || new Set(worlds.map(world => world.group)).size !== 32 ||
    worlds.filter(world => world.split === 'train').length !== 16 || worlds.filter(world => world.split === 'test').length !== 16)
  throw new Error('V35 requires 32 unique request groups and a balanced 16/16 split');

export { domains as sourceDomains, worlds };
export function deriveWorldResult(world) {
  const domain = domains[world.domainIndex];
  const eligible = world.scenarioFacts.filter(row => domain.conditions.every(([label]) => row.flags[label]));
  eligible.sort(domain.direction === 'asc'
    ? (a, b) => a.metric - b.metric || a.id.localeCompare(b.id)
    : (a, b) => b.metric - a.metric || a.id.localeCompare(b.id));
  const rows = eligible.slice(0, world.selectionCardinality);
  return {
    caseId: world.source_summary.requestId,
    selectedItems: rows.length ? rows.map(row => row.id).join('; ') : 'none',
    measure: rows.length ? rows.map(row => String(row.metric)).join('; ') : 'none',
    decision: rows.length ? (world.scenarioAuthority ? domain.action : 'hold') : 'no_action',
  };
}
