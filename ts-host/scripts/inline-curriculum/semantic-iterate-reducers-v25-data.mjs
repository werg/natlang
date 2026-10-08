import { caseFrom } from './semantic-iterate-reducers-v18-novel-data.mjs';
import { selectionContract } from './semantic-iterate-reducer-selection-contract.mjs';

// Sixteen separately authored requests across eight new domains. Rows carry
// exhaustive condition values; evidence prose is rendered from those values
// and checked again below before any source case is emitted.
const domains = [
  {
    slug: 'hazardous-tree-work', domain: 'municipal hazardous tree work orders', split: 'train', action: 'dispatch', count: 1,
    direction: 'desc', metricName: 'verified public-risk points', unit: 'points',
    rule: 'A work order qualifies only when the hazard is confirmed by a field crew, an active street-closure permit covers the work area, and a certified arborist is assigned. Rank qualifying work by verified public-risk points, greatest first. Dispatch requires the parks duty officer release.',
    headings: ['Tree hazard risk register', 'Field, permit, and arborist eligibility review', 'Parks duty officer release'],
    exception: 'A higher risk score does not replace hazard confirmation, a location-matched closure permit, or an assigned certified arborist.',
    conditions: [
      { label: 'Hazard status', yes: 'the field crew confirmed an unstable limb over a public path', no: 'the field crew found no unstable limb over a public path' },
      { label: 'Closure permit', yes: 'permit covers the listed work location', no: 'permit covers a different street block' },
      { label: 'Arborist assignment', yes: 'a certified arborist is assigned', no: 'no certified arborist is assigned' },
    ],
    scenarios: [
      { id: 'TREE-710', authorized: true, authority: 'Parks duty officer PO-4 released TREE-710 for dispatch.', rows: [
        ['TREE-A1', 91, [true,true,true]], ['TREE-A2', 99, [true,false,true]], ['TREE-A3', 84, [true,true,true]],
      ] },
      { id: 'TREE-711', authorized: false, authority: 'No parks duty officer release is recorded for TREE-711.', rows: [
        ['TREE-B1', 88, [true,true,true]], ['TREE-B3', 97, [true,true,false]], ['TREE-B5', 76, [true,true,false]],
      ] },
    ],
  },
  {
    slug: 'vaccine-cold-chain-routing', domain: 'regional vaccine cold-chain shipment routing', split: 'train', action: 'dispatch', count: 2,
    direction: 'desc', metricName: 'sealed dose units available', unit: 'doses',
    rule: 'A shipment qualifies only when its temperature log stays within 2–8 degrees Celsius, the receiving pharmacy is approved, and custody seals are intact. Rank qualifying shipments by sealed dose units available, greatest first. Dispatch requires the regional pharmacist release.',
    headings: ['Vaccine shipment inventory', 'Temperature, receiver, and seal verification', 'Regional pharmacist release'],
    exception: 'A large shipment cannot waive a temperature excursion, an unapproved receiving pharmacy, or a broken custody seal.',
    conditions: [
      { label: 'Temperature log', yes: 'all readings stayed within 2–8 degrees Celsius', no: 'one reading reached 9 degrees Celsius' },
      { label: 'Receiving pharmacy', yes: 'receiver approval is current', no: 'receiver approval has expired' },
      { label: 'Custody seal', yes: 'both custody seals are intact', no: 'the outer custody seal is broken' },
    ],
    scenarios: [
      { id: 'VAX-720', authorized: true, authority: 'Regional pharmacist RP-2 released VAX-720.', rows: [
        ['SHIP-C4', 2400, [true,true,true]], ['SHIP-C7', 3100, [true,true,true]], ['SHIP-C9', 4200, [false,true,true]],
      ] },
      { id: 'VAX-721', authorized: false, authority: 'Regional pharmacist release for VAX-721 is pending.', rows: [
        ['SHIP-D1', 1800, [true,true,true]], ['SHIP-D2', 2100, [true,false,true]], ['SHIP-D6', 1950, [true,true,true]],
      ] },
    ],
  },
  {
    slug: 'fisheries-survey-vessels', domain: 'coastal fisheries survey vessel assignment', split: 'train', action: 'assign', count: 1,
    direction: 'desc', metricName: 'survey stations covered per day', unit: 'stations/day',
    rule: 'A vessel qualifies only when its safety inspection is current, a credentialed observer is assigned, and the harbor weather desk clears its route. Rank qualifying vessels by survey stations covered per day, greatest first. Assignment requires a current fisheries permit.',
    headings: ['Survey vessel coverage register', 'Inspection, observer, and route clearance audit', 'Fisheries permit status'],
    exception: 'Projected station coverage cannot replace a current inspection, credentialed observer, or route clearance.',
    conditions: [
      { label: 'Vessel inspection', yes: 'inspection certificate remains current', no: 'inspection certificate expired' },
      { label: 'Observer credential', yes: 'a credentialed observer is assigned', no: 'observer credential is not current' },
      { label: 'Route clearance', yes: 'harbor weather desk cleared the route', no: 'harbor weather desk withheld route clearance' },
    ],
    scenarios: [
      { id: 'FISH-730', authorized: true, authority: 'Permit FP-73 is current for FISH-730.', rows: [
        ['VESSEL-MARLIN', 14, [true,true,true]], ['VESSEL-TERN', 18, [true,true,true]], ['VESSEL-REEF', 23, [true,false,true]],
      ] },
      { id: 'FISH-731', authorized: false, authority: 'Permit renewal for FISH-731 has not been signed.', rows: [
        ['VESSEL-ASH', 12, [true,false,true]], ['VESSEL-COVE', 17, [true,true,false]], ['VESSEL-DELTA', 15, [false,true,true]],
      ] },
    ],
  },
  {
    slug: 'rail-signal-repair', domain: 'regional rail signal repair prioritization', split: 'train', action: 'repair', count: 1,
    direction: 'desc', metricName: 'signal restoration priority grade on the bounded 1–5 ordinal scale (not additive)', unit: 'grade points',
    rule: 'A repair qualifies only when track isolation is verified, replacement parts are staged, and a signal engineer is assigned. Rank by the bounded 1–5 ordinal restoration grade, highest grade first; the grade is ordinal and must not be added or averaged. Repair requires the rail operations controller authorization.',
    headings: ['Signal restoration grade register', 'Isolation, parts, and engineer readiness', 'Rail operations controller authorization'],
    exception: 'The 1–5 grade is a bounded ordinal category, not a quantity; do not sum, average, or extend its scale.',
    conditions: [
      { label: 'Track isolation', yes: 'track isolation is verified for this signal section', no: 'track isolation has not been verified' },
      { label: 'Replacement parts', yes: 'the matching signal module is staged', no: 'the matching signal module is not staged' },
      { label: 'Engineer assignment', yes: 'a qualified signal engineer is assigned', no: 'no qualified signal engineer is assigned' },
    ],
    scenarios: [
      { id: 'RAIL-740', authorized: true, authority: 'Controller RC-8 authorized RAIL-740.', rows: [
        ['SIG-E12', 4, [true,true,true]], ['SIG-E15', 3, [true,true,true]], ['SIG-E19', 5, [true,false,true]],
      ] },
      { id: 'RAIL-741', authorized: false, authority: 'Controller authorization for RAIL-741 is not recorded.', rows: [
        ['SIG-F02', 3, [true,true,true]], ['SIG-F06', 5, [false,true,true]], ['SIG-F09', 2, [true,true,false]],
      ] },
    ],
  },
  {
    slug: 'floodgate-inspection', domain: 'river floodgate inspection queue', split: 'test', action: 'schedule', count: 1,
    direction: 'desc', metricName: 'hours past the scheduled inspection time', unit: 'hours',
    rule: 'A gate qualifies for inspection when its access road is open, the inspection crew has current confined-space certification, and the gate is isolated from remote operation. Rank by hours past the scheduled inspection time, greatest first. Scheduling requires the basin supervisor assignment.',
    headings: ['Floodgate overdue-hour register', 'Access, certification, and isolation audit', 'Basin supervisor assignment'],
    exception: 'Time overdue does not make a gate accessible, certify a crew, or isolate remote controls.',
    conditions: [
      { label: 'Access road', yes: 'the access road is open to the inspection vehicle', no: 'the access road is closed by standing water' },
      { label: 'Crew certification', yes: 'the assigned crew holds current confined-space certification', no: 'the assigned crew certification has expired' },
      { label: 'Remote isolation', yes: 'remote operation is locked out for inspection', no: 'remote operation remains enabled' },
    ],
    scenarios: [
      { id: 'GATE-750', authorized: true, authority: 'Basin supervisor BS-3 assigned the inspection team for GATE-750.', rows: [
        ['GATE-NORTH', 27, [true,true,true]], ['GATE-EAST', 39, [false,true,true]], ['GATE-WEST', 18, [true,true,true]],
      ] },
      { id: 'GATE-751', authorized: false, authority: 'No basin supervisor assignment is recorded for GATE-751.', rows: [
        ['GATE-CEDAR', 21, [true,true,true]], ['GATE-MARSH', 34, [true,false,true]], ['GATE-LOWER', 14, [true,true,true]],
      ] },
    ],
  },
  {
    slug: 'emergency-tuition-bursary', domain: 'university emergency tuition bursary review', split: 'test', action: 'award', count: 2,
    direction: 'desc', metricName: 'verified unmet tuition credits', unit: 'credits',
    rule: 'An application qualifies only when the student is currently enrolled, financial documents verify the stated unmet tuition, and no duplicate emergency award has been paid for the term. Rank qualifying applications by verified unmet tuition credits, greatest first. Award requires the bursary committee approval.',
    headings: ['Unmet tuition credit register', 'Enrollment, financial documents, and duplicate award check', 'Bursary committee approval'],
    exception: 'A high unmet balance cannot replace current enrollment, verified financial documents, or the duplicate-award check.',
    conditions: [
      { label: 'Enrollment', yes: 'registrar confirms active enrollment this term', no: 'registrar confirms enrollment ended before this term' },
      { label: 'Financial documents', yes: 'required financial documents verify the unmet tuition', no: 'required financial documents are incomplete' },
      { label: 'Duplicate award', yes: 'no emergency award has been paid for this term', no: 'an emergency award was already paid for this term' },
    ],
    scenarios: [
      { id: 'BURS-760', authorized: true, authority: 'Bursary committee BC-5 approved awards under BURS-760.', rows: [
        ['APP-41', 1850, [true,true,true]], ['APP-44', 2400, [true,true,true]], ['APP-48', 3100, [true,false,true]],
      ] },
      { id: 'BURS-761', authorized: false, authority: 'Committee approval for BURS-761 is pending.', rows: [
        ['APP-52', 1700, [true,true,true]], ['APP-57', 2200, [true,true,true]], ['APP-59', 2500, [true,true,false]],
      ] },
    ],
  },
  {
    slug: 'data-center-patch-batch', domain: 'regional data-center security patch batching', split: 'test', action: 'patch', count: 2,
    direction: 'desc', metricName: 'active exposed nodes', unit: 'nodes',
    rule: 'A patch qualifies for the batch only when staging tests pass, rollback is validated, and vendor support remains active. Rank qualifying patches by the number of active exposed nodes addressed, greatest first. Applying the batch requires a signed change window.',
    headings: ['Exposed-node patch register', 'Staging, rollback, and support verification', 'Change-window approval record'],
    exception: 'A patch addressing more nodes cannot waive failed staging tests, unvalidated rollback, or expired vendor support.',
    conditions: [
      { label: 'Staging tests', yes: 'all staging tests passed', no: 'one staging test failed' },
      { label: 'Rollback plan', yes: 'rollback was tested and validated', no: 'rollback has not been tested' },
      { label: 'Vendor support', yes: 'vendor support is active through deployment', no: 'vendor support expired before deployment' },
    ],
    scenarios: [
      { id: 'PATCH-770', authorized: true, authority: 'Change window CW-77 is signed for PATCH-770.', rows: [
        ['PATCH-KR4', 146, [true,true,true]], ['PATCH-LM2', 183, [true,true,true]], ['PATCH-QS8', 211, [true,false,true]],
      ] },
      { id: 'PATCH-771', authorized: false, authority: 'No signed change window covers PATCH-771.', rows: [
        ['PATCH-RT1', 132, [true,true,true]], ['PATCH-VB5', 165, [true,true,true]], ['PATCH-XC3', 199, [false,true,true]],
      ] },
    ],
  },
  {
    slug: 'museum-folio-digitization', domain: 'museum manuscript folio digitization queue', split: 'test', action: 'digitize', count: 2,
    direction: 'desc', metricName: 'catalogued folios ready for capture', unit: 'folios',
    rule: 'A manuscript qualifies for capture only when reproduction rights are cleared, conservation preparation is complete, and its metadata schema is validated. Rank qualifying manuscripts by catalogued folios ready for capture, greatest first. Beginning capture requires the collections director release.',
    headings: ['Ready-folio count register', 'Rights, conservation, and metadata validation', 'Collections director release'],
    exception: 'A larger folio count cannot substitute for cleared rights, completed conservation preparation, or validated metadata.',
    conditions: [
      { label: 'Reproduction rights', yes: 'reproduction rights cover internal digitization', no: 'reproduction rights remain under review' },
      { label: 'Conservation preparation', yes: 'conservation preparation is complete', no: 'conservation preparation is incomplete' },
      { label: 'Metadata schema', yes: 'required metadata fields passed validation', no: 'a required metadata field failed validation' },
    ],
    scenarios: [
      { id: 'MUSE-780', authorized: true, authority: 'Collections director CD-2 released MUSE-780 for capture.', rows: [
        ['MS-ASTER', 124, [true,true,true]], ['MS-BIRCH', 96, [true,true,true]], ['MS-CEDAR', 172, [true,false,true]],
      ] },
      { id: 'MUSE-781', authorized: false, authority: 'Collections director release for MUSE-781 is pending.', rows: [
        ['MS-DOVE', 88, [true,true,true]], ['MS-ELM', 131, [true,true,true]], ['MS-FERN', 209, [true,false,true]],
      ] },
    ],
  },
];

const output = [];
for (const [domainIndex, domain] of domains.entries()) {
  for (const scenario of domain.scenarios) {
    const candidates = scenario.rows.map(([id, metric, flags]) => ({
      id, metric,
      facts: domain.conditions.map((condition, index) => `${condition.label}: ${flags[index] ? condition.yes : condition.no}.`).join(' '),
      flags: Object.fromEntries(domain.conditions.map((condition, index) => [condition.label, flags[index]])),
    }));
    const contract = selectionContract({ count: domain.count, direction: domain.direction, measureName: domain.metricName });
    const compare = (a, b) => (domain.direction === 'asc' ? a.metric - b.metric : b.metric - a.metric) || a.id.localeCompare(b.id);
    const eligible = candidate => domain.conditions.every(condition => candidate.flags[condition.label]);
    const select = values => values.slice(0, domain.count);
    const measure = values => values.length ? values.map(value => String(value.metric)).join('; ') : 'none';
    const sourceGroup = `v25:${scenario.id}:world`;
    const request = `${domain.action[0].toUpperCase()}${domain.action.slice(1)} candidates for request ${scenario.id}.`;
    const ruleText = `${domain.rule} Selection cardinality: ${contract.instruction} Decision mapping: if no item qualifies, use no_action; if a nonempty selection has the required recorded authority, use ${domain.action}; otherwise use hold. Authority does not alter eligibility or ranking.`;
    const spec = {
      slug: scenario.id.toLowerCase(), sourceGroup, domain: domain.domain, owner: `${domain.domain} operations desk`,
      requestId: scenario.id, direction: domain.direction,
      requestText: `${request} ${contract.instruction}`, ruleText,
      instruction: `${request} ${contract.instruction}`,
      registerHeading: domain.headings[0], auditHeading: domain.headings[1], authorityHeading: domain.headings[2],
      candidates,
      compare,
      provisional: values => select([...values].sort(compare)),
      eligible,
      select: values => select([...values].sort(compare)),
      format: values => values.length ? values.map(value => value.id).join('; ') : 'none',
      measure,
      registerLine: value => `${value.id}: ${domain.metricName} ${value.metric} ${domain.unit}.`,
      auditLine: value => `${value.id}: ${value.facts}`,
      exceptionText: domain.exception,
      authorityText: scenario.authority,
      authorized: scenario.authorized,
      selectionFormat: contract.selectionFormat,
      measureFormat: contract.measureFormat,
      decisionFormat: `Use ${domain.action} only when a nonempty eligible selection has the recorded authorization; otherwise use hold. Use no_action only when no item qualifies. Return one bare literal: ${domain.action}, hold, no_action.`,
      finalDecisions: [domain.action, 'hold', 'no_action'], approvedAction: domain.action, noAction: 'no_action',
    };
    const world = caseFrom(spec);
    world.domainIndex = domainIndex;
    world.domainGroup = `v25:${domain.slug}:domain`;
    world.split = domain.split;
    world.selectionCardinality = domain.count;
    world.scenarioFacts = candidates;
    world.scenarioAuthority = scenario.authorized;
    world.conditionSchema = domain.conditions;
    world.metricUnit = domain.unit;
    world.expectedFactDerivation = {
      selected: world.source_summary.selectedItems === 'none' ? [] : world.source_summary.selectedItems.split('; '),
      measure: world.source_summary.measure,
      decision: world.source_summary.decision,
    };
    output.push(world);
  }
}

if (output.length !== 16 || new Set(output.map(world => world.group)).size !== 16)
  throw new Error('V25 requires 16 separately authored factual request groups');
if (output.filter(world => world.split === 'train').length !== 8 || output.filter(world => world.split === 'test').length !== 8)
  throw new Error('V25 requires an 8/8 split with each domain pair kept together');

for (const world of output) {
  const domain = domains[world.domainIndex];
  for (const candidate of world.scenarioFacts) {
    const expectedFacts = domain.conditions.map(condition =>
      `${condition.label}: ${candidate.flags[condition.label] ? condition.yes : condition.no}.`);
    for (const sentence of expectedFacts) if (!candidate.facts.includes(sentence))
      throw new Error(`${world.slug}/${candidate.id}: missing exhaustive ${sentence}`);
    const registerLine = domain && `${candidate.id}: ${domain.metricName} ${candidate.metric} ${domain.unit}.`;
    if (!Number.isInteger(candidate.metric) || !registerLine.includes(String(candidate.metric)) || !registerLine.includes(domain.unit))
      throw new Error(`${world.slug}/${candidate.id}: metric evidence omits exact value or unit`);
  }
  const selected = world.scenarioFacts.filter(candidate => domain.conditions.every(condition => candidate.flags[condition.label]))
    .sort((a, b) => (domain.direction === 'asc' ? a.metric - b.metric : b.metric - a.metric) || a.id.localeCompare(b.id))
    .slice(0, domain.count);
  const derived = {
    caseId: world.source_summary.requestId,
    selectedItems: selected.length ? selected.map(candidate => candidate.id).join('; ') : 'none',
    measure: selected.length ? selected.map(candidate => String(candidate.metric)).join('; ') : 'none',
    decision: selected.length ? (world.scenarioAuthority ? domain.action : 'hold') : 'no_action',
  };
  const authored = { caseId: world.source_summary.requestId, selectedItems: world.source_summary.selectedItems,
    measure: world.source_summary.measure, decision: world.source_summary.decision };
  if (JSON.stringify(derived) !== JSON.stringify(authored)) throw new Error(`${world.slug}: gold differs from exhaustive facts`);
}

export const worlds = output;
export const sourceDomains = domains.map((domain, index) => ({
  slug: domain.slug, domain: domain.domain, domain_group: `v25:${domain.slug}:domain`,
  split: domain.split, world_count: 2, domain_index: index,
}));

export function deriveWorldResult(world) {
  const domain = domains[world.domainIndex];
  const selected = world.scenarioFacts.filter(candidate => domain.conditions.every(condition => candidate.flags[condition.label]))
    .sort((a, b) => (domain.direction === 'asc' ? a.metric - b.metric : b.metric - a.metric) || a.id.localeCompare(b.id))
    .slice(0, domain.count);
  return {
    caseId: world.source_summary.requestId,
    selectedItems: selected.length ? selected.map(candidate => candidate.id).join('; ') : 'none',
    measure: selected.length ? selected.map(candidate => String(candidate.metric)).join('; ') : 'none',
    decision: selected.length ? (world.scenarioAuthority ? domain.action : 'hold') : 'no_action',
  };
}
