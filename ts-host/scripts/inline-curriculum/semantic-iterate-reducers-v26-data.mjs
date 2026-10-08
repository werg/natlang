import { caseFrom } from './semantic-iterate-reducers-v18-novel-data.mjs';
import { selectionContract } from './semantic-iterate-reducer-selection-contract.mjs';

// New fictional source worlds for V26. Each domain pair stays in one split;
// every candidate carries an exhaustive Boolean audit against the conditions.
const domains = [
  {
    slug:'editorial-corrections', domain:'editorial correction triage', split:'train', action:'revise', count:1,
    direction:'desc', metricName:'verified reader-impact reports', unit:'reports',
    rule:'A correction qualifies when the cited text is present in the current edition, the proposed correction is supported by a primary source, and the change does not alter an approved quotation. Rank eligible corrections by verified reader-impact reports, greatest first. Revision requires an editor-in-chief sign-off.',
    authorityRequirement:'an editor-in-chief sign-off',
    headings:['Correction impact register','Edition, source, and quotation audit','Editorial sign-off record'],
    exception:'High reader impact cannot establish a source citation or permit changes to an approved quotation.',
    conditions:[
      {label:'Edition match',yes:'the cited passage appears in the current edition',no:'the cited passage appears only in an archived edition'},
      {label:'Primary source',yes:'a primary source supports the proposed correction',no:'only an unverified secondary report supports it'},
      {label:'Quotation boundary',yes:'the correction leaves the approved quotation unchanged',no:'the correction would alter the approved quotation'},
    ],
    scenarios:[
      {id:'EDIT-810',authorized:true,authority:'Editor-in-chief EC-4 signed EDIT-810.',rows:[['CORR-A',31,[true,true,true]],['CORR-B',45,[true,false,true]],['CORR-C',24,[true,true,true]]]},
      {id:'EDIT-811',authorized:false,authority:'No editor-in-chief sign-off is recorded for EDIT-811.',rows:[['CORR-D',18,[true,true,true]],['CORR-E',27,[false,true,true]],['CORR-F',12,[true,false,true]]]},
    ],
  },
  {
    slug:'archive-provenance', domain:'archive provenance review', split:'train', action:'certify', count:2,
    direction:'desc', metricName:'verified provenance links', unit:'links',
    rule:'A folder qualifies when its transfer record is signed, each listed source institution is identified, and no chain-of-custody gap remains. Rank qualifying folders by verified provenance links, greatest first. Certification requires an archivist-of-record approval.',
    authorityRequirement:'an archivist-of-record approval',
    headings:['Provenance-link register','Transfer, source, and custody audit','Archivist approval record'],
    exception:'A large link count does not repair a missing transfer signature or custody gap.',
    conditions:[
      {label:'Transfer signature',yes:'the transfer record has the donor and receiving archivist signatures',no:'the receiving archivist signature is missing'},
      {label:'Source identity',yes:'every listed institution has a verified name',no:'one listed institution is unidentified'},
      {label:'Custody chain',yes:'the custody ledger has no unexplained gap',no:'the custody ledger has an unexplained gap'},
    ],
    scenarios:[
      {id:'ARCH-820',authorized:true,authority:'Archivist of record AR-2 approved ARCH-820.',rows:[['FOLIO-1',42,[true,true,true]],['FOLIO-2',57,[true,true,true]],['FOLIO-3',63,[true,true,false]]]},
      {id:'ARCH-821',authorized:false,authority:'Archivist approval for ARCH-821 is pending.',rows:[['FOLIO-4',51,[true,true,true]],['FOLIO-5',38,[true,true,true]],['FOLIO-6',66,[true,false,true]]]},
    ],
  },
  {
    slug:'grant-dossier-review', domain:'grant dossier review queue', split:'train', action:'advance', count:2,
    direction:'desc', metricName:'verified required attachments', unit:'attachments',
    rule:'A dossier qualifies when the budget form is signed, the project lead is eligible, and the conflict disclosure is current. Rank qualifying dossiers by verified required attachments, greatest first. Advancing dossiers requires a grants director release.',
    authorityRequirement:'a grants director release',
    headings:['Attachment completeness register','Budget, eligibility, and conflict review','Grants director release record'],
    exception:'A complete attachment count cannot waive an ineligible project lead or outdated conflict disclosure.',
    conditions:[
      {label:'Budget form',yes:'the budget form is signed by the applicant',no:'the budget form lacks the applicant signature'},
      {label:'Project lead',yes:'the project lead meets the current eligibility rule',no:'the project lead does not meet the current eligibility rule'},
      {label:'Conflict disclosure',yes:'the conflict disclosure is current for this cycle',no:'the conflict disclosure expired last cycle'},
    ],
    scenarios:[
      {id:'GRANT-830',authorized:true,authority:'Grants director GD-9 released GRANT-830.',rows:[['DOSSIER-A',16,[true,true,true]],['DOSSIER-B',21,[true,true,true]],['DOSSIER-C',24,[false,true,true]]]},
      {id:'GRANT-831',authorized:false,authority:'No grants director release is recorded for GRANT-831.',rows:[['DOSSIER-D',18,[true,true,true]],['DOSSIER-E',22,[true,false,true]],['DOSSIER-F',25,[true,true,false]]]},
    ],
  },
  {
    slug:'public-record-redactions', domain:'public-record redaction review', split:'train', action:'release', count:2,
    direction:'asc', metricName:'redaction review sequence number', unit:'sequence number',
    rule:'A record qualifies when the request is active, every protected identifier is redacted, and the remaining text passes disclosure review. Rank qualifying records by redaction review sequence number, lowest first. Release requires a records officer authorization.',
    authorityRequirement:'a records officer authorization',
    headings:['Redaction sequence register','Request, identifier, and disclosure audit','Records officer authorization'],
    exception:'An active request cannot waive identifier redaction or disclosure review.',
    conditions:[
      {label:'Request status',yes:'the public-record request is active',no:'the public-record request has been withdrawn'},
      {label:'Protected identifiers',yes:'all protected identifiers are redacted',no:'one protected identifier remains visible'},
      {label:'Disclosure review',yes:'the remaining text passed disclosure review',no:'the remaining text has not passed disclosure review'},
    ],
    scenarios:[
      {id:'RECORD-840',authorized:true,authority:'Records officer RO-6 authorized RECORD-840.',rows:[['DOC-A',4,[true,true,true]],['DOC-B',2,[true,true,true]],['DOC-C',1,[true,false,true]]]},
      {id:'RECORD-841',authorized:false,authority:'Records officer authorization for RECORD-841 is not recorded.',rows:[['DOC-D',3,[true,false,true]],['DOC-E',1,[false,true,true]],['DOC-F',5,[true,true,false]]]},
    ],
  },
  {
    slug:'translation-quality-routing', domain:'translation quality review routing', split:'test', action:'route', count:1,
    direction:'desc', metricName:'verified high-impact terminology issues', unit:'issues',
    rule:'A translation qualifies when the source version is current, a bilingual reviewer confirmed the terminology issue, and the target text preserves the approved meaning. Rank qualifying translations by verified high-impact terminology issues, greatest first. Routing requires a localization lead assignment.',
    authorityRequirement:'a localization lead assignment',
    headings:['Terminology impact register','Source, reviewer, and meaning audit','Localization lead assignment'],
    exception:'A high issue count cannot replace reviewer confirmation or permit a change in approved meaning.',
    conditions:[
      {label:'Source version',yes:'the translation uses the current source version',no:'the translation uses a superseded source version'},
      {label:'Reviewer confirmation',yes:'a bilingual reviewer confirmed the terminology issue',no:'the terminology issue is unconfirmed'},
      {label:'Approved meaning',yes:'the target wording preserves the approved meaning',no:'the target wording changes the approved meaning'},
    ],
    scenarios:[
      {id:'LANG-850',authorized:true,authority:'Localization lead LL-3 assigned LANG-850.',rows:[['LOCALE-A',11,[true,true,true]],['LOCALE-B',17,[true,true,true]],['LOCALE-C',23,[false,true,true]]]},
      {id:'LANG-851',authorized:false,authority:'No localization lead assignment is recorded for LANG-851.',rows:[['LOCALE-D',8,[true,true,true]],['LOCALE-E',14,[true,false,true]],['LOCALE-F',20,[true,true,false]]]},
    ],
  },
  {
    slug:'museum-accession-catalog', domain:'museum accession catalog updates', split:'test', action:'publish', count:2,
    direction:'desc', metricName:'validated catalog fields', unit:'fields',
    rule:'An accession qualifies when its deed of gift is on file, the object identifier matches the intake register, and required descriptive fields pass catalog validation. Rank qualifying accessions by validated catalog fields, greatest first. Publishing requires a collections registrar approval.',
    authorityRequirement:'a collections registrar approval',
    headings:['Catalog field validation register','Gift, identifier, and metadata audit','Collections registrar approval'],
    exception:'More validated fields cannot cure a missing deed or mismatched object identifier.',
    conditions:[
      {label:'Gift deed',yes:'the signed deed of gift is on file',no:'the signed deed of gift is missing'},
      {label:'Object identifier',yes:'the object identifier matches the intake register',no:'the object identifier differs from the intake register'},
      {label:'Required metadata',yes:'all required descriptive fields pass validation',no:'one required descriptive field fails validation'},
    ],
    scenarios:[
      {id:'MUSE-860',authorized:true,authority:'Collections registrar CR-5 approved MUSE-860.',rows:[['ACC-A',36,[true,true,true]],['ACC-B',41,[true,true,true]],['ACC-C',49,[true,false,true]]]},
      {id:'MUSE-861',authorized:false,authority:'Collections registrar approval for MUSE-861 is pending.',rows:[['ACC-D',28,[true,true,true]],['ACC-E',44,[true,true,true]],['ACC-F',53,[true,true,false]]]},
    ],
  },
  {
    slug:'policy-citation-repair', domain:'policy memorandum citation repair', split:'test', action:'correct', count:2,
    direction:'desc', metricName:'verified citation gaps', unit:'gaps',
    rule:'A memorandum qualifies when its current draft is under review, the cited authority remains in force, and a reviewer verified the citation gap. Rank qualifying memoranda by verified citation gaps, greatest first. Corrections require a policy counsel sign-off.',
    authorityRequirement:'a policy counsel sign-off',
    headings:['Citation gap register','Draft, authority, and reviewer audit','Policy counsel sign-off record'],
    exception:'A citation gap to a repealed authority is not repaired by adding the old citation.',
    conditions:[
      {label:'Draft status',yes:'the memorandum draft is the current review version',no:'the memorandum draft has been superseded'},
      {label:'Authority status',yes:'the cited authority remains in force',no:'the cited authority was repealed'},
      {label:'Reviewer evidence',yes:'a policy reviewer verified the citation gap',no:'the reported citation gap is unverified'},
    ],
    scenarios:[
      {id:'POLICY-870',authorized:true,authority:'Policy counsel PC-8 signed POLICY-870.',rows:[['MEMO-A',7,[true,true,true]],['MEMO-B',9,[true,true,true]],['MEMO-C',13,[true,false,true]]]},
      {id:'POLICY-871',authorized:false,authority:'Policy counsel sign-off for POLICY-871 is pending.',rows:[['MEMO-D',5,[true,true,true]],['MEMO-E',8,[true,true,true]],['MEMO-F',12,[true,true,false]]]},
    ],
  },
  {
    slug:'procurement-specification-clarifications', domain:'procurement specification clarification queue', split:'test', action:'clarify', count:1,
    direction:'desc', metricName:'verified bid-impact questions', unit:'questions',
    rule:'A clarification qualifies when it addresses an ambiguous published specification, the answer can be supported by the approved requirements, and the response does not favor one bidder. Rank qualifying clarifications by verified bid-impact questions, greatest first. Issuing a clarification requires a procurement officer release.',
    authorityRequirement:'a procurement officer release',
    headings:['Bid-impact question register','Published text, requirements, and fairness audit','Procurement officer release'],
    exception:'A bid-impact concern cannot justify an unsupported answer or bidder-specific advantage.',
    conditions:[
      {label:'Published ambiguity',yes:'the published specification contains the cited ambiguity',no:'the published specification is clear on this point'},
      {label:'Requirements support',yes:'the approved requirements support the proposed answer',no:'the proposed answer is not supported by approved requirements'},
      {label:'Equal treatment',yes:'the response applies equally to all bidders',no:'the response would favor one bidder'},
    ],
    scenarios:[
      {id:'PROC-880',authorized:true,authority:'Procurement officer PO-7 released PROC-880.',rows:[['Q-A',18,[true,true,true]],['Q-B',26,[true,true,true]],['Q-C',35,[true,false,true]]]},
      {id:'PROC-881',authorized:false,authority:'No procurement officer release is recorded for PROC-881.',rows:[['Q-D',14,[true,true,true]],['Q-E',19,[true,true,true]],['Q-F',32,[false,true,true]]]},
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
    const contract = selectionContract({ count:domain.count, direction:domain.direction, measureName:domain.metricName });
    const compare = (a, b) => (domain.direction === 'asc' ? a.metric - b.metric : b.metric - a.metric) || a.id.localeCompare(b.id);
    const eligible = candidate => domain.conditions.every(condition => candidate.flags[condition.label]);
    const select = values => values.slice(0, domain.count);
    const request = `${domain.action[0].toUpperCase()}${domain.action.slice(1)} candidates for request ${scenario.id}.`;
    const world = caseFrom({
      slug:scenario.id.toLowerCase(), sourceGroup:`v26:${scenario.id}:world`, domain:domain.domain,
      owner:`${domain.domain} operations desk`, requestId:scenario.id,
      requestText:`${request} ${contract.instruction}`,
      ruleText:domain.rule,
      selectionInstruction:contract.instruction,
      authorizationRule:{ requirement:domain.authorityRequirement, scope:'request' },
      instruction:`${request} ${contract.instruction}`,
      registerHeading:domain.headings[0], auditHeading:domain.headings[1], authorityHeading:domain.headings[2],
      candidates, compare, provisional:values=>select([...values].sort(compare)), eligible,
      select:values=>select([...values].sort(compare)),
      format:values=>values.length ? values.map(value=>value.id).join('; ') : 'none',
      measure:values=>values.length ? values.map(value=>String(value.metric)).join('; ') : 'none',
      registerLine:value=>`${value.id}: ${domain.metricName} ${value.metric} ${domain.unit}.`,
      auditLine:value=>`${value.id}: ${value.facts}`,
      exceptionText:domain.exception, authorityText:scenario.authority, authorized:scenario.authorized,
      selectionFormat:contract.selectionFormat, measureFormat:`${contract.measureFormat} Values are digits only; unit: ${domain.unit}.`,
      finalDecisions:[domain.action,'hold','no_action'], approvedAction:domain.action, noAction:'no_action',
    });
    world.domainIndex=domainIndex;
    world.domainGroup=`v26:${domain.slug}:domain`;
    world.split=domain.split;
    world.selectionCardinality=domain.count;
    world.scenarioFacts=candidates;
    world.scenarioAuthority=scenario.authorized;
    world.authorityEvidence=scenario.authority;
    world.conditionSchema=domain.conditions;
    world.metricUnit=domain.unit;
    world.authorityRequirement=domain.authorityRequirement;
    world.expectedFactDerivation={
      selected:world.source_summary.selectedItems==='none'?[]:world.source_summary.selectedItems.split('; '),
      measure:world.source_summary.measure, decision:world.source_summary.decision,
    };
    output.push(world);
  }
}

if (output.length!==16 || new Set(output.map(world=>world.group)).size!==16)
  throw new Error('V26 requires 16 separately authored factual request groups');
if (output.filter(world=>world.split==='train').length!==8 || output.filter(world=>world.split==='test').length!==8)
  throw new Error('V26 requires an 8/8 split with each domain pair kept together');

for (const world of output) {
  const domain=domains[world.domainIndex];
  for (const candidate of world.scenarioFacts) {
    for (const condition of domain.conditions) {
      const sentence=`${condition.label}: ${candidate.flags[condition.label] ? condition.yes : condition.no}.`;
      if (!candidate.facts.includes(sentence)) throw new Error(`${world.slug}/${candidate.id}: missing exhaustive ${sentence}`);
    }
    if (!Number.isInteger(candidate.metric) || !`${candidate.metric} ${domain.unit}`.includes(domain.unit))
      throw new Error(`${world.slug}/${candidate.id}: metric must be an exact integer with unit`);
  }
  const selected=world.scenarioFacts.filter(row=>domain.conditions.every(condition=>row.flags[condition.label]))
    .sort((a,b)=>(domain.direction==='asc'?a.metric-b.metric:b.metric-a.metric)||a.id.localeCompare(b.id))
    .slice(0,domain.count);
  const expected={caseId:world.source_summary.requestId,
    selectedItems:selected.length?selected.map(row=>row.id).join('; '):'none',
    measure:selected.length?selected.map(row=>String(row.metric)).join('; '):'none',
    decision:selected.length?(world.scenarioAuthority?domain.action:'hold'):'no_action'};
  const actual={caseId:world.source_summary.requestId,selectedItems:world.source_summary.selectedItems,
    measure:world.source_summary.measure,decision:world.source_summary.decision};
  if (JSON.stringify(expected)!==JSON.stringify(actual)) throw new Error(`${world.slug}: gold differs from exhaustive facts`);
}

export const worlds=output;
export const sourceDomains=domains.map((domain,index)=>({
  slug:domain.slug,domain:domain.domain,domain_group:`v26:${domain.slug}:domain`,split:domain.split,
  world_count:2,domain_index:index,
}));
export function deriveWorldResult(world) {
  const domain=domains[world.domainIndex];
  const selected=world.scenarioFacts.filter(row=>domain.conditions.every(condition=>row.flags[condition.label]))
    .sort((a,b)=>(domain.direction==='asc'?a.metric-b.metric:b.metric-a.metric)||a.id.localeCompare(b.id))
    .slice(0,domain.count);
  return {caseId:world.source_summary.requestId,
    selectedItems:selected.length?selected.map(row=>row.id).join('; '):'none',
    measure:selected.length?selected.map(row=>String(row.metric)).join('; '):'none',
    decision:selected.length?(world.scenarioAuthority?domain.action:'hold'):'no_action'};
}
