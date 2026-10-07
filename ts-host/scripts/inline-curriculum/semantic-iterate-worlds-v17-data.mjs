// New, authored decision data for V17. Each scenario has independent source facts;
// four scoped files expose request, ranked options, eligibility, and authorization.
const domains = [
  { slug:'accessible_bus_dispatch', label:'accessible bus dispatch', item:'bus run', id:'ABD', scope:'Transit accessibility desk', requirement:'Dispatch only a step-free vehicle with current corridor clearance and a passed ramp inspection.', passFact:'is a step-free vehicle; corridor clearance is current; ramp inspection passed', failFact:'uses a vehicle whose ramp inspection is expired or whose corridor clearance is missing' },
  { slug:'clinic_cold_chain_release', label:'clinic cold-chain release', item:'vaccine lot', id:'CCR', scope:'Clinic pharmacy', requirement:'Release only lots with a corrected logger record and exposure at or below the pharmacy limit.', passFact:'has a corrected logger record and exposure within the pharmacy limit', failFact:'has an unresolved logger excursion or exposure above the pharmacy limit' },
  { slug:'wetland_restoration_work', label:'wetland restoration scheduling', item:'restoration work package', id:'WRW', scope:'Wetland stewardship office', requirement:'Schedule only packages covered by a current habitat permit and outside the seasonal closure.', passFact:'is covered by a current habitat permit and is outside the seasonal closure', failFact:'lacks current permit coverage or overlaps the seasonal closure' },
  { slug:'microgrid_storage_dispatch', label:'microgrid storage dispatch', item:'storage block', id:'MSD', scope:'Island energy control room', requirement:'Dispatch only blocks with a passed inverter test and reserve capacity above the event minimum.', passFact:'passed its inverter test and has reserve capacity above the event minimum', failFact:'failed inverter test or reserve capacity is below the event minimum' },
  { slug:'archive_digitization_queue', label:'archive digitization queue', item:'recording batch', id:'ADQ', scope:'Community archive', requirement:'Digitize only batches covered by public consent and a verified preservation copy.', passFact:'has public digitization consent and a verified preservation copy', failFact:'consent excludes public digitization or the preservation copy is unverified' },
  { slug:'bridge_inspection_repair', label:'bridge inspection repair scheduling', item:'repair package', id:'BIR', scope:'County bridge office', requirement:'Schedule only work packages with current inspection findings and a signed lane-control plan.', passFact:'has current inspection findings and a signed lane-control plan', failFact:'inspection findings are out of date or lane-control plan is unsigned' },
  { slug:'river_water_allocation', label:'river water allocation', item:'withdrawal request', id:'RWA', scope:'River basin authority', requirement:'Approve only withdrawals within the current drought cap with a valid meter certificate.', passFact:'is within the current drought cap and has a valid meter certificate', failFact:'exceeds the current drought cap or has an expired meter certificate' },
  { slug:'emergency_supply_replenishment', label:'emergency supply replenishment', item:'supply lot', id:'ESR', scope:'Regional response warehouse', requirement:'Replenish only lots that passed safety inspection and arrive before the response deadline.', passFact:'passed safety inspection and arrives before the response deadline', failFact:'failed safety inspection or arrives after the response deadline' },
];

const scenarioData = [
  { signed:true, deadline:'14 June 2028', fact:'Highest eligible priority is selected; the highest raw score is not sufficient if the item fails a requirement.' },
  { signed:false, deadline:'2 September 2028', fact:'Tied eligible scores are resolved by ascending complete item identifier; no authorized signature is recorded for this review.' },
  { signed:true, deadline:'19 January 2029', fact:'If no item satisfies every listed requirement, select none and hold.' },
];

function scenarioWorld(domain, scenarioIndex) {
  const scenario = scenarioData[scenarioIndex];
  const domainIndex = domains.indexOf(domain);
  const baseScores = [[82,91,76,88],[95,95,74,68],[89,83,77,72]][scenarioIndex];
  const eligibilityPatterns = [[true,true,false,true],[false,true,true,true],[true,false,true,true],[true,true,true,false]];
  let scores, eligibility;
  if (scenarioIndex === 0) {
    scores = baseScores.map((_,i)=>baseScores[(i+domainIndex)%4] + domainIndex);
    eligibility = eligibilityPatterns[domainIndex%eligibilityPatterns.length];
  } else if (scenarioIndex === 1) {
    const first = domainIndex%4, second=(first+1)%4, excluded=(first+2)%4;
    scores = Array.from({length:4},(_,i)=>i===first||i===second ? 90+domainIndex : 70+((i*7+domainIndex)%13));
    eligibility = Array.from({length:4},(_,i)=>i!==excluded);
  } else {
    scores = baseScores.map((_,i)=>baseScores[(i+domainIndex)%4] + domainIndex*2);
    eligibility = [false,false,false,false];
  }
  const reviewId = `${domain.id}-${scenarioIndex + 1}0${scenarioIndex + 1}`;
  const itemIds = ['A','B','C','D'].map(letter => `${domain.id}-${scenarioIndex + 1}${letter}`);
  const candidates = itemIds.map((itemId, index) => ({ itemId, score:scores[index], eligible:eligibility[index], eligibilityEvidence:eligibility[index] ? domain.passFact : domain.failFact }));
  const eligible = candidates.filter(row => row.eligible).sort((a,b) => b.score-a.score || a.itemId.localeCompare(b.itemId));
  const winner = eligible[0] ?? null;
  const eligibleIds = winner ? eligible.map(row => row.itemId).join('; ') : 'none';
  const possibleEligibleLists = Array.from({length:16},(_,mask)=>candidates.filter((_,i)=>mask&(1<<i)).sort((a,b)=>b.score-a.score||a.itemId.localeCompare(b.itemId)).map(row=>row.itemId).join('; '));
  const possibleEligibleValues = [...new Set(['none', ...possibleEligibleLists.filter(Boolean)])];
  const interimWinner = [...candidates].sort((a,b)=>b.score-a.score || a.itemId.localeCompare(b.itemId))[0];
  const decision = winner && scenario.signed ? 'approve' : 'hold';
  const fields = {
    reviewId:'exact review identifier',
    selectedId:'highest-priority eligible item identifier; use none when no item is eligible',
    eligibleIds:'eligible item identifiers in descending priority score, ties by ascending complete identifier; use none if empty',
    priorityScore:'priority score of selected item as digits; use 0 when none is selected',
    decision:'exactly approve or hold in the final result; pending is only an initial/intermediate placeholder',
  };
  const initial = { reviewId:'UNKNOWN', selectedId:'pending', eligibleIds:'pending', priorityScore:'0', decision:'pending' };
  const passStates = [
    { ...initial, reviewId },
    { ...initial, reviewId, selectedId:interimWinner.itemId, eligibleIds:'pending', priorityScore:String(interimWinner.score) },
    { ...initial, reviewId, selectedId:winner?.itemId ?? 'none', eligibleIds, priorityScore:String(winner?.score ?? 0) },
    { ...initial, reviewId, selectedId:winner?.itemId ?? 'none', eligibleIds, priorityScore:String(winner?.score ?? 0), decision },
  ];
  const group = `v17:${domain.slug}:scenario-${scenarioIndex + 1}`;
  const slug = `${domain.slug}_scenario_${scenarioIndex + 1}`;
  const sourceFiles = {
    'pass-01-review.md': `${domain.scope} opened review ${reviewId} for ${domain.label}. The decision deadline is ${scenario.deadline}. The required condition is: ${domain.requirement} The review asks for one eligible ${domain.item}, selected by the stated priority rule.`,
    'pass-02-priority-register.md': `Priority register for ${reviewId}: ${candidates.map(row=>`${domain.item} ${row.itemId} has priority score ${row.score}`).join('. ')}. Scores are whole-number points; larger is more urgent.`,
    'pass-03-eligibility-audit.md': `Eligibility audit for ${reviewId}: ${candidates.map(row=>`${row.itemId} ${row.eligible ? domain.passFact : domain.failFact}`).join('. ')}. ${scenario.fact}`,
    'pass-04-authorization.md': scenario.signed
      ? `Authorized reviewer signed ${reviewId} for disposition on ${scenario.deadline}. The signature applies to this review and the eligibility rule remains controlling.`
      : `No authorized reviewer has signed ${reviewId} by ${scenario.deadline}. No approval is recorded for this review.`,
  };
  const passes = [
    { name:'review scope', evidence_path:'pass-01-review.md', allowed_fields:['reviewId'], constraint:`Identify the complete review ID, decision deadline, requested ${domain.item}, and eligibility requirement.`, source_scope:`the complete ${reviewId} review notice` },
    { name:'priority register', evidence_path:'pass-02-priority-register.md', allowed_fields:['selectedId','priorityScore'], constraint:'Record the provisional highest score and its item ID from the register. Eligibility has not yet been checked, so retain eligibleIds as pending.', source_scope:`the priority register for ${reviewId}` },
    { name:'eligibility audit', evidence_path:'pass-03-eligibility-audit.md', allowed_fields:['selectedId','eligibleIds','priorityScore'], constraint:'Filter to items that pass every requirement. Sort eligible IDs by descending score then ascending full ID; select the first. Use selectedId none and score 0 when no item passes.', source_scope:`the complete eligibility audit for ${reviewId}` },
    { name:'authorization record', evidence_path:'pass-04-authorization.md', allowed_fields:['decision'], constraint:'Approve only when an eligible item is selected and the authorized reviewer signed this exact review by its deadline; otherwise hold.', source_scope:`the authorization record for ${reviewId}` },
  ];
  return {
    slug, group, domain:domain.label,
    instruction:`Review the ${domain.label} case. Select the highest-priority item that passes every stated eligibility condition, breaking ties by ascending complete item identifier. Approve only if the authorized reviewer signed this exact review by its deadline; otherwise hold.`,
    fields, value_policy:'Use only the current review files. A high score does not override eligibility. Preserve the complete item IDs and output formats exactly.',
    decision_rule:'Select the eligible item with the greatest priority score, resolving ties by ascending complete identifier. If none is eligible, select none and hold. Approve only if an eligible item is selected and the authorized reviewer signed this review by its deadline.',
    initial, passes, passStates, evidence:sourceFiles,
    field_enums:{
      selectedId:{intermediate:['pending','none',...itemIds],final:['none',...itemIds]},
      eligibleIds:{intermediate:['pending',...possibleEligibleValues],final:possibleEligibleValues},
      decision:{intermediate:['pending','approve','hold'],final:['approve','hold']},
    },
    source_logic:{candidates,signatureRecorded:scenario.signed,deadline:scenario.deadline},
    justified_revision:{pass:3,field:'eligibleIds',reason:'The eligibility audit replaces the pending candidate set with only items that pass every requirement, ordered by score and identifier.'},
  };
}

export const worlds = domains.flatMap(domain => scenarioData.map((_,index)=>scenarioWorld(domain,index)));
export const sourceDomainCount = domains.length;
export const sourceScenarioCount = scenarioData.length;
