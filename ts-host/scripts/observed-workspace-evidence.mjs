import {semanticGoalCompleted} from './observed-goal-completion.mjs';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, resolve, posix } from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import { graphCustodyMatches } from '../dist/calls/graph-custody.js';

export function verifyApplicationReturn(binding, records, roots, entry) {
  if (!binding || !Array.isArray(roots) || roots.length !== 1 || binding.root_call_id !== roots[0] || typeof binding.valueJson !== 'string')
    throw new Error('application return does not identify its actual authored root');
  const value = JSON.parse(binding.valueJson), output = records.get(binding.root_call_id)?.output;
  if (output?.complete !== true || output.bytes !== Buffer.byteLength(binding.valueJson) || output.hash !== digest(binding.valueJson))
    throw new Error('application return differs from original root output');
  if(binding.kind==='void'){
    const record=records.get(binding.root_call_id);
    if(entry?.resultType!=='void' || !entry.program?.endsWith('.ts') || record?.definition?.returns!=='void' ||
       record.executor?.kind!=='crisp' || record.outcome!=='done' || binding.valueJson!=='null' ||
       !Object.hasOwn(binding,'selectedResult') || binding.selectedResult!==null)
      throw new Error('void application return does not bind an actual completed TS void contract');
    return;
  }
  if(binding.kind!==undefined && binding.kind!=='value')throw new Error('unknown application return kind');
  if (!Object.hasOwn(binding,'selectedResult') || !(isDeepStrictEqual(value,binding.selectedResult) ||
      value?.results?.length === 1 && Object.hasOwn(value.results[0],'report') && isDeepStrictEqual(value.results[0].report,binding.selectedResult)))
    throw new Error('selected domain return differs from actual application result');
}

/** Full original observations remain authoritative when required roles change. */
export function originalOutcomeObservations(original) {
  const checks=original.checks, observed=Object.hasOwn(original,'observedChecks')?original.observedChecks:checks;
  if(!checks||Array.isArray(checks)||!Object.keys(checks).length||!observed||Array.isArray(observed)||!Object.keys(observed).length||Object.values(observed).some(value=>typeof value!=='boolean')||Object.entries(checks).some(([name,value])=>typeof value!=='boolean'||!Object.hasOwn(observed,name)||observed[name]!==value))throw new Error('original outcome projection differs from its immutable observations');
  if(isDeepStrictEqual(Object.keys(checks).sort(),Object.keys(observed).sort()))return observed;
  const contract=original.outcomeContract,dimensions=original.outcomeDimensions;
  if(!contract||!dimensions)throw new Error('original projected outcome needs exhaustive native roles');
  const names=[],required=[];
  for(const role of ['effects','returnedValue','incidentalValues']){
    const declaration=contract[role],dimension=dimensions[role];
    if(role==='incidentalValues'&&declaration===undefined&&dimension===undefined)continue;
    if(!declaration||!dimension)throw new Error('original roles must account for every immutable observation');
    const roleNames=declaration.checks;
    if(!Array.isArray(roleNames)||roleNames.some(name=>typeof name!=='string'||!name)||typeof declaration.rationale!=='string'||!declaration.rationale.trim()||role==='effects'&&!roleNames.length||role==='returnedValue'&&(typeof declaration.required!=='boolean'||dimension.required!==declaration.required))throw new Error('original outcome roles need explicit native requirements');
    if(roleNames.some(name=>!Object.hasOwn(observed,name)))throw new Error('original roles contain an unobserved check');
    const expected=Object.fromEntries(roleNames.map(name=>[name,observed[name]]));
    if(!isDeepStrictEqual(dimension.checks,expected)||dimension.passed!==Object.values(expected).every(Boolean))throw new Error('original dimensions differ from immutable observations');
    names.push(...roleNames);if(role==='effects'||role==='returnedValue'&&declaration.required)required.push(...roleNames);
  }
  if(new Set(names).size!==names.length||!isDeepStrictEqual([...names].sort(),Object.keys(observed).sort())||!isDeepStrictEqual([...required].sort(),Object.keys(checks).sort()))throw new Error('original outcome roles must partition its exact observations and projection');
  return observed;
}

export async function validateReassessmentCheckRoles(proof, originalChecks, root, originalObservations=originalChecks) {
  const checks = proof.checks;
  if (!checks || Array.isArray(checks) || !Object.keys(checks).length || Object.values(checks).some(value=>value!==true) ||
      !originalChecks || !Object.keys(originalChecks).length)
    throw new Error('completed reassessment needs positive independent required checks');
  if (isDeepStrictEqual(Object.keys(checks).sort(),Object.keys(originalChecks).sort())) return;
  const observations=originalObservations;
  if(!observations||Array.isArray(observations)||!Object.keys(observations).length||Object.values(observations).some(value=>typeof value!=='boolean')||Object.entries(originalChecks).some(([name,value])=>!Object.hasOwn(observations,name)||observations[name]!==value))throw new Error('reassessment basis differs from original immutable observations');
  const contract=proof.outcomeContract, dimensions=proof.outcomeDimensions;
  if (!contract || !dimensions) throw new Error('omitted reassessment checks need source-bound outcome roles');
  const bindings=(proof.checker_sources??[]).filter(item=>item.role==='outcome-contract');
  if(bindings.length!==1 || !isDeepStrictEqual(JSON.parse(await readBound(bindings[0],root)),contract))
    throw new Error('reassessment outcome roles differ from the bound checker contract');
  const names=[],required=[];
  for(const role of ['effects','returnedValue','incidentalValues']) {
    const declaration=contract[role],observed=dimensions[role];
    if(role==='incidentalValues' && declaration===undefined && observed===undefined)continue;
    if(!declaration || !observed)throw new Error('reassessment check roles must account for every original observation');
    const roleNames=declaration.checks,needed=role==='effects'||role==='returnedValue'&&declaration.required===true;
    if(!Array.isArray(roleNames)||roleNames.some(name=>typeof name!=='string'||!name)||
       typeof declaration.rationale!=='string'||!declaration.rationale.trim()||role==='effects'&&!roleNames.length||
       role==='returnedValue'&&(typeof declaration.required!=='boolean'||observed.required!==declaration.required))
      throw new Error('reassessment check roles need explicit requirements and rationales');
    const expected=Object.fromEntries(roleNames.map(name=>[name,needed?checks[name]:observations[name]]));
    if(Object.values(expected).some(value=>typeof value!=='boolean')||!isDeepStrictEqual(observed.checks,expected)||
       observed.passed!==Object.values(expected).every(Boolean))
      throw new Error('reassessment must preserve optional observations and prove required roles');
    names.push(...roleNames);if(needed)required.push(...roleNames);
  }
  if(new Set(names).size!==names.length||!isDeepStrictEqual([...names].sort(),Object.keys(observations).sort())||
     !isDeepStrictEqual([...required].sort(),Object.keys(checks).sort()))
    throw new Error('reassessment roles must partition the exact original check set');
}

/** Keep the old required projection distinct from all retained observations. */
export async function validateReassessmentOutcomeRoles(proof,original,root) {
  let roleBasis=original.checks;
  if(!isDeepStrictEqual(Object.keys(proof.checks??{}).sort(),Object.keys(original.checks??{}).sort())) {
    const observations=originalOutcomeObservations(original);
    // Older bound contracts partition the original required projection. Full
    // observation/promotion contracts must exhaust the complete retained set.
    const declared=['effects','returnedValue','incidentalValues'].flatMap(role=>proof.outcomeContract?.[role]?.checks??[]);
    const projectionContract=declared.every(name=>typeof name==='string')&&isDeepStrictEqual([...new Set(declared)].sort(),Object.keys(original.checks??{}).sort());
    roleBasis=projectionContract?original.checks:observations;
  }
  await validateReassessmentCheckRoles(proof,original.checks,root,roleBasis);
}

export function declaredReturnSelectors(proof) {
  const incidental=proof.outcomeDimensions?.incidentalValues,returned=proof.outcomeDimensions?.returnedValue;
  if(incidental?.passed!==false&&returned?.passed!==false)return [];
  if(incidental?.passed===false&&!Array.isArray(incidental.incorrectReturns))throw new Error('incorrect incidental returns need exact original value bindings');
  const values=[...(incidental?.incorrectReturns??[])];
  for(const item of returned?.incorrectReturns??[])if(!values.some(existing=>isDeepStrictEqual(existing,item)))values.push(item);
  const contains=(value,wanted)=>isDeepStrictEqual(value,wanted)||value!==null&&typeof value==='object'&&Object.values(value).some(item=>contains(item,wanted));
  for(const item of values){
    if(!item.definition?.name||!item.definition?.source||typeof item.valueJson!=='string'||createHash('sha256').update(item.valueJson).digest('hex')!==item.sha256)throw new Error('incorrect return value identity/hash differs');
    if(item.contribution){
      const c=item.contribution,roles=proof.outcomeDimensions??{};
      const failed={...roles.returnedValue?.checks,...roles.incidentalValues?.checks};
      if(c.schema!=='natlang.application-return-contribution/1' || !['root','forwarded','projection','uncertain-projection','incidental-child'].includes(c.kind) ||
         typeof c.root_call_id!=='string'||typeof c.producer_call_id!=='string'||typeof c.root_value_json!=='string'||
         digest(c.root_value_json)!==c.root_sha256||!isDeepStrictEqual(JSON.parse(c.root_value_json),proof.actualResult)||
         !Array.isArray(c.checks)||!c.checks.length||new Set(c.checks).size!==c.checks.length||c.checks.some(check=>typeof check!=='string'||failed[check]!==false)||
         ![c.root_path,c.producer_path].every(path=>Array.isArray(path)&&path.every(key=>typeof key==='string'||Number.isSafeInteger(key)&&key>=0))||
         !Array.isArray(c.lineage)||!c.lineage.length||c.lineage.some(id=>typeof id!=='string')||new Set(c.lineage).size!==c.lineage.length)
        throw new Error('invalid authored application return contribution');
      if(c.kind==='incidental-child'&&c.checks.some(check=>roles.incidentalValues?.checks?.[check]!==false))throw new Error('incidental child contribution needs actual failed incidental checks');
    }else if(!contains(proof.actualResult,JSON.parse(item.valueJson)))throw new Error('incorrect return value is outside the actual application result');
  }
  return values;
}


/** Recheck authored data flow against original typed outputs and the complete native call tree. */
export function validateReturnContribution(item, records, roots) {
  const c=item.contribution;if(!c)return;
  const root=records.get(c.root_call_id),producer=records.get(c.producer_call_id);
  const matches=(record,text,hash)=>record?.outcome==='done'&&record.output?.complete===true&&record.output.hash===hash&&record.output.bytes===Buffer.byteLength(text);
  const sampled=record=>record?.outcome==='done'&&['agent','crisp-agent'].includes(record.executor?.kind)&&typeof record.executor?.model_id==='string'&&!!record.executor.model_id&&record.cost?.model_requests>0&&record.definition?.source?.endsWith('.nl')&&
    (record.model_turns??[]).some(turn=>turn.kind==='generation'&&turn.reply&&!turn.error);
  const chainOf=record=>{
    const chain=[],seen=new Set();let current=record;
    while(current){if(seen.has(current.call_id))return null;seen.add(current.call_id);chain.unshift(current.call_id);
      if(current===root)return chain;current=records.get(current.parent_call_id);}
    return null;
  };
  const crispChain=record=>{const chain=chainOf(record);return chain&&chain.slice(0,-1).every(id=>records.get(id)?.executor?.kind==='crisp'&&records.get(id)?.outcome==='done')?chain:null;};
  const at=(value,path)=>{for(const key of path){const valid=Array.isArray(value)?Number.isSafeInteger(key)&&key>=0&&key<value.length:value!==null&&typeof value==='object'&&typeof key==='string';if(!valid||!Object.hasOwn(value,key))return {found:false};value=value[key];}return {found:true,value};};
  if(roots.length!==1||roots[0]!==c.root_call_id||!root||!producer||
     !matches(root,c.root_value_json,c.root_sha256)||!matches(producer,item.valueJson,item.sha256)||
     producer.definition?.name!==item.definition.name||producer.definition?.source!==item.definition.source||
     !isDeepStrictEqual(chainOf(producer),c.lineage))
    throw new Error('application return contribution differs from actual captured outputs');
  const rootPart=at(JSON.parse(c.root_value_json),c.root_path),producerPart=at(JSON.parse(item.valueJson),c.producer_path);
  if(c.kind==='root'){
    if(producer!==root||c.lineage.length!==1||c.root_path.length||c.producer_path.length)
      throw new Error('root contribution must identify its whole actual root return');
    return;
  }
  if(c.kind==='incidental-child'){
    if(producer===root||!sampled(producer)||c.root_path.length||c.producer_path.length)
      throw new Error('incidental child contribution needs an actual sampled child return');
    return;
  }
  if(producer===root||!sampled(producer)||!crispChain(producer))
    throw new Error('forwarded or projected contribution needs an actual sampled NL producer and crisp ancestry');
  if(c.kind==='forwarded'){
    if(c.root_path.length||c.producer_path.length)throw new Error('forwarding contribution cannot project values');
    for(const id of c.lineage.slice(0,-1)){
      const ancestor=records.get(id);
      if([...records.values()].filter(record=>record.parent_call_id===id).length!==1||!matches(ancestor,item.valueJson,item.sha256))
        throw new Error('forwarding contribution must preserve the sole child whole return');
    }
    return;
  }
  if(typeof c.producer_source_suffix!=='string'||!c.producer_source_suffix||!producer.definition.source.endsWith(c.producer_source_suffix)||!Array.isArray(c.candidates))
    throw new Error('projection needs its authored source suffix and complete actual candidate returns');
  const eligible=[...records.values()].filter(record=>sampled(record)&&record.definition.source.endsWith(c.producer_source_suffix)&&crispChain(record));
  if(new Set(c.candidates.map(candidate=>candidate.call_id)).size!==c.candidates.length||
     !isDeepStrictEqual(c.candidates.map(candidate=>candidate.call_id).sort(),eligible.map(record=>record.call_id).sort()))
    throw new Error('projection candidates differ from the complete actual source inventory');
  const matching=[];
  for(const candidate of c.candidates){
    if(typeof candidate.value_json!=='string'||digest(candidate.value_json)!==candidate.sha256||!matches(records.get(candidate.call_id),candidate.value_json,candidate.sha256))
      throw new Error('projection candidate differs from actual captured return bytes');
    const part=at(JSON.parse(candidate.value_json),c.producer_path);
    if(rootPart.found&&part.found&&isDeepStrictEqual(rootPart.value,part.value))matching.push(candidate.call_id);
    if(candidate.call_id===producer.call_id&&(candidate.value_json!==item.valueJson||candidate.sha256!==item.sha256))
      throw new Error('selected producer differs from the actual candidate return');
  }
  if(c.kind==='projection'){
    if(matching.length!==1||matching[0]!==producer.call_id||!rootPart.found||!producerPart.found)
      throw new Error('application return projection is absent, ambiguous or differs from actual producer value');
  }else{
    const reason=matching.length>1?'ambiguous-actual-producer':matching.length===0?'no-verified-actual-producer':null;
    const uncertain=matching.length?matching:eligible.map(record=>record.call_id);
    if(!reason||c.uncertainty!==reason||!uncertain.includes(producer.call_id))
      throw new Error('uncertain contribution does not bind an actual ambiguous or unbound return');
  }
}


export function validateReturnContributions(items,records,roots){
  for(const item of items)validateReturnContribution(item,records,roots);
  for(const item of items){
    const c=item.contribution;if(c?.kind!=='uncertain-projection')continue;
    const chosen=items.filter(other=>other.contribution&&isDeepStrictEqual({...other.contribution,producer_call_id:null,lineage:null},{...c,producer_call_id:null,lineage:null})).map(other=>other.contribution.producer_call_id);
    const at=(value,path)=>{for(const key of path){const valid=Array.isArray(value)?Number.isSafeInteger(key)&&key>=0&&key<value.length:value!==null&&typeof value==='object'&&typeof key==='string';if(!valid||!Object.hasOwn(value,key))return {found:false};value=value[key];}return {found:true,value};};
    const root=at(JSON.parse(c.root_value_json),c.root_path);
    const matching=c.candidates.filter(candidate=>{const part=at(JSON.parse(candidate.value_json),c.producer_path);return root.found&&part.found&&isDeepStrictEqual(root.value,part.value);}).map(candidate=>candidate.call_id);
    const needed=matching.length?matching:c.candidates.map(candidate=>candidate.call_id);
    if(needed.some(id=>!chosen.includes(id)))throw new Error('uncertain return contribution omitted an actual possible producer');
  }
}

async function readBound(binding, root) {
  if (!binding?.path || !/^[a-f\d]{64}$/u.test(binding.sha256 ?? ''))
    throw new Error('invalid observed workspace artifact binding');
  const bytes = await readFile(resolve(root, binding.path));
  if (createHash('sha256').update(bytes).digest('hex') !== binding.sha256)
    throw new Error(`observed workspace artifact changed: ${binding.path}`);
  return bytes;
}

export function verifyObservedCallTree(records, root) {
  const byId = new Map();
  const children = new Map();
  for (const record of records) {
    if (byId.has(record.call_id)) throw new Error('observed call graph repeats an invocation');
    byId.set(record.call_id, record);
    const group = children.get(record.parent_call_id) ?? [];
    group.push(record.call_id);
    children.set(record.parent_call_id, group);
  }
  const hasCustody = records.some(record => record.graph_custody != null);
  const pending = [root];
  const seen = new Set();
  while (pending.length) {
    const id = pending.pop();
    const call = byId.get(id);
    const ids = children.get(id) ?? [];
    if (!call || seen.has(id) || (id === root && call.parent_call_id !== null))
      throw new Error('observed call graph has invalid parent links');
    // A fresh task may be a recorded causal child without inheriting execution authority.
    // graphCustodyMatches binds each call to its own task and exact child inventory.
    if (hasCustody && !graphCustodyMatches(call, ids))
      throw new Error('observed call graph differs from recorded child inventory');
    seen.add(id);
    pending.push(...ids);
  }
  if (seen.size !== records.length) throw new Error('observed call graph contains unreachable invocations');
}

const digest = bytes => createHash('sha256').update(bytes).digest('hex');

export function verifyEntry(interaction, record, partial) {
  const entry = interaction.entry ?? {program: 'skills/open-user-goal/runUserGoal.nl',
    callable: 'runUserGoal', arguments: {goal: interaction.goal}};
  if (typeof entry.program !== 'string' || !/\.(nl|ts)$/u.test(entry.program) ||
      entry.program.startsWith('/') || entry.program.split('/').includes('..') ||
      interaction.program !== entry.program || typeof entry.callable !== 'string' || !entry.callable ||
      !entry.arguments || typeof entry.arguments !== 'object' || Array.isArray(entry.arguments) ||
      record.definition?.name !== entry.callable || (!partial && record.outcome !== 'done') ||
      !isDeepStrictEqual(Object.keys(record.inputs ?? {}).sort(), Object.keys(entry.arguments).sort()))
    throw new Error('observed root differs from its actual application entry');
  if (interaction.entry) {
    const {program_root: programRoot, definition: {source}} = record;
    if (typeof programRoot !== 'string' || typeof source !== 'string' ||
        programRoot.startsWith('/') || source.startsWith('/') ||
        programRoot.split('/').includes('..') || source.split('/').includes('..') ||
        posix.join(programRoot, source) !== entry.program ||
        !entry.argumentJson || !isDeepStrictEqual(Object.keys(entry.argumentJson).sort(), Object.keys(entry.arguments).sort()))
      throw new Error('observed application entry needs its actual source and exact native argument JSON');
  }
  if(entry.program.endsWith('.ts') && (record.executor?.kind!=='crisp' || record.definition?.id!==`ts:${record.definition?.source}#default`))
    throw new Error('observed application entry does not bind its actual TS default export');
  if (entry.resultType === 'void' && record.definition?.returns !== 'void')
    throw new Error('void application entry differs from its declared return contract');
  for (const [name, value] of Object.entries(entry.arguments)) {
    const text = interaction.entry ? entry.argumentJson[name] : JSON.stringify(value);
    if (typeof text !== 'string' || !isDeepStrictEqual(JSON.parse(text), value))
      throw new Error('observed application argument JSON differs from its declared value');
    const bytes = Buffer.from(text), input = record.inputs[name];
    if (input?.complete !== true || input.bytes !== bytes.length || input.hash !== digest(bytes))
      throw new Error('observed application input differs from the captured invocation');
  }
}

export async function verifyNativeInvocationCertificate(invocation,records,scope,root,details){
  if(!Object.hasOwn(invocation,'certificate'))throw Error('native invocation requires a source-bound certificate');
  const certificate=JSON.parse(await readBound(invocation.certificate,root));
  const raw=await readBound(certificate.originalPacket,root);
  const packet=JSON.parse(raw[0]===0x1f&&raw[1]===0x8b?gunzipSync(raw):raw);
  const original=JSON.parse(await readBound(scope.original_outcome,root));
  const exercise=packet.execution?.diagnostics?.natlangTeacher??{};
  const proof=packet.execution?.diagnostics?.openGoalVerification??{};
  const captured=new Map((exercise.calls??[]).map(item=>[item.callId,item.record]));
  if(details&&!isDeepStrictEqual(new Map((exercise.calls??[]).map(item=>[item.callId,item])),details))throw Error('native invocation packet differs from original events or call metadata');
  const record=records.get(invocation.call_id);
  if(certificate.schema!=='natlang.independent-native-return/1'||!isDeepStrictEqual(original,packet.result)||
     original.passed!==false||!isDeepStrictEqual(captured,records)||record?.outcome!=='done'||
     !isDeepStrictEqual(certificate.definition,record.definition)||!isDeepStrictEqual(certificate.inputs,record.inputs)||
     !isDeepStrictEqual(certificate.output,record.output))throw Error('native invocation certificate differs from original failed capture');
  if(!scope.assessor)throw Error('automatic native scope lacks its source-bound assessor');
  await readBound(scope.assessor,root);
  const source=(await readBound(certificate.checker,root)).toString('utf8');
  const match=/^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/u.exec(source);
  const matches=(ref,text)=>typeof text==='string'&&ref?.complete===true&&ref.bytes===Buffer.byteLength(text)&&ref.hash===digest(text);
  if(!match||JSON.parse(certificate.instructionsJson)!==match[2]||!matches(record.definition.instructions,certificate.instructionsJson))
    throw Error('native invocation contract differs from actual authored instructions');
  const project=(value,parts)=>{if(!Array.isArray(parts)||!parts.length)throw Error('native invocation needs explicit evidence paths');for(const part of parts){if(value===null||typeof value!=='object'||!Object.hasOwn(value,part))throw Error('native invocation evidence path absent');value=value[part];}return value;};
  const contracts={
    'gad-question-judgment/1':{source:'reviewDurableQuestion/reviewQuestion/interpretQuestion.nl',returns:'DurableQuestionJudgment',binding:{returnPath:['actualResult','question'],inputPaths:{goal:['goal'],history:['diagnostics','history']},checkNames:['exactIndividualPublications','exactQuestionDisposition']}},
    'gad-participant-judgment/1':{source:'reviewDurableQuestion/reviewParticipant/evaluateRoster.nl',returns:'DurableParticipantJudgment',binding:{returnPath:['actualResult','participant'],inputPaths:{goal:['goal'],question:['actualResult','question'],roster:['diagnostics','roster']},checkNames:['exactIndividualPublications','exactRetainedRoster','exactParticipantDisposition']}}
  };
  const contract=contracts[certificate.contract],binding=certificate.nativeBinding??{},checks=binding.checkNames;
  if(!contract||record.definition.id!=='nl:'+contract.source||record.definition.source!==contract.source||record.definition.returns!==contract.returns||!isDeepStrictEqual(binding,contract.binding))
    throw Error('native invocation has an unregistered or mismatched complete-result contract');
  const observed=proof.observedChecks??proof.checks??{};
  if(!Array.isArray(checks)||!checks.length||checks.some(name=>observed[name]!==true))throw Error('native invocation cites absent or failed independent checks');
  const paths=binding.inputPaths??{},inputs=certificate.expectedInputs??{},texts=certificate.expectedInputJson??{};
  const keys=value=>Object.keys(value).sort();
  if(!isDeepStrictEqual(keys(paths),keys(record.inputs))||!isDeepStrictEqual(keys(inputs),keys(paths))||!isDeepStrictEqual(keys(texts),keys(paths))||
     Object.keys(inputs).some(name=>!isDeepStrictEqual(project(proof,paths[name]),inputs[name])||!isDeepStrictEqual(JSON.parse(texts[name]),inputs[name])||!matches(record.inputs[name],texts[name]))||
     !isDeepStrictEqual(project(proof,binding.returnPath),certificate.expectedReturn)||!isDeepStrictEqual(JSON.parse(certificate.expectedReturnJson),certificate.expectedReturn)||!matches(record.output,certificate.expectedReturnJson))
    throw Error('native invocation value differs from its independent input/return evidence');
  const descendants=new Set([record.call_id]);let changed=true;
  while(changed){changed=false;for(const [id,call]of records)if(descendants.has(call.parent_call_id)&&!descendants.has(id)){descendants.add(id);changed=true;}}
  if([...descendants].some(id=>records.get(id).effects?.length))throw Error('native pure-return certificate contains authority-bearing effects');
}

async function trainingScope(receipt, row, root) {
  const scope = receipt.training_scope;
  if (!scope) {
    if (receipt.status !== 'passed' || row.outcome.training_scope != null)
      throw new Error('partial observed decisions require independently attested training scope');
    return null;
  }
  if (receipt.status !== 'partial' || scope.schema !== 'natlang.training-scope/2' ||
      !isDeepStrictEqual(scope, row.outcome.training_scope) ||
      JSON.parse(await readBound(scope.original_outcome, root)).passed !== false ||
      !Array.isArray(scope.invocations) || !scope.invocations.length)
    throw new Error('invalid independent partial training scope');
  if(scope.assessor)await readBound(scope.assessor,root);
  if(!scope.assessor||scope.invocations.some(invocation=>!Object.hasOwn(invocation,'certificate')))
    throw Error('native certificates require training-scope/2 and its source-bound assessor');
  const selected = new Map();
  for (const invocation of scope.invocations) {
    if (typeof invocation.call_id !== 'string' || selected.has(invocation.call_id) ||
        !invocation.checks || !Object.keys(invocation.checks).length ||
        Object.values(invocation.checks).some(value => value !== true) ||
        !Array.isArray(invocation.evidence) || !invocation.evidence.length ||
        (invocation.ordinals != null && (!Array.isArray(invocation.ordinals) || !invocation.ordinals.length ||
          new Set(invocation.ordinals).size !== invocation.ordinals.length ||
          invocation.ordinals.some(index => !Number.isSafeInteger(index) || index < 0))))
      throw new Error('partial invocation lacks independently verified decision evidence');
    for (const evidence of invocation.evidence) {
      if (!['workspace', 'cdp'].includes(evidence.kind)) throw new Error('partial decisions require native workspace evidence');
      await readBound(evidence, root);
    }
    selected.set(invocation.call_id, invocation.ordinals == null ? null : new Set(invocation.ordinals));
  }
  return selected;
}

export async function verifyOriginalRejection(packet, proof, root) {
  const passed=packet.result?.passed,bindings=(proof.evidence??[]).filter(item=>item.role==='original-rejected-review');
  if(!bindings.length)return passed===false;
  if(bindings.length!==1 || typeof passed!=='boolean')return false;
  const review=JSON.parse(await readBound(bindings[0],root)),observed=packet.execution?.diagnostics?.openGoalVerification;
  const input=observed?.preparedContentProject?.artifact?.semanticReviewInput??observed?.semanticReviewInput;
  const object=value=>value!==null && typeof value==='object' && !Array.isArray(value);
  const exchanges=[review.exchange,...(Array.isArray(review.exchanges)?review.exchanges:[])];
  return review.schema==='natlang.live-workflow-review/1' && review.verdict?.accepted===false && object(input) && !Array.isArray(input) && Object.keys(input).length>0 &&
    isDeepStrictEqual(review.input,input) && exchanges.some(item=>object(item) && object(item.request) && object(item.reply));
}

async function completedReassessment(receipt, manifest, row, root) {
  const reassessment=receipt.reassessment;
  if (!reassessment) {
    if (row.outcome.reassessment) throw new Error('unattested completed reassessment');
    return;
  }
  if (receipt.status!=='passed' || receipt.training_scope || !isDeepStrictEqual(reassessment,row.outcome.reassessment))
    throw new Error('completed reassessment cannot relabel partial decisions');
  const original=JSON.parse(await readBound(reassessment.original_outcome,root));
  const assessment=JSON.parse(await readBound(reassessment.assessment,root));
  const validator=JSON.parse(await readBound(receipt.validator,root));
  const raw=await readBound(validator.original_packet,root);
  const packet=JSON.parse(validator.original_packet.path.endsWith('.gz')?gunzipSync(raw):raw);
  const native=packet.execution.diagnostics.natlangTeacher;
  const prior=packet.execution.diagnostics.openGoalVerification;
  const captured=await Promise.all(manifest.calls.map(async binding=>JSON.parse(await readBound(binding,root))));
  const byId=calls=>Object.fromEntries(calls.map(item=>[item.callId,item]));
  if(native.calls?.length!==captured.length || !isDeepStrictEqual(byId(native.calls),byId(captured)) ||
      !isDeepStrictEqual(native.rootCallIds,receipt.root_call_ids) ||
      !isDeepStrictEqual(native.entry??null,manifest.interaction?.entry??null) || prior.goal!==manifest.interaction?.goal)
    throw new Error('completed reassessment does not bind exact original application graph');
  const workspace=receipt.evidence.filter(item=>item.kind==='workspace');
  if(workspace.length!==1)throw new Error('completed reassessment requires one independent whole-outcome proof');
  const proof=JSON.parse(await readBound(workspace[0],root)).proof;
  const originalCompleted=JSON.parse(await readBound(validator.completed_outcome,root));
  const normalized=value=>({...value,...Object.fromEntries(['evidence','checker_sources'].map(field=>[field,(value[field]??[]).map(({path,...binding})=>binding)]))});
  if(assessment.completed_outcome?.sha256!==validator.completed_outcome.sha256 || !isDeepStrictEqual(normalized(proof),normalized(originalCompleted)))
    throw new Error('completed reassessment differs from bound independent verdict');
  if(!await verifyOriginalRejection(packet,proof,root) || !isDeepStrictEqual(original,packet.result) ||
      assessment.schema!=='natlang.live-reassessment/1' || assessment.task_id!==manifest.id || assessment.original_sha256!==digest(raw) ||
      proof?.schema!=='natlang.live-outcome-reassessment/1' || proof.passed!==true || proof.task_id!==manifest.id ||
      proof.original_sha256!==assessment.original_sha256 || proof.goal!==prior.goal ||
      !isDeepStrictEqual(proof.entry??null,native.entry??null) || !isDeepStrictEqual(proof.actualResult??null,native.report??null) ||
      !proof.checks || !Object.keys(proof.checks).length ||
      Object.values(proof.checks).some(value=>value!==true) || !proof.rationale ||
      !Array.isArray(proof.evidence) || !proof.evidence.length || !Array.isArray(proof.checker_sources) || !proof.checker_sources.length)
    throw new Error('completed reassessment differs from actual original execution or independent outcome');
  await validateReassessmentOutcomeRoles(proof,prior,root);
  for(const artifact of [...proof.evidence,...proof.checker_sources])await readBound(artifact,root);
}

export function reviewedDecisionOrdinals(selectors, record, artifact, matched) {
  const excluded=new Set();
  selectors.forEach((item,index)=>{
    if(item.call_id!==record.call_id)return;
    const turns=record.model_turns??[];
    if(item.call_sha256!==artifact.sha256 || !isDeepStrictEqual(item.definition,record.definition) ||
       !Number.isSafeInteger(item.ordinal) || item.ordinal<0 || turns[item.ordinal]?.kind!=='generation' ||
       !turns[item.ordinal]?.reply || typeof turns[item.ordinal].reply!=='object' || Array.isArray(turns[item.ordinal].reply) ||
       typeof item.reason!=='string' || !item.reason.trim() || excluded.has(item.ordinal))
      throw new Error('decision review differs from captured call bytes or generation');
    excluded.add(item.ordinal);matched.add(index);
  });
  return excluded;
}

function decisionEligibility(record, detail, selected, incorrectReturn, reviewed, completedGoal) {
  const actual = record.model_turns ?? [];
  const events = detail.events ?? [];
  const failed = new Set();
  for (const event of events.filter(event => event.kind === 'action' && ['error', 'rejected', 'blocked', 'budget'].includes(event.outcome))) {
    const owners = actual.flatMap((exchange, index) => exchange.kind === 'generation' &&
      exchange.reply?.raw_calls?.some(call => call.id === event.tool_call_id) ? [index] : []);
    if (!event.tool_call_id || owners.length !== 1) throw new Error('failed action lacks exact captured generation ownership');
    failed.add(owners[0]);
  }
  const actionEvents = Map.groupBy(events.filter(event => event.kind === 'action'), event => event.tool_call_id);
  const ordinals = selected?.get(record.call_id);
  if (selected?.has(record.call_id) && ((ordinals === null && record.outcome !== 'done') ||
      (ordinals !== null && [...ordinals].some(index => actual[index]?.kind !== 'generation'))))
    throw new Error('partial scope names an invalid invocation or generation');
  const executor = Boolean(record.executor?.kind !== 'crisp' && (record.outcome === 'done' || completedGoal ||
    selected?.has(record.call_id) && ordinals !== null));
  return actual.map((exchange, index) => {
    if (exchange.schema !== 'natlang.model-exchange/1') throw new Error('unsupported captured model exchange');
    if (exchange.kind !== 'generation' || exchange.error) return false;
    const reply = exchange.reply, gateway = reply?.raw_response?.vibestudio;
    const content = gateway?.response?.content;
    if (!Array.isArray(exchange.request?.messages) || !gateway?.model || !gateway.api || !gateway.turnId ||
        !gateway.attribution?.callerId || !gateway.attribution?.callerKind || !gateway.context ||
        gateway.response?.role !== 'assistant' || !Array.isArray(content) || !Array.isArray(reply.raw_calls))
      throw new Error('observed generation lacks its original gateway exchange');
    const originalCalls = content.filter(block => block.type === 'toolCall');
    if (reply.raw_calls.length !== originalCalls.length || originalCalls.some((call, offset) => {
      const raw = reply.raw_calls[offset];
      return !call.id || raw.id !== call.id || raw.function?.name !== call.name ||
        !isDeepStrictEqual(JSON.parse(raw.function?.arguments ?? 'null'), call.arguments);
    }) || !isDeepStrictEqual(reply.calls ?? [], originalCalls.map(call => [call.name, call.arguments])) ||
        (reply.text ?? '') !== content.filter(block => block.type === 'text').map(block => block.text).join('') ||
        ![content.filter(block => block.type === 'thinking').map(block => block.thinking).join('\n'),
          content.filter(block => block.type === 'thinking' && !block.redacted).map(block => block.thinking).join('\n')].includes(reply.reasoning ?? ''))
      throw new Error('observed normalized model reply differs from original gateway response');
    return (!selected || selected.has(record.call_id) && (ordinals === null || ordinals.has(index))) &&
      !reviewed.has(index) && index !== incorrectReturn && executor && !failed.has(index) &&
      (record.outcome === 'done' || selected !== null || originalCalls.length > 0) &&
      originalCalls.every(call => actionEvents.get(call.id)?.length === 1 &&
        ['ok', 'completed'].includes(actionEvents.get(call.id)[0].outcome)) &&
      Boolean(reply.text || originalCalls.length);
  });
}

/** Verify the on-disk artifacts before allowing the generic native exporter to use an observed capture. */
export async function verifyObservedWorkspaceEvidence(row) {
  const binding = row.provenance.source_manifest;
  const root = dirname(binding.path);
  const manifest = JSON.parse(await readBound(binding, process.cwd()));
  const attestation = manifest.verification;
  const receipt = JSON.parse(await readBound(attestation, root));
  const equal = isDeepStrictEqual;
  if (row.provenance.execution_kind !== 'observed-live-workspace' || row.task.source !== 'observed-live-workspace' ||
      row.outcome.accepted !== true || manifest.split !== 'train' || !equal(manifest.learner_kinds, ['natlang-executor']) ||
      manifest.schema !== 'natlang.live-teacher-task/1' || receipt.schema !== 'natlang.live-task-verification/1' ||
      !equal(attestation, row.provenance.verification) || receipt.task_id !== row.task.id ||
      (receipt.split ?? manifest.split) !== manifest.split ||
      receipt.run_id !== row.outcome.world.run_id || !['passed', 'partial'].includes(receipt.status) ||
      receipt.status !== row.outcome.status || receipt.status === 'partial' && row.outcome.task_completed !== false ||
      !receipt.checks || !Object.keys(receipt.checks).length || Object.values(receipt.checks).some(value => value !== true) ||
      !['id', 'family', 'split', 'source_namespace', 'source_ids', 'source_groups', 'source_revisions', 'interaction'].every(
        key => equal(manifest[key], row.task[key])) ||
      !equal(receipt.interaction, row.task.interaction) || !equal(manifest.world, receipt.world) ||
      !equal(receipt.world, row.outcome.world) || !equal(receipt.checks, row.outcome.checks) ||
      !equal(receipt.root_call_ids, row.outcome.root_call_ids) ||
      !equal(manifest.code, row.provenance.code) || manifest.code?.dirty !== false)
    throw new Error('observed workspace trajectory differs from the independently verified manifest');
  const evidenceKinds = new Set();
  const workspaceEvidence = [];
  let incorrectOptionalReturn = false;
  const returnSelectors=[];
  const decisionSelectors=[];
  const matchedDecisionSelectors=new Set();
  const applicationReturns=[];
  const matchedReturnSelectors=new Set();
  for (const evidence of receipt.evidence ?? []) {
    const evidenceBytes = await readBound(evidence, root);
    evidenceKinds.add(evidence.kind);
    if (evidence.kind === 'workspace') {
      const payload=JSON.parse(evidenceBytes);
      workspaceEvidence.push(payload);
      if(Object.hasOwn(payload,"applicationReturn"))applicationReturns.push(payload.applicationReturn);
      const proof=payload.proof??{};
      returnSelectors.push(...declaredReturnSelectors(proof));
      if(proof.decisionReview!==undefined){
        if(proof.decisionReview?.schema!=='natlang.live-decision-review/1' || !Array.isArray(proof.decisionReview.excluded))throw new Error('invalid source-bound decision review');
        decisionSelectors.push(...proof.decisionReview.excluded);
      }
      const returned = proof.outcomeDimensions?.returnedValue;
      incorrectOptionalReturn ||= returned?.required === false && returned.passed === false;
    }
  }
  if (!Array.isArray(manifest.required_effect_kinds) || !manifest.required_effect_kinds.length ||
      manifest.required_effect_kinds.some(kind => !['workspace', 'cdp'].includes(kind) || !evidenceKinds.has(kind)))
    throw new Error('observed workspace outcome evidence is incomplete');
  await readBound(receipt.validator, root);

  await completedReassessment(receipt, manifest, row, root);
  const completedGoal = semanticGoalCompleted(row.task.interaction, workspaceEvidence.map(item => item.proof ?? {}),
    receipt, workspaceEvidence, receipt.reassessment != null);
  if (row.outcome.task_completed !== completedGoal) throw new Error('observed task completion differs from bound whole-goal evidence');
  const selected = await trainingScope(receipt, row, root);
  const decisions = Map.groupBy(row.trajectory, item => item.invocation_id);
  const returnExclusions = [];
  const capturedRecords = new Map();
  const capturedDetails = new Map();
  const captureScopes = new Map(receipt.calls.map(scope => [scope.call_id, scope]));
  const callFiles = new Map(manifest.calls.map(artifact => [artifact.path, artifact]));
  if (captureScopes.size !== receipt.calls.length || callFiles.size !== manifest.calls.length)
    throw new Error('observed capture repeats an attested call or artifact');
  const roots = new Set(receipt.root_call_ids);
  const goal = row.task.interaction?.goal;
  const records = [];
  if (row.task.interaction?.kind !== 'open-user-goal' || roots.size !== 1 || typeof goal !== 'string' || !goal.trim())
    throw new Error('observed open goal requires one independently attested application root');
  if(returnSelectors.some(item=>item.contribution))for(const call of row.outcome.invocation_ledger){
    const original=JSON.parse(await readBound(call.artifact,root));
    if(capturedRecords.has(call.call_id))throw new Error('observed capture repeats an invocation');
    capturedRecords.set(call.call_id,original.record);
  }
  validateReturnContributions(returnSelectors,capturedRecords,receipt.root_call_ids);
  for (const call of row.outcome.invocation_ledger) {
    const original = JSON.parse(await readBound(call.artifact, root));
    const record = original.record;
    capturedDetails.set(record.call_id,original);
    capturedRecords.set(record.call_id, record);
    records.push({
      call_id: record.call_id, parent_call_id: record.parent_call_id, task_id: record.task_id,
      graph_custody: record.graph_custody, effects: record.effects,
    });
    if (record.call_id !== call.call_id || record.version !== 'natlang.calls/1' ||
        original.summary.definition_id !== record.definition?.id ||
        !['agent', 'crisp', 'crisp-agent'].includes(record.executor?.kind) ||
        !equal(call.artifact, callFiles.get(call.artifact.path)) ||
        original.callId !== call.call_id || !equal(call.scope, captureScopes.get(call.call_id)) ||
        ['call_id', 'owner_kind', 'owner_key', 'run_id'].some(key => original.summary[key] !== call.scope[key]) ||
        original.summary.parent_call_id !== call.parent_call_id ||
        !equal(record.definition, call.definition) || !equal(record.executor, call.executor) ||
        !equal(record.parent_call_id, call.parent_call_id) || record.outcome !== call.outcome ||
        !equal(record.inputs ?? null, call.inputs) || !equal(record.output ?? null, call.output) ||
        !equal(record.graph_custody ?? null, call.graph_custody) ||
        ['parent_action_index', 'started_at', 'ended_at'].some(key => !equal(record[key] ?? null, call[key] ?? null)) ||
        !equal(original.events ?? [], call.events) || !equal(record.effects ?? [], call.effects))
      throw new Error('observed invocation differs from its captured call or verified scope');
    if (record.parent_call_id === null) {
      if (!roots.delete(call.call_id)) throw new Error('observed root was not independently attested');
      verifyEntry(row.task.interaction, record, selected !== null || completedGoal);
    }

    const actual = record.model_turns ?? [];
    const projected = decisions.get(call.call_id) ?? [];
    let selectorMatch=false;
    returnSelectors.forEach((item,index)=>{
      const matches=(!item.contribution||record.call_id===item.contribution.producer_call_id)&&record.definition?.name===item.definition.name&&record.definition?.source===item.definition.source&&record.output?.complete===true&&record.output.hash===item.sha256&&record.output.bytes===Buffer.byteLength(item.valueJson);
      if(matches){matchedReturnSelectors.add(index);selectorMatch=true;}
    });
    const lastGeneration = selectorMatch || incorrectOptionalReturn && record.parent_call_id === null
      ? actual.findLastIndex(exchange => exchange.kind === 'generation') : -1;
    if (lastGeneration >= 0) returnExclusions.push({invocation_id: call.call_id, ordinal: lastGeneration});
    const reviewed=reviewedDecisionOrdinals(decisionSelectors,record,call.artifact,matchedDecisionSelectors);
    const eligible = decisionEligibility(record, original, selected, lastGeneration, reviewed, completedGoal);
    if (projected.length !== actual.length)
      throw new Error('observed trajectory omitted captured exchanges');
    for (const [index, decision] of projected.entries()) {
      const exchange = actual[index];
      if (decision.training_admission?.approved !== eligible[index] ||
          index !== decision.live_exchange.ordinal ||
          !equal(decision.live_exchange.call_artifact, call.artifact) ||
          decision.parent_invocation_id !== record.parent_call_id ||
          decision.function !== record.definition.name || decision.definition_id !== record.definition.id ||
          !equal(decision.context, exchange.request?.messages ?? null) ||
          !equal(decision.tools_offered, exchange.request?.tools ?? []) ||
          !equal(decision.live_exchange.gateway, exchange.reply?.raw_response?.vibestudio ?? null) ||
          decision.phase !== (exchange.request?.decision_reasoning !== null && typeof exchange.request?.decision_reasoning === 'object' && !Array.isArray(exchange.request.decision_reasoning)
            ? 'decision_reasoning' : exchange.kind === 'generation' ? 'action' : exchange.kind) ||
          !equal(decision.error, exchange.error ?? null))
        throw new Error('observed decision differs from the recorded model exchange');
      if (exchange.kind === 'generation' && !exchange.error &&
          (!decision.assistant || !Array.isArray(exchange.reply?.raw_calls) ||
           !equal(decision.assistant.raw_calls, exchange.reply.raw_calls) ||
           decision.assistant.content !== (exchange.reply.text ?? '') ||
           decision.assistant.reasoning !== exchange.reply.raw_response.vibestudio.response.content
             .filter(block => block.type === 'thinking' && !block.redacted).map(block => block.thinking).join('\n') ||
           !equal(decision.assistant.calls, exchange.reply.raw_response.vibestudio.response.content
             .filter(block => block.type === 'toolCall')
             .map(call => ({id: call.id, source_tool: call.name, arguments: call.arguments})))))
        throw new Error('observed decision target differs from the model reply');
    }
    captureScopes.delete(call.call_id);
    callFiles.delete(call.artifact.path);
    decisions.delete(call.call_id);
  }
  if(matchedDecisionSelectors.size!==decisionSelectors.length || !equal(row.outcome.excluded_review_decisions??[],decisionSelectors))throw new Error('decision review names an absent captured generation or differs from outcome');
  if (captureScopes.size || callFiles.size || decisions.size || roots.size)
    throw new Error('observed trajectory does not cover the full verified call tree');
  if (returnSelectors.some((item,index)=>item.required!==false&&!matchedReturnSelectors.has(index)))throw new Error('incorrect return binding does not match its actual original invocation');
  if ((selected && [...selected.keys()].some(id => !capturedRecords.has(id))) ||
      row.outcome.excluded_return_decisions !== undefined && !equal(row.outcome.excluded_return_decisions, returnExclusions))
    throw new Error('observed decision admission differs from independent training scope or return evidence');
  for(const invocation of receipt.training_scope?.invocations??[])await verifyNativeInvocationCertificate(invocation,capturedRecords,receipt.training_scope,root,capturedDetails);
  const applicationEntry=row.task.interaction?.entry;
  if(applicationEntry?.resultType==='void' &&
      (capturedRecords.get(receipt.root_call_ids[0])?.outcome === 'done' || applicationReturns.length > 0) &&
      (applicationReturns.length!==1 || applicationReturns[0].kind!=='void'))
    throw new Error('void application entry needs its explicit actual return witness');
  for(const binding of applicationReturns)verifyApplicationReturn(binding,capturedRecords,receipt.root_call_ids,applicationEntry);
  verifyObservedCallTree(records, receipt.root_call_ids[0]);
  for (const decision of row.trajectory) {
    const chain = [];
    let record = capturedRecords.get(decision.invocation_id);
    while (record) { chain.unshift(record.call_id); record = capturedRecords.get(record.parent_call_id); }
    if (!equal(chain, decision.live_exchange.call_ancestry) || decision.live_exchange.root_call_id !== chain[0] ||
        decision.live_exchange.target_source !== 'actual-model-reply' || decision.live_exchange.student_sampling_claimed !== false)
      throw new Error('observed decision ancestry differs from the complete captured tree');
  }
}
