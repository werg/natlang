import { adaptedCase, requireValue, equal, sha, evalCall, returnCall } from './common.mjs';
import { nlFile } from '../inline-curriculum/lib.mjs';

const WORDS = {one:1,two:2,three:3,four:4,first:1,second:2,third:3,fourth:4,fifth:5,sixth:6,seventh:7};
const COLORS = {red:'r',green:'g',orange:'o',yellow:'y',brown:'b',purple:'p'};
function alchemyState(text) {
  const fields=text.split(' ');
  requireValue(fields.length===7 && fields.every((field,i)=>new RegExp(`^${i+1}:(?:_|[rgoybp]{1,4})$`).test(field)), 'invalid_alchemy_state');
  return fields.map(field=>field.split(':')[1].replace('_',''));
}
function drain(before, text) {
  const cups=alchemyState(before);
  const match=/^(?:throw out|drain|discard|remove) (?:(one|two|three|four|\d+) units?(?: of| from)? )?(?:the )?(.+?)(?:\.)?$/.exec(text.toLowerCase());
  requireValue(match, 'unsupported_alchemy_action');
  const target=match[2].replace(/^(?:contents of (?:the )?)/,'').replace(/(?: beaker| one)$/,'');
  let index;
  if (/^(?:first|second|third|fourth|fifth|sixth|seventh)$/.test(target)) index=WORDS[target]-1;
  else if (COLORS[target]) {
    const candidates=cups.flatMap((value,i)=>value && [...value].every(c=>c===COLORS[target])?[i]:[]);
    requireValue(candidates.length===1, 'ambiguous_alchemy_color'); index=candidates[0];
  } else throw new Error('unresolved_alchemy_reference');
  const amount=match[1] ? WORDS[match[1]] ?? Number(match[1]) : cups[index].length;
  requireValue(Number.isSafeInteger(amount) && amount>0 && amount<=cups[index].length, 'invalid_alchemy_drain');
  cups[index]=cups[index].slice(0,-amount);
  return cups.map((cup,i)=>`${i+1}:${cup||'_'}`).join(' ');
}
export function validateStateChain(original) {
  const sem=original.semantics;
  requireValue(Array.isArray(sem?.steps) && sem.steps.length>0, 'empty_state_sequence');
  let current=sem.initial_state;
  for (const step of sem.steps) {
    requireValue(equal(current,step.before), 'broken_state_chain');
    requireValue(typeof step.utterance==='string' && step.utterance.trim() && step.after!==undefined && step.after!=='?', 'missing_state_annotation');
    current=step.after;
  }
  return current;
}

export function stateSequence(original, info, {sourceVerified=false, schema=null}={}) {
  const final=validateStateChain(original), sem=original.semantics;
  let quality='eligible', checks=['state_chain_continuity'], context=sem.task_context ?? '';
  if (!sourceVerified) {quality='held'; checks.push('missing_raw_annotation_join');}
  if (info.source==='SCONE') {
    if (sem.domain==='alchemy') {
      context='Seven beakers are encoded left to right as position:contents. _ means empty. Each letter is one unit, with r red, g green, o orange, y yellow, p purple, b brown. Drain removes the stated number of units, or all units when no amount is given. Preserve other beakers and renumber none. Resolve ordinal references from left to right.';
      try {
        for (const step of sem.steps) requireValue(equal(drain(step.before,step.utterance),step.after), 'alchemy_annotation_mismatch');
        checks.push('independent_drain_transition_oracle');
      } catch(error) {quality='held';checks.push(error.message);}
    } else {
      quality='held';
      checks.push(sem.domain==='scene'?'scene_annotations_shifted_or_unverified':'missing_independent_tangram_oracle');
      context=`Domain ${sem.domain}. State is an ordered position:value list. Preserve every unaffected object. Source annotations require review before training.`;
    }
  } else {
    requireValue(schema && context, 'missing_dialogue_schema');
    checks.push('original_service_frames_verified','service_schema_preserved');
    context += '\nRequested slots refer to the current request; persistent slot values carry forward unless explicitly changed. Return the full service-local state. Do not clear other slots merely because they were not mentioned again. Copy explicitly stated slot values without adding inferred defaults.';
    const slotNames=new Set(schema.slots.map(slot=>slot.name));
    const intentNames=new Set(['NONE',...schema.intents.map(intent=>intent.name)]);
    let history='';
    for (const step of sem.steps) {
      history += '\n'+step.utterance;
      const state=step.after;
      requireValue(state && intentNames.has(state.active_intent) && Array.isArray(state.requested_slots)
        && new Set(state.requested_slots).size===state.requested_slots.length
        && state.requested_slots.every(slot=>slotNames.has(slot)) && state.slot_values && typeof state.slot_values==='object', 'invalid_dialogue_state');
      for (const [slot,values] of Object.entries(state.slot_values)) {
        requireValue(slotNames.has(slot)&&Array.isArray(values)&&values.length>0&&values.every(v=>typeof v==='string'), 'invalid_dialogue_slot');
        if (values.length!==1 || !history.toLowerCase().includes(values[0].toLowerCase())) {
          quality='held'; checks.push(values.length!==1?'ambiguous_slot_alternates':'unstated_slot_normalization_or_default');
        }
      }
    }
  }
  const stateType=info.source==='SCONE'?'string':'{ active_intent:string, requested_slots:string[], slot_values:Record<string,string[]> }';
  const visible=sem.steps.map((step,index)=>({id:`transition:${sha([original.id,index]).slice(0,20)}`,text:step.utterance}));
  const record=adaptedCase(original,info,{family:info.source==='SCONE'?'scone':'dialogue_state',
    root:{name:'apply_instructions',args:{initial:'State',instructions:'Instruction[]'},returns:'State',
      instructions:'Starting at initial, apply the instructions in order. Give each transition the complete current state and prior instruction history. Return the final state. Preserve unaffected state.'},
    files:{'types.ts':`export type State = ${stateType};\nexport type Instruction = {id:string,text:string};\n`,
      'apply_instructions/transition.nl':nlFile({args:{state:'State',step:'Instruction',history:'string'},returns:'State',
        instructions:`Apply step.text to state using history to resolve earlier references. ${context}`})},
    inputs:{initial:sem.initial_state,instructions:visible}, expected:final,
    actions:[evalCall('let current = initial;\nfor (let index=0; index<instructions.length; index++) {\n  current = await transition(current, instructions[index], instructions.slice(0,index).map(step=>step.text).join("\\n"));\n}\nreturn current;'),returnCall(final)],
    children:sem.steps.map((step,index)=>({match:JSON.stringify(visible[index].id),value:step.after})),
    quality,checks:[...new Set(checks)]});
  record.external_source.original_row=original;
  if(schema) record.external_source.service_schema=schema;
  return record;
}

/** Join service-specific sequences to original frames, retaining every intervening turn. */
export function verifyDialogue(original, dialogue, schema) {
  const service=original.semantics.domain;
  requireValue(dialogue.dialogue_id===original.source_ids[0] && dialogue.services.includes(service) && schema.service_name===service, 'dialogue_source_join_mismatch');
  let state={active_intent:'NONE', requested_slots:[], slot_values:{}}, cursor=0;
  const steps=[];
  for (const [index,turn] of dialogue.turns.entries()) {
    if (turn.speaker!=='USER') continue;
    const frame=turn.frames.find(frame=>frame.service===service);
    if (!frame?.state) continue;
    steps.push({before:state, after:frame.state, utterance:dialogue.turns.slice(cursor,index+1).map(t=>`${t.speaker}: ${t.utterance}`).join('\n')});
    state=frame.state;cursor=index+1;
  }
  requireValue(equal(steps,original.semantics.steps), 'dialogue_annotation_join_mismatch');
  return true;
}
